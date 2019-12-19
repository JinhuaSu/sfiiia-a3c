import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from env.Environment import Environment
from env.envs import create_atari_env
from models.model import ActorCritic


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad.cpu()


def train(rank, args, shared_model,model, counter, lock, optimizer=None):
    writer = SummaryWriter(log_dir=args.log_path)
    torch.manual_seed(args.seed + rank)
    device = args.use_gpu[rank%len(args.use_gpu)] if len(args.use_gpu) > 0 else -1

    if args.play_sf:
        roms_path = args.roms  # Replace this with the path to your ROMs
        env = Environment("env"+str(rank), roms_path,difficulty=args.difficulty,frame_ratio =3,frames_per_step = 1,throttle =False)
        #model = ActorCritic(3, 9*10+17,device)
        env.start()
        time_loss_l = []
        time_count = torch.tensor(0)
    else:
        env = create_atari_env(args.env_name)
        env.seed(args.seed + rank)
        model = ActorCritic(env.observation_space.shape[0], env.action_space.n)


    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    if not args.play_sf:
        state = env.reset()
        state = torch.from_numpy(state)
    else:
        state, reward, round_done, stage_done, done = env.step(8, 9)
        state = torch.from_numpy(state.T)
        if device >=0:
            state = state.to(device)

    done = True

    episode_length = 0
    epoch = 0
    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:#game done
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
            if device >= 0:
                cx,hx = cx.to(device),hx.to(device)
        else:
            cx = cx.detach()
            hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            time_count += 1
            episode_length += 1
            
            value, logit, (hx, cx) = model((state.float().unsqueeze(0),
                                            (hx, cx)))
            prob = F.softmax(logit, dim=-1) # Or I just make some the input , Or use another output way
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            if args.play_sf:
                action_id = action.cpu().numpy()[0,0]
                if action_id < 90:
                    move_action, attack_action = action_id//10,action_id%10
                else:
                    move_action, attack_action = -1,action_id%90
                state, reward, round_done, stage_done, done = env.step(move_action, attack_action)
                state = state.T

                reward = reward[args.reward_mode]
                reward -= time_count % 60
                if done:
                    env.new_game()
                    time_count == 0
                    reward -= 200
                if stage_done:
                    env.next_stage()
                    time_count == 0
                    reward += 200
                if round_done:
                    env.next_round()
                    time_count == 0
            else:
                state, reward, done, _ = env.step(action.numpy())
                reward = max(min(reward, 1), -1)
            done = done or episode_length >= args.max_episode_length
            with lock:
                counter.value += 1

            if done:
                episode_length = 0
                if args.play_sf:
                    env.new_game()
                    state, reward, round_done, stage_done, _ = env.step(8, 9)
                    state = state.T
                    reward = reward[args.reward_mode]
                else:
                    state = env.reset()
            state = torch.from_numpy(state)
            if device >=0:
                state = state.to(device)
                reward = torch.tensor(reward)
                reward = reward.to(device)
                value = value.to(device)
                log_prob= log_prob.to(device)

            values.append(value)
            time_loss_l.append(time_count.to(device))
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                    break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.float().unsqueeze(0), (hx, cx)))
            r = value.detach()
        if device >=0:
            r = r.to(device)


        values.append(r)
        policy_loss = 0
        value_loss = 0
        time_loss = 0
        gae = torch.zeros(1, 1)
        if device >=0:
            gae=gae.to(device)
        for i in reversed(range(len(rewards))):
            r = args.gamma * r + rewards[i]
            advantage = r - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # generalized advantage estimation
            delta_t = rewards[i] + args.gamma * \
                values[i + 1] - values[i]
            gae = gae * args.gamma * args.gae_lambda + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

        optimizer.zero_grad()
        epoch += 1 
        writer.add_scalars(args.reward_mode+'/%sloss_group'%rank,\
                                                {'policy_loss':policy_loss,\
                                                'value_loss':value_loss,\
                                                'time_loss':time_loss,\
                                                'total_loss':policy_loss + args.value_loss_coef * value_loss
},epoch)
        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
