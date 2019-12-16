import torch
import torch.nn.functional as F
import torch.optim as optim

from env.Environment import Environment
from env.envs import create_atari_env
from models.model import ActorCritic


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, counter, lock, optimizer=None):
    torch.manual_seed(args.seed + rank)

    if args.play_sf:
        roms_path = args.roms  # Replace this with the path to your ROMs
        env = Environment("env"+str(rank), roms_path,frame_ratio =3,frames_per_step = 1,throttle =False)
        model = ActorCritic(3, 9*10)
        env.start()
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
    done = True

    episode_length = 0
    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:#game done
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            print('step',step)
            episode_length += 1
            print(state.unsqueeze(0).shape)
            print(state.unsqueeze(0).dtype)
            
            value, logit, (hx, cx) = model((state.float().unsqueeze(0),
                                            (hx, cx)))
            prob = F.softmax(logit, dim=-1) # Or I just make some the input , Or use another output way
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            if args.play_sf:
                action_id = action.numpy()[0,0]
                move_action, attack_action = action_id//10,action_id%10
                state, reward, round_done, stage_done, done = env.step(move_action, attack_action)
                state = state.T
                reward = reward['P1']
                if done:
                    env.new_game()
                if stage_done:
                    env.next_stage()
                if round_done:
                    env.next_round()
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
                    reward = reward['P1']
                else:
                    state = env.reset()
            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                    break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.float().unsqueeze(0), (hx, cx)))
            r = value.detach()


        values.append(r)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
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

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
