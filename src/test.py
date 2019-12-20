import time
from collections import deque

import torch
import torch.nn.functional as F

from env.envs import create_atari_env
from models.model import ActorCritic
from env.Environment import Environment

#I must get totally known about the whole structure of the model. And I should have a taste or VITA tea to control my baozou feelings

def test(rank, args, shared_model,model, counter):
    torch.manual_seed(args.seed + rank)

    device = args.use_gpu[rank%len(args.use_gpu)] if len(args.use_gpu) > 0 else -1
    device = 0
    if args.play_sf:
        roms_path = args.roms  # Replace this with the path to your ROMs
        if args.mode == 'PvP':
            print('PvP throttle:%s'%args.throttle)
            env = Environment("env"+str(rank), roms_path,difficulty=args.difficulty,frame_ratio =3,frames_per_step = 1,throttle =args.throttle)
        else:
            env = Environment("env"+str(rank), roms_path,difficulty=args.difficulty,frame_ratio =3,frames_per_step = 1,throttle =False)
        env.start()
        state, reward, round_done, stage_done, done = env.step(8, 9)
        state = state.T
    else:
        env = create_atari_env(args.env_name)
        env.seed(args.seed + rank)
        model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
        state = env.reset()
    model.eval()
    state = torch.from_numpy(state)
    if device >=0:
        state = state.to(device)
    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    step = 0
    while True:
        # Sync with the shared model
        if done: 
            num_stage = 0
            print('test start!')
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 1024)
            hx = torch.zeros(1, 1024)
            if device >=0:
                cx = cx.to(device)
                hx = hx.to(device)
        else:
            cx = cx.detach()
            hx = hx.detach()
        episode_length += 1
        with torch.no_grad():
            value, logit, (hx, cx) = model((state.float().unsqueeze(0), (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.multinomial(num_samples=1).detach()
        #action = prob.max(1, keepdim=True)[1].cpu().numpy()
        if args.play_sf:
            action_id = action.cpu().numpy()[0,0]
            if action_id < 90:
                move_action, attack_action = action_id//10,action_id%10
            else:
                move_action, attack_action = -1,action_id%90
            state, reward, round_done, stage_done, done = env.step(move_action, attack_action)
            reward = reward[args.reward_mode]
            state = state.T
            if done:
                env.new_game()
            if stage_done:
                num_stage  += 1
                env.next_stage()
            if round_done:
                env.next_round()
        else:
            state, reward, done, _ = env.step(action[0, 0])
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        #if args.mode == 'train' and actions.count(actions[0]) == actions.maxlen:
        #    done = True
        if done:
            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}, win_stage_num {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length,num_stage))
            step += 1
            if args.mode == 'train' and step % args.save_per_min == 0:
                print('saving model params at step %s' % step)
                torch.save(shared_model.state_dict(),'%s/model_params_step_%s.pkl' %(args.model_path+args.reward_mode,step))
            reward_sum = 0
            episode_length = 0
            actions.clear()
            if args.play_sf:
                env.new_game()
                state, reward, _, _, _ = env.step(8, 9)
                state = state.T
                reward = reward[args.reward_mode]
            else:
                state = env.reset()
            if args.mode == 'train':
                print('test mode sleep for 60 seconds')
                time.sleep(60)
        state =  torch.from_numpy(state)
        
        if device >=0:
            state = state.to(device)
            reward = torch.tensor(reward)
            reward = reward.to(device)
