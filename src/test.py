import time
from collections import deque

import torch
import torch.nn.functional as F

from env.envs import create_atari_env
from models.model import ActorCritic
from env.Environment import Environment

#I must get totally known about the whole structure of the model. And I should have a taste or VITA tea to control my baozou feelings

def test(rank, args, shared_model, counter):
    torch.manual_seed(args.seed + rank)

    if args.play_sf:
        roms_path = args.roms  # Replace this with the path to your ROMs
        env = Environment("env"+str(rank), roms_path,frame_ratio =3,frames_per_step = 1,throttle =False)
        model = ActorCritic(3, 9*10)
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
    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    step = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        if done: 
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, logit, (hx, cx) = model((state.float().unsqueeze(0), (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()
        if args.play_sf:
            action_id = action[0,0]
            move_action, attack_action = action_id//10,action_id%10
            state, reward, round_done, stage_done, done = env.step(move_action, attack_action)
            reward = reward['P1']
            state = state.T
            if stage_done:
                env.next_stage()
            elif round_done:
                env.next_round()
        else:
            state, reward, done, _ = env.step(action[0, 0])
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True
        if done:
            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length))
            step += 1
        if step % args.save_per_min == 0:
            print('saving model params at step %s' % step)
            torch.save(shared_model.state_dict(),'%smodel_params_step_%s.pkl' %(args.model_path,step))
            reward_sum = 0
            episode_length = 0
            actions.clear()
            if args.play_sf:
                env.new_game()
                state, reward, round_done, stage_done, done = env.step(8, 9)
                reward = torch.tensor(reward['P1'],requires_grad =True)
            else:
                state = env.reset()
            time.sleep(60)

        state =  torch.from_numpy(state)
