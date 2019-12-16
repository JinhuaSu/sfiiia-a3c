from __future__ import print_function

import argparse
import os

import torch
import torch.multiprocessing as mp

from models import my_optim
from env.envs import create_atari_env
from models.model import ActorCritic
from test import test
from train import train

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--gae-lambda', type=float, default=1.00,
                    help='lambda parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=4,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='PongDeterministic-v4',
                    help='environment to train on (default: PongDeterministic-v4)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')
parser.add_argument('--use_gpu', default=False,
                    help='use gpu to train something')
parser.add_argument('--play_sf', default=True,type=bool,
                    help='play sfiii3n')
parser.add_argument('--roms', default='/ssd/su/sfiiia-a3c/roms/',type=str,
                    help='dir roms')
parser.add_argument('--save_per_min', default=5,type=int,
                    help='train x minutes and save a checkpoint')
parser.add_argument('--model_path', default='../models/',type=str)

parser.add_argument('--mode', default='train',type=str,choices=['train','test','PvP'])

parser.add_argument('--test_from', default='',type=str)

parser.add_argument('--device', default='server',type=str,choices=['server','laptop'])

parser.add_argument('--log_path', default='../logs/',type=str)

parser.add_argument('--reward_mode', default='baseline',type=str,choices=['baseline'])

parser.add_argument('--difficulty', default=5,type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.device == 'server':
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3" if args.use_gpu else ""
        os.system("Xvfb :1 -screen 1 800x600x16 +extension RANDR &")
        os.environ["DISPLAY"] = ":1"
    torch.manual_seed(args.seed)
    if args.play_sf:
        print('Play sfiii3n!')
        shared_model = ActorCritic(3, 9*10)
        if args.test_from != "":
            shared_model.load_state_dict(torch.load(args.test_from))
    else:
        env = create_atari_env(args.env_name)
        shared_model = ActorCritic(
        env.observation_space.shape[0], env.action_space.n)
    shared_model.share_memory()


    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    processes = []

    counter = mp.Value('i', 0) #pytorch mutliprocess tool
    lock = mp.Lock()

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter))
    p.start()
    processes.append(p)

    if args.mode == 'train':
        for rank in range(0, args.num_processes):
            p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer))
            p.start()
            processes.append(p)
    for p in processes:
        p.join()
