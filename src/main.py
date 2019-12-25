from __future__ import print_function

import argparse
import os

import torch
import torch.multiprocessing

from models import my_optim
from env.envs import create_atari_env
from models.model import ActorCritic
from test import test
from train import train
class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:raise argparse.ArgumentTypeError('Boolean value expected.')
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
parser.add_argument('--num-steps', type=int, default=20*3,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=10000000,
                    help='maximum length of an episode (default: 10000000)')
parser.add_argument('--env-name', default='PongDeterministic-v4',
                    help='environment to train on (default: PongDeterministic-v4)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')
parser.add_argument('--use_gpu', default="",
                    help='use gpu to train something')
parser.add_argument('--play_sf', default=True,type=bool,
                    help='play sfiii3n')
parser.add_argument('--roms', default='/ssd/su/sfiiia-a3c/roms/',type=str,
                    help='dir roms')
parser.add_argument('--save_per_min', default=20,type=int,
                    help='train x minutes and save a checkpoint')
parser.add_argument('--model_path', default='../models/',type=str)

parser.add_argument('--mode', default='train',type=str,choices=['train','test','PvP'])

parser.add_argument('--test_from', default='',type=str)

parser.add_argument('--device', default='server',type=str,choices=['server','laptop'])

parser.add_argument('--log_path', default='../logs/',type=str)

parser.add_argument('--reward_mode', default='P1',type=str,choices=['P1','absolute_diff'])

parser.add_argument('--difficulty', default=5,type=int)

parser.add_argument('--log_freq', default=10,type=int)

parser.add_argument('--img_path', default='../images/',type=str)

parser.add_argument('--throttle', default=True,type=str2bool)

def multi_main(args):
    """ Spawns 1 process per GPU """
    init_logger()

    nb_gpu = args.world_size
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i
        procs.append(mp.Process(target=run, args=(args,
            device_id, error_queue,), daemon=True))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()



if __name__ == '__main__':
    args = parser.parse_args()
    if args.device == 'server':
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3" if args.use_gpu!=""else ""
        os.system("Xvfb :1 -screen 1 800x600x16 +extension RANDR &")
        os.environ["DISPLAY"] = ":1"
    torch.manual_seed(args.seed)
    args.use_gpu = [int(x) for x in args.use_gpu.split(",")]
    main_device = args.use_gpu[0] if len(args.use_gpu)>0 else -1 
    if args.play_sf:
        print('Play sfiii3n!')
        shared_model = ActorCritic(3, 9*10+17,-1)
        if args.test_from != "":
            print('load model from %s'%args.test_from)
            shared_model.load_state_dict(torch.load(args.test_from))
    else:
        env = create_atari_env(args.env_name)
        shared_model = ActorCritic(
        env.observation_space.shape[0], env.action_space.n)
    shared_model.share_memory()

    if args.img_path != '' and os.path.exists(args.img_path) is False:
        os.makedirs(args.img_path)


    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()
        
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)


    processes = []

    counter = mp.Value('i', 0) #pytorch mutliprocess tool
    lock = mp.Lock()
    
    test_model = ActorCritic(3, 9*10+17,main_device)
    p = mp.Process(target=test, args=(args.num_processes, args, shared_model,test_model, counter), daemon=True)
    p.start()
    processes.append(p)
    print(" Starting process pid: %d  " % processes[0].pid)
    error_handler.add_child(processes[0].pid)

    if args.mode == 'train':
        for rank in range(0, args.num_processes):
            device = args.use_gpu[rank%len(args.use_gpu)] if len(args.use_gpu) > 0 else -1
            worker_model = ActorCritic(3, 9*10+17,device)
            p = mp.Process(target=train, args=(rank, args, shared_model,worker_model, counter, lock, optimizer), daemon=True)
            p.start()
            processes.append(p)
            print(" Starting process pid: %d  " % processes[rank+1].pid)
            error_handler.add_child(processes[rank+1].pid)
    for p in processes:
        p.join()
