import os
import argparse

import torch.multiprocessing as mp

class MultiTrainer():
    def __init__(self,func,master_host='127.0.0.1',master_port='8888'):

        self.func = func  

        parser = argparse.ArgumentParser()
        parser.add_argument('-n', '--nodes', default=1,
                            type=int, metavar='N')
        parser.add_argument('-g', '--gpus', default=1, type=int,
                            help='number of gpus per node')
        parser.add_argument('-nr', '--nr', default=0, type=int,
                            help='ranking within the nodes')
        parser.add_argument('-mh', '--mhost', default=master_host, type=str,
                            help='master node host')
        parser.add_argument('-mp', '--mport', default=master_port, type=str,
                            help='master node port')
        # parser.add_argument('--epochs', default=10, type=int, 
        #                     metavar='N',
        #                     help='number of total epochs to run')
        self.args = parser.parse_args()
        self.args.world_size = self.args.gpus * self.args.nodes    

        os.environ['MASTER_ADDR'] = self.args.mhost            
        os.environ['MASTER_PORT'] = self.args.mport                     

    def spawn(self):
        mp.spawn(self.func, nprocs=self.args.gpus, args=(self.args,))