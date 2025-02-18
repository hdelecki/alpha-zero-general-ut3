import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
from tqdm import tqdm
sys.path.append('..')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim

import argparse
from .UT3NNet import UT3NNet as unet
from .LinearUT3NNet import LinearUT3NNet as linunet

"""
NeuralNet wrapper class for the game of Ultimate TicTacToe.

Author: Evgeny Tyurin, github.com/hdelecki

Based on (copy-pasted from) the NNet by SourKream and Surag Nair.
"""

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,
    'linear_hiddens': 512,
    'conv': True,
})
class NNetWrapper(NeuralNet):
    def __init__(self, game):
        #self.conv = args.conv
        if args.conv:
            self.nnet = unet(game, args)
        else:
            self.nnet = linunet(game, args)
            
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(args.epochs):
            #print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                
                if not args.conv:
                    new_boards = []
                    for b in boards:
                        l = b[0].flatten()
                        g = b[1][::3, ::3].flatten()
                        p = b[2][::3, ::3].flatten()
                        new_boards.append(np.concatenate((l,g,p)))
                    boards = tuple(new_boards)
                
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()
        
        if not args.conv:
            l = board[0].flatten()
            g = board[1][::3, ::3].flatten()
            p = board[2][::3, ::3].flatten()
            #new_boards.append(np.concatenate((l,g,p)))
            board = np.concatenate((l,g,p))

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda: board = board.contiguous().cuda()
        #board = board.view(1, self.board_x, self.board_y)
        board = board.unsqueeze(0)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
# class NNetWrapper(NeuralNet):
#     def __init__(self, game):
#         self.nnet = unet(game, args)
#         self.board_x, self.board_y = game.getBoardSize()
#         self.action_size = game.getActionSize()

#     def train(self, examples):
#         """
#         examples: list of examples, each example is of form (board, pi, v)
#         """
#         input_boards, target_pis, target_vs = list(zip(*examples))
#         input_boards = np.asarray(input_boards)
#         target_pis = np.asarray(target_pis)
#         target_vs = np.asarray(target_vs)
#         self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = args.batch_size, epochs = args.epochs)

#     def predict(self, board):
#         """
#         board: np array with board
#         """
#         # timing
#         start = time.time()

#         # preparing input
#         board = board[np.newaxis, :, :]

#         # run
#         pi, v = self.nnet.model.predict(board)

#         #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
#         return pi[0], v[0]

#     def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
#         filepath = os.path.join(folder, filename)
#         if not os.path.exists(folder):
#             print("Checkpoint Directory does not exist! Making directory {}".format(folder))
#             os.mkdir(folder)
#         else:
#             print("Checkpoint Directory exists! ")
#         self.nnet.model.save_weights(filepath)

#     def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
#         # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
#         filepath = os.path.join(folder, filename)
#         if not os.path.exists(filepath):
#             raise("No model in path '{}'".format(filepath))
#         self.nnet.model.load_weights(filepath)
