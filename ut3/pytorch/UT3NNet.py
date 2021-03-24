import sys
sys.path.append('..')
from utils import *

import argparse
import torch
from torch import nn
import torch.nn.functional as F
# from keras.models import *
# from keras.layers import *
# from keras.optimizers import *

"""
NeuralNet for the game of TicTacToe.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloNNet by SourKream and Surag Nair.
"""
class UT3NNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.size = game.getBoardSize()
        self.channels = game.getBoardChannels()
        self.action_size = game.getActionSize()
        self.args = args      
        super(UT3NNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=self.channels,
                               out_channels=args.num_channels,
                               kernel_size=(3,3),
                               stride=3)
        self.conv2 = nn.Conv2d(in_channels=args.num_channels,
                               out_channels=args.num_channels,
                               kernel_size=3,
                               stride=1)
        # self.conv3 = nn.Conv2d(in_channels=args.num_channels,
        #                        out_channels=args.num_channels,
        #                        kernel_size=3,
        #                        stride=1)
        # self.conv4 = nn.Conv2d(in_channels=args.num_channels,
        #                        out_channels=args.num_channels,
        #                        kernel_size=3,
        #                        stride=1)
        
        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        # self.bn3 = nn.BatchNorm2d(args.num_channels)
        # self.bn4 = nn.BatchNorm2d(args.num_channels)


        self.fc1 = nn.Linear(args.num_channels, args.linear_hiddens)
        #self.fc1 = nn.Linear(args.num_channels*(self.size[0]-4)*(self.size[1]-4), args.linear_hiddens)
        self.fc_bn1 = nn.BatchNorm1d(args.linear_hiddens)

        self.fc2 = nn.Linear(args.linear_hiddens, args.linear_hiddens)
        self.fc_bn2 = nn.BatchNorm1d(args.linear_hiddens)

        self.fc3 = nn.Linear(args.linear_hiddens, self.action_size)

        self.fc4 = nn.Linear(args.linear_hiddens, 1)
        
    def forward(self, xin):
        x = F.relu(self.bn1(self.conv1(xin)))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))                         
        # x = F.relu(self.bn4(self.conv4(x)))

        x = torch.flatten(x, start_dim=1)
        
        x = F.dropout(F.relu(self.fc_bn1(self.fc1(x))), p=self.args.dropout, training=self.training)
        x = F.dropout(F.relu(self.fc_bn2(self.fc2(x))), p=self.args.dropout, training=self.training)
        
        pi = self.fc3(x)                                                                         # batch_size x action_size
        v = self.fc4(x)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)


# class UT3NNet(nn.Module):
#     def __init__(self, game, args):
#         # game params
#         self.size = game.getBoardSize()
#         self.channels = game.getBoardChannels()
#         self.action_size = game.getActionSize()
#         self.args = args      
#         super(UT3NNet, self).__init__()
        
#         self.conv1 = nn.Conv2d(in_channels=self.channels,
#                                out_channels=args.num_channels,
#                                kernel_size=(3,3),
#                                stride=3)
#         self.conv2 = nn.Conv2d(in_channels=args.num_channels,
#                                out_channels=args.num_channels,
#                                kernel_size=3,
#                                stride=1)
        
#         self.bn1 = nn.BatchNorm2d(args.num_channels)
#         self.bn2 = nn.BatchNorm2d(args.num_channels)

#         self.fc1 = nn.Linear(args.num_channels, args.linear_hiddens)
#         self.fc_bn1 = nn.BatchNorm1d(args.linear_hiddens)

#         self.fc2 = nn.Linear(args.linear_hiddens, args.linear_hiddens)
#         self.fc_bn2 = nn.BatchNorm1d(args.linear_hiddens)

#         self.fc3 = nn.Linear(args.linear_hiddens, self.action_size)

#         self.fc4 = nn.Linear(args.linear_hiddens, 1)
        
#     def forward(self, xin):
#         x = F.relu(self.bn1(self.conv1(xin)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         # x = F.relu(self.bn3(self.conv3(x)))                         
#         # x = F.relu(self.bn4(self.conv4(x)))
#         # import pdb
#         # pdb.set_trace()

#         x = torch.flatten(x, start_dim=1)
        
#         x = F.relu(self.fc_bn1(self.fc1(x)))
#         x = F.relu(self.fc_bn2(self.fc2(x)))
        
#         pi = self.fc3(x)                                                                         # batch_size x action_size
#         v = self.fc4(x)                                                                          # batch_size x 1

#         return F.log_softmax(pi, dim=1), torch.tanh(v)
        
        
        
        
        
        
        
        
