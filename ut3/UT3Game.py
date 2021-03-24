from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .UT3Logic import Board
import numpy as np

"""
Game class implementation for the game of Ultimate TicTacToe.

Author: Harrison Delecki, github.com/hdelecki

Based on the OthelloGame by Surag Nair.
"""
class UT3Game(Game):
    def __init__(self, n=3, conv=True):
        self.conv = conv
        self.n = n
        #self.last_move = None

    def getArray(self, board):
        if self.conv:
            global_rep = np.repeat(np.repeat(board.global_pieces, 3, axis=1), 3, axis=0)
            local_rep = board.local_pieces
            play_rep = np.repeat(np.repeat(board.play_map, 3, axis=1), 3, axis=0)
            #valid_rep = np.zeros(local_rep.shape)
            #0valids = board.get_legal_moves(player=1)
            #valid_rep[tuple(valids.T.tolist())] = 1.0
            return np.stack((local_rep, global_rep, play_rep))
        else:
            raise NotImplementedError()

    def getBoardChannels(self):
        #return 2
        if self.conv:
            return 3
        else:
            return 1

    def getInitBoard(self):
        # return initial board (numpy board)
        #self.last_move = None
        b = Board(self.n)
        return self.getArray(b)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n**2, self.n**2)

    def getActionSize(self):
        # return number of actions
        return self.n**4

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        # if action == self.n*self.n:
        #     return (board, -player)
        # b = Board(self.n)
        # b.pieces = np.copy(board)
        # move = (int(action/self.n), action%self.n)
        # b.execute_move(move, player)
        # return (b.pieces, -player)
        b = Board(self.n)
        b.local_pieces = np.copy(board[0])
        b.global_pieces = np.copy(board[1][::3, ::3])
        b.play_map = np.copy(board[2][::3, ::3])
        #b.last_move = self.last_move
        move = np.unravel_index(action, (self.n**2, self.n**2))
        #move = int(action/self.n**2), action%self.n**2
        b.execute_move(move, player)
        #self.last_move = b.last_move
        return self.getArray(b), -player

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        #valid = [0]*self.getActionSize()
        valid = np.zeros(self.getActionSize())
        b = Board(self.n)
        b.local_pieces = np.copy(board[0])
        b.global_pieces = np.copy(board[1][::3, ::3])
        b.play_map = np.copy(board[2][::3, ::3])
        valid_coords = b.get_legal_moves(player)
        valid_idx = np.ravel_multi_index(valid_coords.T, (self.n**2, self.n**2))
        valid[valid_idx] = True
        return valid


    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        brd = Board(self.n)
        brd.local_pieces = np.copy(board[0])
        brd.global_pieces = np.copy(board[1][::3, ::3])
        brd.play_map = np.copy(board[2][::3, ::3])
        
        if brd.is_win(1):
            return player
        elif brd.is_win(-1):
            return -player
        elif brd.is_full():
            return brd.draw

        # for player in -1, 1:
        #     if brd.is_win(player):
        #         return player
        #     if brd.is_full():
        #         return brd.draw
        return 0

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        #return np.where(board, player*board, board)
        if player == 1:
            return board
        else:
            board[:2,:,:] *= -1
            return board
        
    def getSymmetries(self, board, pi):
        # rotate, mirror
        assert(len(pi) == self.getActionSize())  # 1 for pass
        pi_board = np.reshape(pi, self.getBoardSize())
        sym, x, y = [], -2, -1
        
        # sym.append((board, pi))
        # return sym

        for rot in range(1, 5):
            for flip in True, False:
                newB = np.rot90(board, rot, (x, y))
                newPi = np.rot90(pi_board, rot, (x, y))
                if flip:
                    newB = np.flip(newB, y)
                    newPi = np.flip(newPi, y)
                sym.append((newB, list(newPi.ravel())))
        return sym

    def stringRepresentation(self, board):
        return board.tostring()


    def display(self, board, indent='  '):
        # print('Last Move:')
        # print(board.last_move)
        print('')
        print(indent + '   0 | 1 | 2 ‖ 3 | 4 | 5 ‖ 6 | 7 | 8')
        print('')
        for n, row in enumerate(board[0]):
            if n:
                if n % 3:
                    sep = '---+---+---'
                    print(indent + '- ' + sep + '‖' + sep + '‖' + sep)
                else:
                    sep = '==========='
                    print(indent + '= ' + sep + '#' + sep + '#' + sep)
            row = ' ‖ '.join(' | '.join(map(str, map(int, row[i:i+3]))) for i in range(0, len(row), 3))
            print(indent + str(n) + '  ' + row.replace('-1','O').replace('1','X').replace('0','.'))
        print('')

def display(board, indent='  '):
    # print('Last Move:')
    # print(board.last_move)
    print('')
    print(indent + '   0 | 1 | 2 ‖ 3 | 4 | 5 ‖ 6 | 7 | 8')
    print('')
    for n, row in enumerate(board[0]):
        if n:
            if n % 3:
                sep = '---+---+---'
                print(indent + '- ' + sep + '‖' + sep + '‖' + sep)
            else:
                sep = '==========='
                print(indent + '= ' + sep + '#' + sep + '#' + sep)
        row = ' ‖ '.join(' | '.join(map(str, map(int, row[i:i+3]))) for i in range(0, len(row), 3))
        print(indent + str(n) + '  ' + row.replace('-1','O').replace('1','X').replace('0','.'))
    print('')

    # @staticmethod
    # def display(board):
    #     n = board.shape[0]

    #     print("   ", end="")
    #     for y in range(n):
    #         print (y,"", end="")
    #     print("")
    #     print("  ", end="")
    #     for _ in range(n):
    #         print ("-", end="-")
    #     print("--")
    #     for y in range(n):
    #         print(y, "|",end="")    # print the row #
    #         for x in range(n):
    #             piece = board[y][x]    # get the piece to print
    #             if piece == -1: print("X ",end="")
    #             elif piece == 1: print("O ",end="")
    #             else:
    #                 if x==n:
    #                     print("-",end="")
    #                 else:
    #                     print("- ",end="")
    #         print("|")

    #     print("  ", end="")
    #     for _ in range(n):
    #         print ("-", end="-")
    #     print("--")
