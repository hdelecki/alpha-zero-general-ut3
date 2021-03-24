import numpy as np
'''
Board class for the game of Ultimate Tic Tac Toe.
Default board size is 3x3.
Board data:
  1=white(O), -1=black(X), 0=empty
  first dim is column , 2nd is row:

Author: Harrison Delecki, github.com/hdelecki

Based on the board for the game of Othello by Eric P. Nichols.

'''
# from bkcharts.attributes import color
class Board():

    def __init__(self, n=3):
        "Set up initial board configuration."
        self.n = n
        self.global_pieces = np.zeros((n,n), dtype=np.float32)
        self.local_pieces = np.zeros((n**2, n**2), dtype=np.float32)
        self.play_map = np.ones((n,n), dtype=np.float32)
        self.draw = np.finfo(self.local_pieces.dtype).eps

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.local_pieces[index]


    def get_local_game(self, coords):
        xl = coords[0]
        yl = coords[1]
        return self.local_pieces[self.n*xl:self.n*xl+self.n, self.n*yl:self.n*yl+self.n]

    def local_idx_in_global(self, local_idx):
        return np.array([int(local_idx[0]/self.n), int(local_idx[1]/self.n)])
    
    def get_move_in_local(self, move):
        return move%self.n

    def get_legal_moves(self, player):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black)
        @param color not used and came from previous version.        
        """
        play_locs = np.repeat(np.repeat(self.play_map, 3, axis=1), 3, axis=0)
        return np.transpose(np.where((self.local_pieces==0.0) & (play_locs==1.0)))


    def has_legal_moves(self):
        return np.any(self.local_pieces==0.0)

    def is_full(self, board=None):
        if board is None: board = self.global_pieces
        return board.all()
    
    def is_win(self, player, board=None):
        """Check whether the given player has collected a triplet in any direction; 
        @param color (1=white,-1=black)
        """
        if board is None: board = self.global_pieces
        for i in range(self.n):
            if (board[i, :]==player).all() or (board[:, i]==player).all(): return True
        if (board.diagonal() == player).all() or (board[::-1].diagonal() == player).all(): return True
        return False

    def execute_move(self, move, player):
        """Perform the given move on the board; 
        color gives the color pf the piece to play (1=white,-1=black)
        """

        (x,y) = move

        # Add the piece to the empty square.
        assert self.local_pieces[move] == 0
        self.local_pieces[move] = player
        idx_of_game = self.local_idx_in_global(move)
        local_board = self.get_local_game(idx_of_game)

        if self.is_full(board=local_board):
            self.global_pieces[idx_of_game[0], idx_of_game[1]] = self.draw

        for player in [-1, 1]:
            if self.is_win(player, local_board):
                self.global_pieces[idx_of_game[0], idx_of_game[1]] = player
                valid_idxs = self.get_legal_moves(player)
                for idx in valid_idxs:
                    self.local_pieces[idx[0], idx[1]] == self.draw
                
        # Update last move
        coords = self.get_move_in_local(np.array(move))
        next_board = self.get_local_game(coords)
        if self.is_full(next_board) or np.abs(self.global_pieces[coords[0], coords[1]]) == 1:
            self.play_map = np.ones((self.n, self.n), dtype=np.float32)
            self.play_map[self.global_pieces>0.0] = 0.0
        else:
            self.play_map = np.zeros((self.n, self.n), dtype=np.float32)
            self.play_map[coords[0], coords[1]] = 1.0


