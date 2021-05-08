import random
import numpy as np
import random

LAYER = 6
ROW = 6
COLUMN = 6


class Game:
    def __init__(
        self,
        layers, rows, cols,
        sentinel=-1, empty=0, player_one=1, player_two=2
    ):
        # Board
        self.sentinel = sentinel
        self.empty = empty
        self.layers = layers
        self.rows = rows
        self.cols = cols
        self.rods = self.create_rods()
        self.board = self.create_empty_board()
        # Player
        self.player_one = player_one
        self.player_two = player_two
        # Score
        self.scores = {player_one: 0, player_two: 0}
        self.potential = {player_one: 0, player_two: 0}
        self.kth_line = []
        # Hash
        self.table = self.create_table()


    def create_rods(self):
        '''
        Create valid rod positions as logical array
        '''
        rods = np.ones((self.rows, self.cols), dtype=bool)
        rods[0, [0, 1, self.cols-2, self.cols-1]] = False
        rods[1, [0, self.cols-1]] = False
        rods[self.rows-1, [0, 1, self.cols-2, self.cols-1]] = False
        rods[self.rows-2, [0, self.cols-1]] = False
        return rods


    def create_empty_board(self):
        '''
        Create an empty board and populate invalid spaces
        '''
        board = np.full((self.layers, self.rows, self.cols), self.empty, dtype=int)
        board[:, ~self.rods] = self.sentinel
        return board


    def create_table(self):
        '''
        Create a lookup table for zobrist hashing
        '''
        return np.random.randint(2147483647, 9223372036854775807, size=(self.layers, self.rows, self.cols, 2), dtype=np.uint64)


    def generate_moves(self):
        '''
        Generate all possible moves from current position as array of tuples
        '''
        rows, cols = np.where(self.rods)
        valid_rods = [self.empty in self.board[:, r, c] for (r, c) in zip(rows, cols)]
        return list(map(tuple, np.array(list(zip(rows, cols)))[valid_rods]))


    def drop_piece(self, r, c, piece):
        '''
        Drop a piece and update relavant data
        '''
        rod = self.board[:, r, c]
        if self.empty in rod:
            free_layer = np.nonzero(rod == 0)[0][0]
            rod[free_layer] = piece
            self.update_score(free_layer, r, c, piece)
        else:
            print("Invalid Move")


    def infer_move(self, board):
        '''
        Tries to infer the moves that lead to the given position
        '''
        difference = self.board != board
        layers, rows, cols = np.where(difference)
        order = np.argsort(layers)
        layers = layers[order]
        rows = rows[order]
        cols = cols[order]
        for (i, e) in enumerate(board[layers, rows, cols]):
            self.drop_piece(
                rows[i],
                cols[i],
                e
            )


    def update_score(self, l, r, c, piece):
        '''
        Updates scores count and k, should be called after every move.
        '''
        directions = np.array([ # there should be 13 directions (3*3*3 - 1) / 2
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [0, 1, -1],
            [1, 0, -1],
            [1, -1, 0],
            [1, 1, -1],
            [1, -1, 1],
            [-1, 1, 1],
            [1, 1, 1]])
        for dir in directions:
            segment = self.get_segment(l, r, c, dir)
            for i in range(segment.size-3):
                oppo_piece = self.player_one if piece == self.player_two else self.player_two
                self.evaluate_window(segment[i:i+4].tolist(), piece, oppo_piece)
        

    def get_segment(self, l, r, c, dir):
        '''Get the segment along direction `dir` fixed at (l, r, c)'''

        # Calculate the limit along each dimension
        limits = np.concatenate((
            [self.layers - l - 1, 0 - l][::dir[0]] / dir[0] if dir[0] else [],
            [self.layers - r - 1, 0 - r][::dir[1]] / dir[1] if dir[1] else [],
            [self.layers - c - 1, 0 - c][::dir[2]] / dir[2] if dir[2] else [],
            [3, -3]  # only reach until 3 piece away
        )).astype(int)

        # Get maximum reach in the direction(two sided)
        start = max(limits[1::2], default=0)
        end = min(limits[0::2], default=0)

        # Get segment
        return self.board[
                [l + dir[0] * j for j in range(start, end+1)],
                [r + dir[1] * j for j in range(start, end+1)],
                [c + dir[2] * j for j in range(start, end+1)]]


    def evaluate_window(self, window, player, opponent):
        ''' All the score values are hard coded here '''
        if window.count(player) == 4:
            self.kth_line.append(player)
            self.scores[player] += (100 // len(self.kth_line))
            self.potential[player] -= 5
        elif window.count(player) == 3 and window.count(self.empty) == 1:
            self.potential[player] += 3
        elif window.count(player) == 2 and window.count(self.empty) == 2:
            self.potential[player] += 2


    def zobrist_hash(self):
        # Score
        hash = np.uint64(sum((e-1) * (1<<i) for (i, e) in enumerate(self.kth_line)))
        # Pieces
        for l in range(self.layers):
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.board[l, i, j] > 0:
                        hash ^= self.table[l, i, j, self.board[l, i, j]-1]
        return hash


    def print_board(self):
        for l in range(self.layers):
            print('Layer ', l)
            for i in range(self.rows):
                symbol_map = lambda e: {
                    self.sentinel: ' ',
                    self.empty: '-',
                    self.player_one: 'o',
                    self.player_two: 'x'
                }[e]
                print("".join(map(symbol_map, self.board[l][i])))


board1 = Game(LAYER, ROW, COLUMN)
board2 = Game(LAYER, ROW, COLUMN)

board2.drop_piece(2, 2, 1)
board2.drop_piece(2, 2, 1)
board2.drop_piece(2, 2, 1)
board2.drop_piece(2, 2, 1)

board1.infer_move(board2.board)
print(board1.scores)