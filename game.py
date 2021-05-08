import random
import numpy as np
from copy import deepcopy
import STcpClient

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
        self.dropped_pieces = 0
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


    def get_state(self):
        return deepcopy((self.board, self.scores, self.potential, self.kth_line))


    def set_state(self, board, scores, potential, kth_line):
        self.board = deepcopy(board)
        self.scores = deepcopy(scores)
        self.potential = deepcopy(potential)
        self.kth_line = deepcopy(kth_line)


    def is_terminal(self):
        return self.dropped_pieces == 64  # 32 piece for each side


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


    def generate_moves(self, *, player=None):
        '''
        Generate all possible moves from current position as array of arrays
        '''
        rows, cols = np.where(self.rods)
        valid_rods = [self.empty in self.board[:, r, c] for (r, c) in zip(rows, cols)]
        moves = np.array(list(zip(rows, cols)))[valid_rods]
        if player:
            moves = sorted(
                moves,
                key=lambda m: self.evaluate_move(m[0], m[1], player),
                reverse=True
            )
        return moves 


    def evaluate_move(self, r, c, player):
        ''' Return an evaluation on the move '''
        # Preserve state
        board, scores, potential, kth_line = self.get_state()

        # Advance
        self.drop_piece(r, c, player)
        score = self.evaluate(player)

        # Restore
        self.set_state(board, scores, potential, kth_line)

        return score


    def evaluate(self, player):
        opponenet = self.get_opponent(player)
        return (self.scores[player] - self.scores[opponenet]) * 10 + (self.potential[player] - self.potential[opponenet])


    def drop_piece(self, r, c, piece):
        '''
        Drop a piece and update relavant data
        '''
        rod = self.board[:, r, c]
        if self.empty in rod:
            free_layer = np.nonzero(rod == 0)[0][0]
            rod[free_layer] = piece
            self.dropped_pieces += 1
            self.update_score(free_layer, r, c, piece)
        else:
            print("Invalid Move")


    def infer_move(self, board):
        '''
        Tries to infer the moves that lead to the given position
        '''
        try:
            difference = self.board != np.array(board)
            if difference.any():
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
        except:
            print(self.board)
            print(board)


    def update_score(self, l, r, c, piece):
        '''
        Updates scores count and k, should be called after every move.
        '''
        oppo_piece = self.get_opponent(piece)
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

        # Return if window contains sentinel
        if window.count(self.sentinel) != 0:
            return

        # Count pieces
        # one: 0
        # two: 2
        # three: 5
        # four: 0(value stored in scores)
        pieces = (
            window.count(self.empty),
            window.count(player),
            window.count(opponent)
        )
        
        if pieces == (2, 2, 0):  # one -> two
            self.potential[player] += 2
        elif pieces == (1, 3, 0):  # two -> three
            self.potential[player] += 3
        elif pieces == (0, 4, 0):  # three -> four(complete line)
            self.kth_line.append(player)
            self.scores[player] += (100 // len(self.kth_line))
            self.potential[player] -= 5
        elif pieces == (1, 1, 2):  # block opponent's two
            self.potential[opponent] -= 2
        elif pieces == (0, 1, 3):  # block opponent's three
            self.potential[opponent] -= 5


    def get_opponent(self, player):
        return self.player_one if player == self.player_two else self.player_two


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


class MinimaxAI:
    def __init__(self, board):
        self.board = board
        self.best_move = (0, 0)


    def update(self, new_board):
        self.board.infer_move(new_board)


    def get_step(self, player):
        self.search(3, -99999, 99999, player)
        return self.best_move
        

    def search(self, depth, alpha, beta, player):
        if(depth == 0 or self.board.is_terminal()):
            return self.board.evaluate(player)

        for (r, c) in self.board.generate_moves():
            # Preserve state
            p_board, p_scores, p_potential, p_kth_line = self.board.get_state()
            
            # Search ahead
            self.board.drop_piece(r, c, player)
            opponent = self.board.get_opponent(player)
            evaluation = -self.search(depth-1, -beta, -alpha, opponent);  

            # Restore          
            self.board.set_state(p_board, p_scores, p_potential, p_kth_line)

            if(evaluation >= beta):
                break

            if(evaluation > alpha):
                alpha = evaluation
                self.best_move = (r, c)

        return alpha


ai = MinimaxAI(Game(LAYER, ROW, COLUMN))
old = 0
while True:
    (stop_program, id_package, board, is_black) = STcpClient.GetBoard()
    if stop_program:
        break
    if old != is_black:
        ai = MinimaxAI(Game(LAYER, ROW, COLUMN))    #clear
    old = is_black
    player = 1 if is_black else 2
    ai.update(np.array(board))
    Step = ai.get_step(player)
    STcpClient.SendStep(id_package, Step)
