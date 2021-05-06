
import STcpClient
import random
import numpy as np

'''
    輪到此程式移動棋子
    board : 棋盤狀態(list of list), board[l][i][j] = l layer, i row, j column 棋盤狀態(l, i, j 從 0 開始)
            0 = 空、1 = 黑、2 = 白、-1 = 四個角落
    is_black : True 表示本程式是黑子、False 表示為白子

    return Step
    Step : single touple, Step = (r, c)
            r, c 表示要下棋子的座標位置 (row, column) (zero-base)
'''
WINDOW_LENGTH = 4
EMPTY = 0
LAYER = 6
ROW = 6
COLUMN = 6


class Board:
    def __init__(self):
        self.board = np.zeros((LAYER, ROW, COLUMN), dtype = int)
        self.kth_line = []
        self.curr_kth = 0
        #for i in range(6):
        self.board[0:6, 0, [0,1,COLUMN-2,COLUMN-1]]=-1
        self.board[0:6, 1, [0,COLUMN-1]]=-1
            #self.board[i][0][0]=-1   #left down
            #self.board[i][0][1]=-1
            #self.board[i][0][COLUMN-1]=-1     #right down
            #self.board[i][0][COLUMN-2]=-1
            #self.board[i][1][COLUMN-1]=-1
        self.board[0:6, ROW-1, [0,1,COLUMN-2,COLUMN-1]]=-1
        self.board[0:6, ROW-2, [0,COLUMN-1]]=-1
            #self.board[i][ROW-1][0]=-1   #left up
            #self.board[i][ROW-1][1]=-1
            #self.board[i][ROW-2][0]=-1
            #self.board[i][ROW-1][COLUMN-1]=-1   #right up
            #self.board[i][ROW-1][COLUMN-2]=-1
            #self.board[i][ROW-2][COLUMN-1]=-1

    def print_board(self):
        for i in range(6):
            print('Layer ', i)
            print(np.flip(self.board[i], 0))

    def get_current_layer(self):
        for i in range(1, 6):
            clear_flag = 0
            for j in range(6):
                for k in range(6):
                    if self.board[i][j][k] > 0: #layer i has piece
                        clear_flag = 1
            if clear_flag == 0: #this layer has no piece
                return (i-1)
        return 5
        
    def line_count(self, piece):
        line = 0
        curr_layer = int(self.get_current_layer())
        for i in range(curr_layer+1):
            # Check horizontal lines
            for c in range(COLUMN-3):
                for r in range(1, ROW-1):
                    if np.array_equiv(self.board[i, r, c:c+4], [piece, piece, piece, piece]):
                    #if self.board[i][r][c] == piece and self.board[i][r][c+1] == piece and self.board[i][r][c+2] == piece and self.board[i][r][c+3] == piece:
                        line += 1
            # Check vertical lines
            for c in range(1, COLUMN-1):
                for r in range(ROW-3):
                    if np.array_equiv(self.board[i, r:r+4, c], [piece, piece, piece, piece]):
                    #if self.board[i][r][c] == piece and self.board[i][r+1][c] == piece and self.board[i][r+2][c] == piece and self.board[i][r+3][c] == piece:
                        line += 1
            # Check positively sloped lines
            for c in range(COLUMN-3):
                for r in range(ROW-3):
                    if self.board[i][r][c] == piece and self.board[i][r+1][c+1] == piece and self.board[i][r+2][c+2] == piece and self.board[i][r+3][c+3] == piece:
                        line += 1
            # Check negatively sloped lines
            for c in range(COLUMN-3):
                for r in range(3, ROW):
                    if self.board[i][r][c] == piece and self.board[i][r-1][c+1] == piece and self.board[i][r-2][c+2] == piece and self.board[i][r-3][c+3] == piece:
                        line += 1
        # Check 3D lines
        #tmp_layer = self.get_current_layer()
        if curr_layer < 3: #no lines in 3D situation
            line += 0
        else:
            for c in range(6):  # Check same position's height line
                for r in range(6):
                    for i in range(curr_layer-2):
                        if np.array_equiv(self.board[i:i+4, r, c], [piece, piece, piece, piece]):
                        #if self.board[i][r][c] == piece and self.board[i+1][r][c] == piece and self.board[i+2][r][c] == piece and self.board[i+3][r][c] == piece:
                            line += 1
            for i in range(curr_layer-2):
                # Check horizontal and positively sloped lines
                for c in range(COLUMN-3):
                    for r in range(1, ROW-1):
                        if self.board[i][r][c] == piece and self.board[i+1][r][c+1] == piece and self.board[i+2][r][c+2] == piece and self.board[i+3][r][c+3] == piece:
                            line += 1
                # Check horizontal and negatively sloped lines
                for c in range(COLUMN-3):
                    for r in range(1, ROW-1):
                        if self.board[i+3][r][c] == piece and self.board[i+2][r][c+1] == piece and self.board[i+1][r][c+2] == piece and self.board[i][r][c+3] == piece:
                            line += 1
                # Check vertical and positively sloped lines
                for c in range(1, COLUMN-1):
                    for r in range(ROW-3):
                        if self.board[i][r][c] == piece and self.board[i+1][r+1][c] == piece and self.board[i+2][r+2][c] == piece and self.board[i+3][r+3][c] == piece:
                            line += 1
                # Check vertical and negatively sloped lines
                for c in range(1, COLUMN-1):
                    for r in range(ROW-3):
                        if self.board[i+3][r][c] == piece and self.board[i+2][r+1][c] == piece and self.board[i+1][r+2][c] == piece and self.board[i][r+3][c] == piece:
                            line += 1
                # Check positively sloped and positively sloped lines (斜向右上，高度往上長)
                for c in range(COLUMN-3):
                    for r in range(ROW-3):
                        if self.board[i][r][c] == piece and self.board[i+1][r+1][c+1] == piece and self.board[i+2][r+2][c+2] == piece and self.board[i+3][r+3][c+3] == piece:
                            line += 1
                # Check negatively sloped and positively sloped lines (斜向右下，高度往上長)
                for c in range(COLUMN-3):
                    for r in range(3, ROW):
                        if self.board[i][r][c] == piece and self.board[i+1][r-1][c+1] == piece and self.board[i+2][r-2][c+2] == piece and self.board[i+3][r-3][c+3] == piece:
                            line += 1
                # Check positively sloped and positively sloped lines (斜向右上，高度往下長)
                for c in range(COLUMN-3):
                    for r in range(ROW-3):
                        if self.board[i+3][r][c] == piece and self.board[i+2][r+1][c+1] == piece and self.board[i+1][r+2][c+2] == piece and self.board[i][r+3][c+3] == piece:
                            line += 1
                # Check negatively sloped and positively sloped lines (斜向右下，高度往下長)
                for c in range(COLUMN-3):
                    for r in range(3, ROW):
                        if self.board[i+3][r][c] == piece and self.board[i+2][r-1][c+1] == piece and self.board[i+1][r-2][c+2] == piece and self.board[i][r-3][c+3] == piece:
                            line += 1            
        return line
    

    def current_k(self):
        k = self.line_count(1) + self.line_count(2)
        return k
    

    def drop_piece(self, row, col, piece):          #add kth line
        for i in range(6):
            if self.board[i][row][col] == 0:
                #prev_k = self.current_k()
                self.board[i][row][col] = piece
                now_k = self.current_k()
                if self.curr_kth != now_k:
                    self.kth_line.append(piece)
                    self.curr_kth = now_k
                return
        return False


    def get_valid_position(self, r, c):
        for i in range(6):
            if self.board[i][r][c] == 0:
                return i, r, c
        return -1, r, c

    def evaluate_window(self, window, piece):
        score = 0
        opp_piece = 2
        if window.count(piece) == 4:
            score += 100
        elif window.count(piece) == 3 and window.count(EMPTY) == 1:
            score += 5
        elif window.count(piece) == 2 and window.count(EMPTY) == 2:
            score += 2
	    
        if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
            score -= 2

        return score

       
    def score_pace(self, temp_board, pace):
        score = 0
        h = pace[0]
        r = pace[1]
        c = pace[2]
        temp_board.drop_piece(r, c, 1)
        ## Score center column
        #center_array = [int(i) for i in list(temp_board.board[h][:, COLUMN//2])]
        #center_count = center_array.count(1)
        #score += center_count * 3
	
        ## Score Horizontal
        row_array = [int(i) for i in list(temp_board.board[h][r,:])]
        for c in range(COLUMN-3):
            window = row_array[c:c+WINDOW_LENGTH]
            score += temp_board.evaluate_window(window, 1)     #count 1s in the window
            
        ## Score Vertical
        col_array = [int(i) for i in list(temp_board.board[h][:,c])]
        for r in range(ROW-3):
            window = col_array[r:r+WINDOW_LENGTH]
            score += temp_board.evaluate_window(window, 1)

        #print(score)
        return score

    def choose_next_pace(self, paces_3d):
        score = -10000
        h = 0
        r = 0
        c = 0
        for pace in paces_3d:   #find highest score
            if pace[0] >= 0:
                temp_board = Board()
                temp_board.board = self.board.copy()
                tmp_score = self.score_pace(temp_board, pace)
                if tmp_score > score:
                    score = tmp_score
                    h, r, c = pace
        if score > 0: print(score)
        return r, c

        
def GetStep(board, is_black):
    """
    Example:

    x = random.randint(0, 5)
    y = random.randint(0, 5)
    return (x, y)
    """

board1 = Board()

paces = [[0, 2], [0, 3],
          [1, 1], [1, 2], [1, 3], [1, 4],
          [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5],
          [3, 0], [3, 1], [3, 3], [3, 3], [3, 4], [3, 5],
          [4, 1], [4, 4], [4, 4], [4, 4],
          [5, 2], [5, 3]]

for i in range(32):
    paces_3d = []
    for pace in paces:
        paces_3d.append(board1.get_valid_position(pace[0], pace[1]))
    x1, y1 = board1.choose_next_pace(paces_3d)
    #x1, y1 = random.choice(paces)
    #h, x1, y1 = board1.get_valid_position(x1, y1)
    #print(board1.score_pace([h, x1, y1]))
    x2, y2 = random.choice(paces)
    print(x1, y1)
    print(x2, y2)
    board1.drop_piece(x1, y1, 1)
    board1.drop_piece(x2, y2, 2)


#a, b, c = board1.get_valid_position(3, 3)
#print(board1.score_pace([a,b,c]))

board1.print_board()    

#print(board1.current_k())
print('black', board1.line_count(1))
print('white', board1.line_count(2))
print(board1.kth_line)

'''
while(True):
    (stop_program, id_package, board, is_black) = STcpClient.GetBoard()
    if(stop_program):
        break

    Step = GetStep(board, is_black)
    STcpClient.SendStep(id_package, Step)
'''
