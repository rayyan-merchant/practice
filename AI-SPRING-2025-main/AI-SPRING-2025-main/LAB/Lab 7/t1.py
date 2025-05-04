import random

WHITE = 'W'
BLACK = 'B'
EMPTY = '.'

DIRECTIONS = [(-1, -1), (-1, 1)]

class CheckersGame:
    def __init__(self):
        self.board = self.setup_board()
        self.current_player = WHITE
        self.game_over = False

    def setup_board(self):
        board = [[EMPTY] * 8 for _ in range(8)]
        for row in range(0, 3, 2):
            for col in range(0, 8, 2):
                board[row][col + 1] = WHITE
        for row in range(5, 8, 2):
            for col in range(0, 8, 2):
                board[row][col + 1] = BLACK
        return board

    def print_board(self):
        for row in self.board:
            print(' '.join(row))
        print()

    def is_valid_move(self, start, end):
        sx, sy = start
        ex, ey = end
        if not (0 <= ex < 8 and 0 <= ey < 8):
            return False
        if self.board[sx][sy] == EMPTY or self.board[ex][ey] != EMPTY:
            return False
        return abs(sx - ex) == 1 and abs(sy - ey) == 1

    def is_capture_move(self, start, end):
        sx, sy = start
        ex, ey = end
        mx, my = (sx + ex) // 2, (sy + ey) // 2
        if self.board[mx][my] == (BLACK if self.current_player == WHITE else WHITE):
            return True
        return False

    def make_move(self, start, end):
        sx, sy = start
        ex, ey = end
        self.board[ex][ey] = self.board[sx][sy]
        self.board[sx][sy] = EMPTY
        if self.is_capture_move(start, end):
            mx, my = (sx + ex) // 2, (sy + ey) // 2
            self.board[mx][my] = EMPTY

    def player_turn(self):
        print("Player's turn (White):")
        self.print_board()
        start = tuple(map(int, input("Enter the start position (row,col): ").split(',')))
        end = tuple(map(int, input("Enter the end position (row,col): ").split(',')))

        if self.is_valid_move(start, end):
            self.make_move(start, end)
        elif self.is_capture_move(start, end):
            self.make_move(start, end)
            print(f"Player moves: {start} → {end} [Capture!]")
        else:
            print("Invalid move, try again.")
            self.player_turn()

    def ai_turn(self):
        print("AI's turn (Black):")
        self.print_board()
        available_moves = self.get_valid_moves(BLACK)
        if available_moves:
            move = random.choice(available_moves)
            self.make_move(move[0], move[1])
            print(f"AI moves: {move[0]} → {move[1]}")
        else:
            print("AI has no valid moves.")

    def get_valid_moves(self, player):
        valid_moves = []
        for row in range(8):
            for col in range(8):
                if self.board[row][col] == player:
                    for dx, dy in DIRECTIONS:
                        nx, ny = row + dx, col + dy
                        if 0 <= nx < 8 and 0 <= ny < 8 and self.board[nx][ny] == EMPTY:
                            valid_moves.append(((row, col), (nx, ny)))
                        elif 0 <= nx < 8 and 0 <= ny < 8 and self.is_capture_move((row, col), (nx, ny)):
                            valid_moves.append(((row, col), (nx, ny)))
        return valid_moves

    def check_game_over(self):
        white_pieces = sum(row.count(WHITE) for row in self.board)
        black_pieces = sum(row.count(BLACK) for row in self.board)

        if white_pieces == 0 or black_pieces == 0:
            return True

        if not self.get_valid_moves(WHITE) or not self.get_valid_moves(BLACK):
            return True

        return False

    def play(self):
        while not self.game_over:
            if self.current_player == WHITE:
                self.player_turn()
                self.current_player = BLACK
            else:
                self.ai_turn()
                self.current_player = WHITE

            if self.check_game_over():
                self.game_over = True
                print("Game Over!")
                break

game = CheckersGame()
game.play()
