
import copy
import math

EMPTY = '.'
WHITE = 'W'
BLACK = 'B'

BOARD_SIZE = 8

def create_board():
    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    for row in range(3):
        for col in range(BOARD_SIZE):
            if (row + col) % 2 != 0:
                board[row][col] = BLACK
    for row in range(5, 8):
        for col in range(BOARD_SIZE):
            if (row + col) % 2 != 0:
                board[row][col] = WHITE
    return board

def print_board(board):
    for row in range(BOARD_SIZE):
        print(f"{row}: ", ' '.join(board[row]))
    print("   ", ' '.join(map(str, range(BOARD_SIZE))))

def get_all_moves(board, player):
    directions = [(-1, -1), (-1, 1)] if player == WHITE else [(1, -1), (1, 1)]
    moves = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == player:
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr][nc] == EMPTY:
                        moves.append(((r, c), (nr, nc)))
                    # Check for capture
                    cr, cc = r + 2 * dr, c + 2 * dc
                    if (
                        0 <= cr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and
                        board[nr][nc] not in [player, EMPTY] and
                        board[cr][cc] == EMPTY
                    ):
                        moves.append(((r, c), (cr, cc)))
    return moves

def make_move(board, move):
    start, end = move
    r1, c1 = start
    r2, c2 = end
    new_board = copy.deepcopy(board)
    new_board[r2][c2] = new_board[r1][c1]
    new_board[r1][c1] = EMPTY
    # Capture logic
    if abs(r2 - r1) == 2:
        new_board[(r1 + r2) // 2][(c1 + c2) // 2] = EMPTY
    return new_board

def evaluate_board(board):
    white_score = sum(row.count(WHITE) for row in board)
    black_score = sum(row.count(BLACK) for row in board)
    return black_score - white_score

def minimax(board, depth, maximizing, alpha, beta):
    player = BLACK if maximizing else WHITE
    moves = get_all_moves(board, player)
    if depth == 0 or not moves:
        return evaluate_board(board), None

    best_move = None
    if maximizing:
        max_eval = -math.inf
        for move in moves:
            new_board = make_move(board, move)
            eval, _ = minimax(new_board, depth - 1, False, alpha, beta)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = math.inf
        for move in moves:
            new_board = make_move(board, move)
            eval, _ = minimax(new_board, depth - 1, True, alpha, beta)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

def play_game():
    board = create_board()
    print_board(board)

    while True:
        # Human player move
        moves = get_all_moves(board, WHITE)
        if not moves:
            print("AI wins! No moves left for player.")
            break
        print("Your move (e.g., 2 3 3 4): ")
        try:
            r1, c1, r2, c2 = map(int, input().split())
            if ((r1, c1), (r2, c2)) in moves:
                board = make_move(board, ((r1, c1), (r2, c2)))
                print(f"Player moves: ({r1},{c1}) → ({r2},{c2})")
            else:
                print("Invalid move. Try again.")
                continue
        except Exception:
            print("Invalid input format. Try again.")
            continue

        print_board(board)

        # AI move
        moves = get_all_moves(board, BLACK)
        if not moves:
            print("You win! No moves left for AI.")
            break
        _, move = minimax(board, 4, True, -math.inf, math.inf)
        if move:
            board = make_move(board, move)
            (r1, c1), (r2, c2) = move
            print(f"AI moves: ({r1},{c1}) → ({r2},{c2})")
        print_board(board)

if __name__ == "__main__":
    play_game()
