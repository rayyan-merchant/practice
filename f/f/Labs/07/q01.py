import draughts
from draughts.models import Figure, Color
import os

def evaluate(board: draughts.BaseBoard) -> float:
    material = 0
    positional = 0
    
    for i, piece in enumerate(board.position):
        # adding piece values
        if piece == Figure.BLACK_MAN.value:
            material += 2 if piece == Figure.BLACK_KING.value else 1
        elif piece == Figure.WHITE_MAN.value:
            material -= 2 if piece == Figure.WHITE_KING.value else 1

        # adding positional advantage for pieces
        if abs(piece) == Figure.KING:
            row = i // 5
            if piece == Figure.BLACK_MAN.value:
                positional += row / 10
            elif piece == Figure.WHITE_MAN.value:
                positional -= (9 - row) / 10
    
    # mobility score depending on the amount of options
    mobility = len(list(board.legal_moves))
    if board.turn == Color.WHITE:
        positional += mobility * 0.01
    else:
        positional -= mobility * 0.01
    
    return material + positional

def alphabeta(
    board: draughts.StandardBoard,
    depth: int = 3,
    alpha: float = float('-inf'),
    beta: float = float('inf'),
    maximizing: bool = True
):
    if depth == 0 or board.game_over:
        return evaluate(board), None
    
    best_move = None
    if maximizing:
        max_val = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            v, _ = alphabeta(board, depth - 1, alpha, beta, False)
            board.pop()
            if v > max_val:
                max_val = v
                best_move = move
            alpha = max(alpha, max_val)
            if beta <= alpha:
                break
        return max_val, best_move
    else:
        min_val = float('inf')
        for move in board.legal_moves:
            board.push(move)
            v, _ = alphabeta(board, depth-1, alpha, beta, True)
            board.pop()
            if v < min_val:
                min_val = v
                best_move = move
            beta = min(beta, min_val)
            if beta <= alpha:
                break
        return min_val, best_move
    
def ai_move(board: draughts.BaseBoard, depth: int = 3):
    _, best_move = alphabeta(board, depth)
    return best_move

board = draughts.get_board("standard")
DEPTH = 5

prev_move = None
while not board.game_over:
    print(board)
    prev_move and print("AI moved:", prev_move)
    print("\nAll legal moves:")
    for i, move in enumerate(board.legal_moves):
        print(f"{i}: {move}")
    
    while True:
        try:
            inp = input("Select move number: ")
            if not inp.strip():
                continue
            inp = int(inp)
            if 0 <= inp < len(board.legal_moves):
                break
            print("Invalid input. Please enter a number from the list.")
        except ValueError:
            print("Please enter a valid number.")
    
    board.push(board.legal_moves[inp])
    
    if not board.game_over:
        prev_move = ai_move(board, DEPTH)
        if prev_move:
            board.push(prev_move)
        else:
            break
    
    os.system("cls" if os.name == 'nt' else "clear")

print("Game over!")
print(board)
print(board.result)
