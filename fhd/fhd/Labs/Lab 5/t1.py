import chess
import chess.engine
import heapq

def evaluate_board(board):
    """Evaluates the board position using a chess engine."""
    with chess.engine.SimpleEngine.popen_uci("stockfish") as engine:
        result = engine.analyse(board, chess.engine.Limit(time=0.1))
        return result['score'].relative.score(mate_score=10000)  # Adjusted for mate

def generate_moves(board):
    """Generates all possible legal moves from the current board state."""
    return list(board.legal_moves)

def beam_search_chess(board, beam_width=3, depth_limit=3):
    """Performs beam search to find the best move sequence."""
    beam = [(evaluate_board(board), board, [])]  # (score, board state, move sequence)
    
    for _ in range(depth_limit):
        candidates = []
        
        for score, current_board, move_sequence in beam:
            for move in generate_moves(current_board):
                new_board = current_board.copy()
                new_board.push(move)
                new_score = evaluate_board(new_board)
                candidates.append((new_score, new_board, move_sequence + [move]))
        
        # Select top-k boards based on evaluation score (higher is better in chess)
        beam = heapq.nlargest(beam_width, candidates, key=lambda x: x[0])
    
    # Return the best sequence found
    return beam[0][2], beam[0][0] if beam else ([], 0)

# Example Usage
board = chess.Board()
beam_width = 3
depth_limit = 3
best_moves, best_score = beam_search_chess(board, beam_width, depth_limit)

print(f"Best Move Sequence: {[board.san(move) for move in best_moves]}")
print(f"Evaluation Score: {best_score}")
