import chess
import chess.engine

def beam_search(board, beam_width, depth_limit):
    beam = [{
        'board': board.copy(),
        'moves': [],
        'score': evaluate_board(board)
    }]
    
    for depth in range(depth_limit):
        new_beam = []
        
        for candidate in beam:
            current_board = candidate['board']
            
            if current_board.is_game_over():
                new_beam.append(candidate)
                continue
                
            for move in current_board.legal_moves:
                board_copy = current_board.copy()
                board_copy.push(move)
                
                new_candidate = {
                    'board': board_copy,
                    'moves': candidate['moves'] + [move],
                    'score': evaluate_board(board_copy)
                }
                new_beam.append(new_candidate)
        
        new_beam.sort(key=lambda x: x['score'], reverse=True)
        beam = new_beam[:beam_width]
        
        if beam and beam[0]['board'].is_checkmate():
            break
    
    if not beam:
        return [], 0
    
    best_candidate = beam[0]
    return best_candidate['moves'], best_candidate['score']

def evaluate_board(board):
    if board.is_checkmate():
        if board.turn:  # Black just moved and gave checkmate
            return -9999
        else:
            return 9999
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
        return 0
    
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            score += value if piece.color == chess.WHITE else -value
    
    # Add small bonus for check
    if board.is_check():
        score += 0.5 if board.turn == chess.BLACK else -0.5
    
    return score

# Example usage
if __name__ == "__main__":
    board = chess.Board()
    beam_width = 3
    depth_limit = 2
    
    best_moves, best_score = beam_search(board, beam_width, depth_limit)
    print(f"Best move sequence: {[move.uci() for move in best_moves]}")
    print(f"Evaluation score: {best_score}")
