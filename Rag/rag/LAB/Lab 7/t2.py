def alpha_beta(cards, left, right, is_max, alpha, beta):
    if left > right:
        return 0

    if is_max:
        max_score = float('-inf')
        score_left = cards[left] + alpha_beta(cards, left + 1, right, False, alpha, beta)
        max_score = max(max_score, score_left)
        alpha = max(alpha, max_score)
        if beta <= alpha:
            return max_score

        score_right = cards[right] + alpha_beta(cards, left, right - 1, False, alpha, beta)
        max_score = max(max_score, score_right)
        alpha = max(alpha, max_score)
        return max_score
    else:
        if cards[left] < cards[right]:
            return alpha_beta(cards, left + 1, right, True, alpha, beta)
        else:
            return alpha_beta(cards, left, right - 1, True, alpha, beta)

def max_move(cards):
    left = 0
    right = len(cards) - 1
    max_score = float('-inf')
    move = None

    score_left = cards[left] + alpha_beta(cards, left + 1, right, False, float('-inf'), float('inf'))
    if score_left > max_score:
        max_score = score_left
        move = 'left'

    score_right = cards[right] + alpha_beta(cards, left, right - 1, False, float('-inf'), float('inf'))
    if score_right > max_score:
        max_score = score_right
        move = 'right'

    return move

def play_game(cards):
    print(f"Initial Cards: {cards}")
    max_score = 0
    min_score = 0
    turn = 'max'

    while cards:
        if turn == 'max':
            move = max_move(cards)
            if move == 'left':
                picked = cards.pop(0)
            else:
                picked = cards.pop()
            max_score += picked
            print(f"Max picks {picked}, Remaining Cards: {cards}")
            turn = 'min'
        else:
            if len(cards) == 1 or cards[0] < cards[-1]:
                picked = cards.pop(0)
            else:
                picked = cards.pop()
            min_score += picked
            print(f"Min picks {picked}, Remaining Cards: {cards}")
            turn = 'max'

    print(f"Final Scores - Max: {max_score}, Min: {min_score}")
    if max_score > min_score:
        print("Winner: Max")
    elif min_score > max_score:
        print("Winner: Min")
    else:
        print("It's a Tie!")

cards = [4, 10, 6, 2, 9, 5]
play_game(cards)
