def calculate_probabilities():
    total_cards = 52
    red_cards = 26  # Hearts and Diamonds
    hearts = 13
    diamonds = 13
    face_cards = 12  # Jack, Queen, King of each suit
    
    p_red = red_cards / total_cards 
    
    p_heart_given_red = hearts / red_cards
    
    diamond_face_cards = 3
    
    p_diamond_given_face = diamond_face_cards / face_cards
    
    return {
        'Probability of red card': p_red,
        'Probability of heart given red': p_heart_given_red,
        'Probability of diamond given face card': p_diamond_given_face
    }

probabilities = calculate_probabilities()
for description, prob in probabilities.items():
    print(f"{description}: {prob:.2f} ({prob*100:.1f}%)")
