import random

NUM_SHIPS = 10

class BattleshipGame:
    def __init__(self):
        self.grid_size = 10
        self.player_grid = self.create_empty_grid()
        self.ai_grid = self.create_empty_grid()
        self.player_ships = NUM_SHIPS
        self.ai_ships = NUM_SHIPS
        self.ai_attacks = []
        self.player_attacks = []
        self.place_ships(self.player_grid, self.player_ships)
        self.place_ships(self.ai_grid, self.ai_ships)

    def create_empty_grid(self):
        return [['~' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

    def print_grid(self, grid):
        print("  A B C D E F G H I J")
        for i, row in enumerate(grid):
            print(f"{i + 1} {' '.join(row)}")

    def place_ships(self, grid, num_ships):
        ships_placed = 0
        while ships_placed < num_ships:
            row = random.randint(0, self.grid_size - 1)
            col = random.randint(0, self.grid_size - 1)
            if grid[row][col] == '~':  # If the spot is empty
                grid[row][col] = 'S'
                ships_placed += 1

    def attack(self, grid, attacks, row, col):
        if grid[row][col] == 'S':  # Hit
            grid[row][col] = 'X'
            return "Hit"
        elif grid[row][col] == '~':  # Miss
            grid[row][col] = 'O'
            return "Miss"
        return "Already attacked"

    def player_turn(self):
        while True:
            print("Your turn to attack!")
            self.print_grid(self.ai_grid)
            move = input("Enter your attack (e.g. B4): ").strip().upper()
            if len(move) != 2:
                print("Invalid input. Try again.")
                continue
            col = ord(move[0]) - ord('A')
            row = int(move[1]) - 1
            if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
                if (row, col) not in self.player_attacks:
                    self.player_attacks.append((row, col))
                    result = self.attack(self.ai_grid, self.player_attacks, row, col)
                    print(f"You → {move}: {result}")
                    break
                else:
                    print("You have already attacked this position.")
            else:
                print("Invalid coordinate. Try again.")

    def ai_turn(self):
        print("AI's turn to attack!")
        while True:
            row = random.randint(0, self.grid_size - 1)
            col = random.randint(0, self.grid_size - 1)
            if (row, col) not in self.ai_attacks:
                self.ai_attacks.append((row, col))
                result = self.attack(self.player_grid, self.ai_attacks, row, col)
                print(f"AI → {chr(col + 65)}{row + 1}: {result}")
                break

    def play(self):
        while self.player_ships > 0 and self.ai_ships > 0:
            self.player_turn()
            self.ai_turn()
            self.print_grid(self.player_grid)
            self.print_grid(self.ai_grid)
            self.player_ships = sum(row.count('S') for row in self.player_grid)
            self.ai_ships = sum(row.count('S') for row in self.ai_grid)
            print(f"Your ships left: {self.player_ships}, AI's ships left: {self.ai_ships}")
        
        if self.player_ships == 0:
            print("AI wins!")
        elif self.ai_ships == 0:
            print("You win!")

game = BattleshipGame()
game.play()
