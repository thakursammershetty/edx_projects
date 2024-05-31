import random
import time  # Import time module


class Nim:
    def __init__(self, piles):
        self.piles = piles
        self.player = 0
        self.winner = None

    @classmethod
    def available_actions(cls, piles):
        actions = set()
        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                actions.add((i, j))
        return actions

    @staticmethod
    def other_player(player):
        return 0 if player == 1 else 1

    def switch_player(self):
        self.player = Nim.other_player(self.player)

    def move(self, action):
        pile, count = action
        if self.piles[pile] >= count:
            self.piles[pile] -= count
        else:
            raise ValueError("Invalid move")

        if all(pile == 0 for pile in self.piles):
            self.winner = self.player

        self.switch_player()


class NimAI:
    def __init__(self, alpha=0.5, epsilon=0.1):
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

    def get_q_value(self, state, action):
        state = tuple(state)
        if (state, action) in self.q:
            return self.q[(state, action)]
        else:
            return 0

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        new_q = old_q + self.alpha * (reward + future_rewards - old_q)
        state = tuple(state)
        self.q[(state, action)] = new_q

    def best_future_reward(self, state):
        state = tuple(state)
        actions = Nim.available_actions(state)
        best_reward = 0
        for action in actions:
            reward = self.get_q_value(state, action)
            if reward > best_reward:
                best_reward = reward
        return best_reward

    def choose_action(self, state, epsilon=True):
        actions = list(Nim.available_actions(state))
        if not epsilon:
            best_action = max(actions, key=lambda action: self.get_q_value(state, action))
            return best_action

        if random.random() < self.epsilon:
            return random.choice(actions)
        else:
            best_action = max(actions, key=lambda action: self.get_q_value(state, action))
            return best_action


def train(n):
    ai = NimAI()
    for i in range(n):
        game = Nim([1, 3, 5, 7])
        last = {0: {"state": None, "action": None}, 1: {"state": None, "action": None}}
        while True:
            state = game.piles.copy()
            action = ai.choose_action(state)
            last[game.player]["state"] = state
            last[game.player]["action"] = action
            game.move(action)
            new_state = game.piles.copy()
            if game.winner is not None:
                ai.update_q_value(last[game.player]["state"], last[game.player]["action"], ai.get_q_value(last[game.player]["state"], last[game.player]["action"]), 1, 0)
                ai.update_q_value(last[game.other_player(game.player)]["state"], last[game.other_player(game.player)]["action"], ai.get_q_value(last[game.other_player(game.player)]["state"], last[game.other_player(game.player)]["action"]), -1, 0)
                break
            elif last[game.other_player(game.player)]["state"] is not None:
                reward = 0
                future_rewards = ai.best_future_reward(new_state)
                ai.update_q_value(last[game.other_player(game.player)]["state"], last[game.other_player(game.player)]["action"], ai.get_q_value(last[game.other_player(game.player)]["state"], last[game.other_player(game.player)]["action"]), reward, future_rewards)
    return ai


def play(ai):
    game = Nim([1, 3, 5, 7])

    while True:
        print("\nPiles:")
        for i, pile in enumerate(game.piles):
            print(f"Pile {i}: {pile}")
        print()

        if game.winner is not None:
            print("GAME OVER")
            time.sleep(1)
            return

        print("Your Turn")
        pile = int(input("Choose Pile: "))
        count = int(input("Choose Count: "))
        game.move((pile, count))

        print("\nPiles:")
        for i, pile in enumerate(game.piles):
            print(f"Pile {i}: {pile}")
        print()

        if game.winner is not None:
            print("GAME OVER")
            time.sleep(1)
            return

        print("AI's Turn")
        pile, count = ai.choose_action(game.piles)
        print(f"AI chose to take {count} from pile {pile}.")
        game.move((pile, count))
        time.sleep(1)

