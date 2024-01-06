import random
import numpy as np


def calculate_reward(own_move, opponent_move):
    if own_move == opponent_move:
        return 0
    elif (
            (own_move == 0 and opponent_move == 2) or
            (own_move == 1 and opponent_move == 0) or
            (own_move == 2 and opponent_move == 1)
    ):
        return 1
    else:
        return -1


class MyAgent:

    def __init__(self, num_actions, name="bot_agent"):
        self._num_actions = num_actions  # 3
        self._player_id = 0
        self.name = name
        self.initial_learning_rate = 0.1
        self.learning_rate = self.initial_learning_rate
        self.discount_factor = 0.98
        self.initial_exploration_rate = 0.8
        self.exploration_rate = self.initial_exploration_rate
        scores = {'Rock': -0.281889463641504,
                  'Paper': -0.3315825285542527,
                  'Scissors': -0.44611235670590477}
        #                  {0: 2.326888528435059, 1: 2.1705915008470855, 2: 1.5188835179425637}

        self.q_values = {0: scores['Rock'], 1: scores['Paper'], 2: scores["Scissors"]}
        self.state = []

    def choose_move(self):
        if random.random() < self.exploration_rate:
            return random.choice([0, 1, 2])
        else:
            max_q_value = max(self.q_values.values())
            best_moves = [move for move, q_value in self.q_values.items() if q_value == max_q_value]
            chosen_move = random.choice(best_moves)
            return chosen_move

    def get_best_action_and_update_q_values(self, opponent_move):

        action = self.choose_move()
        reward = calculate_reward(action, opponent_move)

        self.q_values[action] += self.learning_rate * (
                reward + self.discount_factor * max(self.q_values.values()) - self.q_values[action]
        )

        return action

    def step(self, opponent_action):

        if len(self.state) == 0:
            action = np.random.choice([0, 1, 2])
        else:
            action = self.get_best_action_and_update_q_values(self.state[-1])
        probs = np.zeros(self._num_actions)
        probs[action] = 1
        self.state.append(action)
        self.state.append(opponent_action)
        return action


if __name__ == '__main__':
    bot = MyAgent(3)
    print("*" * 5)
    print("Welcome to Rock scissors paper game! You are facing bot trained using Q-Learning algorithm.")
    print("Lets see how it performs")
    print("Game rules:\n1. For Rock - press 0, for Paper - 1, Scissors - 2")
    print("*" * 5)
    rule_map = {0: 'rock', 1: 'scissors', 2: 'paper'}
    while True:
        player_action = int(input("Enter a number:"))
        while 2 < player_action or player_action > 2:
            player_action = int(input("Enter a valid number:"))
        bot_action = bot.step(player_action)
        if bot_action == player_action:
            print("Draw")
        elif (
                (bot_action == 0 and player_action == 2) or
                (bot_action == 1 and player_action == 0) or
                (bot_action == 2 and player_action == 1)
        ):
            print("You lost, bot played " + rule_map[bot_action])
        else:
            print("You won! Bot played " + rule_map[bot_action])
