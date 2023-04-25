'''
This is the starting script for the Q-learning.
Decision Making Under Uncertainty Final Project (Spring 2023)

If you want to see the learning happen with a gui (slow), choose to set
game = BILLIARDS_GAME(). Otherwise, let game = BILLIARDS_GAME_NO_DISPLAY().

Author: Doncey Albin
'''

from q_learning import Q_LEARNING
from billiards_game_computer import BILLIARDS_GAME_COMPUTER
from billiards_game_human import BILLIARDS_GAME_HUMAN

def Q_learning():
    q_learning = Q_LEARNING(test=False, display=True, num_epochs=200_000, num_obj_balls = 15)
    q_learning.begin()

def human_test():
    game = BILLIARDS_GAME_HUMAN()
    game.setup_game()
    game.begin_human()

if __name__ == "__main__":
    #human_test()
    Q_learning()