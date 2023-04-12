import numpy as np
from q_table import Q_DICT
import pickle 
from datetime import datetime
import os
import time
from billiards_game_computer import BILLIARDS_GAME_COMPUTER

class Q_LEARNING:
    def __init__(self, test = True, display = False, num_epochs=10000, num_cue_ball_states = 10, num_target_ball_states = 10):
        self.Q = Q_DICT()
        self.test = test
        self.display = display
        self.num_epochs = num_epochs
        self.num_cue_ball_states = num_cue_ball_states
        self.num_target_ball_states = num_target_ball_states
        min_ball_pos_x = 100
        max_ball_pos_x = 1100
        min_ball_pos_y = 100
        max_ball_pos_y = 578
        self.cue_ball_states = []
        self.target_ball_states = []
        cue_ball_xstates_range = np.linspace(min_ball_pos_x, max_ball_pos_x, num = num_cue_ball_states)
        cue_ball_ystates_range = np.linspace(min_ball_pos_y, max_ball_pos_y, num = num_cue_ball_states)
        target_ball_xstates_range = np.linspace(min_ball_pos_x, max_ball_pos_x, num = num_target_ball_states)
        target_ball_ystates_range = np.linspace(min_ball_pos_y, max_ball_pos_y, num = num_target_ball_states)
        for _ in range(num_cue_ball_states):
            self.cue_ball_states.append([np.random.choice(cue_ball_xstates_range), np.random.choice(cue_ball_ystates_range)])
        for _ in range(num_target_ball_states):
            self.target_ball_states.append([np.random.choice(target_ball_xstates_range), np.random.choice(target_ball_ystates_range)])

    def load_Q_table(self):
        with open('q_dicts/q_dict_best.pkl', 'rb') as fp:
            self.Q.Q_dict = pickle.load(fp)
            print('\n*Loaded Q-dict from file*')
            print(f"Size of loaded q-dict: {len(self.Q.Q_dict)}\n")

    def save_Q_table(self):
        #now = datetime.now()
        #dt_string = now.strftime("%d_%m_%Y")
        #with open(f'q_dicts/q_dict_{dt_string}.pkl', 'wb') as fp:
        with open('q_dicts/q_dict_best.pkl', 'wb') as bb:
            pickle.dump(self.Q.Q_dict, bb)
            print('\n*Dictionary saved successfully to file*')
            print(f"Size of saved q-dict: {len(self.Q.Q_dict)}\n")

    def begin(self):
        if self.test == True:
            self.begin_testing()
        else:
            self.begin_training()
    
    def begin_testing(self):
        self.load_Q_table()
        self.game.begin()

    def begin_training(self):
        #self.load_Q_table()
        powers = self.Q.powers
        angles = self.Q.angles
        epoch = 1 # Start epoch
        while(epoch <= self.num_epochs):
            self.game = BILLIARDS_GAME_COMPUTER(self.target_ball_states, self.target_ball_states, self.display)
            self.game.setup_game()
            while(self.game.run):
                power = np.random.choice(powers)
                angle = np.random.choice(angles)
                cue_state, target_state, reward = self.game.shoot(angle, power)
                self.Q.update_q_dict(cue_state, target_state, angle, power, reward)
            epoch += 1
            self.save_Q_table() # Updates Q-table after every epoch

        #self.game.pygame.quit()