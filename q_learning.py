import numpy as np
from q_table import Q_DICT
import pickle 
from datetime import datetime
import os
import time
from billiards_game_computer import BILLIARDS_GAME_COMPUTER
from tqdm import tqdm

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
        self.cue_ball_states = [[211.11111111111111, 578.0], [211.11111111111111, 524.8888888888889], [211.11111111111111, 206.22222222222223], [100.0, 578.0], [100.0, 418.6666666666667], [322.22222222222223, 471.7777777777778], [1100.0, 578.0], [433.33333333333337, 312.44444444444446], [988.8888888888889, 365.55555555555554], [100.0, 524.8888888888889]]
        self.target_ball_states = [[766.6666666666667, 100.0], [211.11111111111111, 471.7777777777778], [766.6666666666667, 259.33333333333337], [877.7777777777778, 153.11111111111111], [544.4444444444445, 100.0], [1100.0, 471.7777777777778], [877.7777777777778, 524.8888888888889], [655.5555555555555, 471.7777777777778], [766.6666666666667, 100.0], [100.0, 418.6666666666667]]
    
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
        powers = self.Q.powers
        angles = self.Q.angles
        epoch = 1 # Start epoch
        for epoch in tqdm(range(self.num_epochs)):
            self.game = BILLIARDS_GAME_COMPUTER( self.display)
            self.game.setup_game()
            while(self.game.run):
                cue_state = self.cue_ball_states[np.random.randint(len(self.cue_ball_states))]
                target_state= self.target_ball_states[np.random.randint(len(self.target_ball_states))]
                angle, power, max_q = self.Q.get_max_action_pair(cue_state, target_state)
                print(f"\nmax_q= {max_q}\n")
                if max_q == 0.00:
                    angle = np.random.choice(angles)
                    power = np.random.choice(powers)
                reward = self.game.shoot(angle, power, cue_state, target_state)
                self.Q.update_q_dict(cue_state, target_state, angle, power, reward)
            self.save_Q_table() # Updates Q-table after every epoch

        #self.game.pygame.quit()

    def begin_training(self):
        self.load_Q_table()
        powers = self.Q.powers
        angles = self.Q.angles
        for epoch in tqdm(range(self.num_epochs)):
            self.game = BILLIARDS_GAME_COMPUTER(self.display)
            self.game.setup_game()
            while(self.game.run):
                cue_state = self.cue_ball_states[np.random.randint(len(self.cue_ball_states))]
                target_state= self.target_ball_states[np.random.randint(len(self.target_ball_states))]
                power = np.random.choice(powers)
                angle = np.random.choice(angles)
                reward = self.game.shoot(angle, power, cue_state, target_state)
                self.Q.update_q_dict(cue_state, target_state, angle, power, reward)
            self.save_Q_table() # Updates Q-table after every epoch

        #self.game.pygame.quit()