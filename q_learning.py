import numpy as np
from q_table import Q_DICT
import pickle 
from datetime import datetime
import os
import time
from billiards_game_computer import BILLIARDS_GAME_COMPUTER
from tqdm import tqdm

class Q_LEARNING:
    def __init__(self, test = True, display = False, num_epochs=10000, num_ball_pos_states = 10):
        self.Q = Q_DICT()
        self.test = test
        self.display = display
        self.num_epochs = num_epochs
        self.alpha = 0.20
        self.gamma = 0.99
        self.eps   = 0.02
        self.cue_ball_states = []
        self.target_ball_states = []
        self.powers = [0, 500, 2000, 5000, 8000, 10000, 15000, 20000]
        self.thetas = np.linspace(0, 360, 720)
    
    def load_Q_tables(self):
        '''Loads all q-dicts from saved pkl files'''
        print("Loading all saved q-dicts:")
        for i in range(16):
            with open(f'q_dicts/q_dict_{i}.pkl', 'rb') as fp:
                self.Q.Q_dict_list[i] = pickle.load(fp)
                print(f"-> Loaded q_dict_{i} from file"
                      f"    -> size: {len(self.Q.Q_dict_list[i])}\n")

    def save_Q_tables(self):
        '''Saves all current q-dicts to their appropriate pkl file'''
        print("Saving all q-dicts:\n")
        for i in range(16):
            with open(f'q_dicts/q_dict_{i}.pkl', 'wb') as bb:
                pickle.dump(self.Q.Q_dict_list[i], bb)
                print(f"-> q-dict_{i} saved successfully to file\n"
                      f"    -> size: {len(self.Q.Q_dict_list[i])}")

    def begin(self):
        if self.test:
            self.begin_testing()
        else:
            self.begin_training()
    
    def begin_testing(self):
        print("\n**Testing Q-Learning (only exploiting best actions)**\n")

    def begin_training(self):
        print("\n**Training Q-Learning**\n")
        #self.load_Q_tables()
        for epoch in tqdm(range(self.num_epochs)):
            self.game = BILLIARDS_GAME_COMPUTER(self.display)
            self.game.setup_game()
            self.run_episode()
            self.save_Q_tables() # Save Q-tables every epoch

    def run_episode(self):
        cue_s = [888, 339]
        obj_s_list = [[250, 267], [250, 304], [250, 341], [250, 378], [250, 415], [287, 285], [287, 322], [287, 359], [287, 396], [324, 303], [324, 340], [324, 377], [361, 321], [361, 358], [398, 339]]
        while(self.game.run):
            # Choose to take a random action or exploit best
            theta, power = self.choose_actions(cue_s, obj_s_list)
            # Take a step in the game
            cue_sp, obj_sp_list, reward = self.take_action(cue_s, obj_s_list, theta, power)
            # Set current state to sp
            cue_s = cue_sp
            obj_s_list = obj_sp_list
            # Update q-table
            self.q_learn(cue_s, obj_s_list, theta, power, reward, cue_sp, obj_sp_list)
    
    def choose_actions(self, cue_s, obj_s_list):
        '''choose theta and power with eps-greedy approach'''
        if np.random.rand() < self.eps:
            theta, power, max_q = self.Q.get_max_action_pair(cue_s, obj_s_list)
        else:
            power = np.random.choice(self.powers)
            theta = np.random.choice(self.thetas)
        return theta, power
    
    def take_action(self, cue_s, obj_s_list, theta, power):
        cue_sp, obj_sp_list, reward = self.game.shoot(theta, power, cue_s, obj_s_list)
        return cue_sp, obj_sp_list, reward
    
    def q_learn(self, cue_s, obj_s_list, theta, power, reward, cue_sp, obj_sp_list):
        Q_current = self.Q.get_q_val(cue_s, obj_s_list, theta, power)
        #print(f"Q_current: {Q_current}")
        _, _, max_Qp    = self.Q.get_max_action_pair(cue_sp, obj_sp_list)
        #print(f"Max_Qp: {max_Qp}")
        Q_new     = self.alpha*(reward + self.gamma*max_Qp - Q_current)
        self.Q.update_q_dict(cue_s, obj_s_list, theta, power, Q_new)