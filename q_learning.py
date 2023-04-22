import numpy as np
from q_table import Q_DICT
import pickle 
from datetime import datetime
import os
import time
from billiards_game_computer import BILLIARDS_GAME_COMPUTER
from tqdm import tqdm

class Q_LEARNING:
    def __init__(self, test = True, display = False, num_epochs=10000, num_obj_balls = 15):
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
        self.num_obj_balls = num_obj_balls

    def load_Q_tables(self):
        '''Loads all q-dicts from saved pkl files'''
        print("Loading all saved q-dicts:")
        for i in range(16):
            if i < 10: dict_name = f'q_dicts/q_dict_0{i}.pkl'
            else: dict_name = f'q_dicts/q_dict_{i}.pkl'
            with open(dict_name, 'rb') as fp:
                self.Q.Q_dict_list[i] = pickle.load(fp)
                print(f"-> Loaded q_dict_{i} from file"
                      f"    -> size: {len(self.Q.Q_dict_list[i])}\n")

    def save_Q_tables(self):
        '''Saves all current q-dicts to their appropriate pkl file'''
        print("Saving all q-dicts:\n")
        for i in range(16):
            if i < 10: dict_name = f'q_dicts/q_dict_0{i}.pkl'
            else: dict_name = f'q_dicts/q_dict_{i}.pkl'
            with open(dict_name, 'wb') as bb:
                pickle.dump(self.Q.Q_dict_list[i], bb)
                print(f"-> q-dict_{i} saved successfully to file\n"
                      f"    -> size: {len(self.Q.Q_dict_list[i])}")
                
    def begin(self):
        #self.load_Q_tables()
        if self.test:
            print("\n**Testing Q-Learning (only exploiting best actions)**\n")
            for epoch in tqdm(range(self.num_epochs)):
                self.game = BILLIARDS_GAME_COMPUTER(self.display, self.num_obj_balls)
                self.game.setup_game()
                self.run_episode(self.num_obj_balls)
                self.save_Q_tables() # Save Q-tables every epoch
        else:
            print("\n**Training Q-Learning**\n")
            for epoch in tqdm(range(self.num_epochs)):
                self.game = BILLIARDS_GAME_COMPUTER(self.display, self.num_obj_balls)
                self.game.setup_game()
                self.run_episode(self.num_obj_balls)
                self.save_Q_tables() # Save Q-tables every epoch

    def run_episode(self, num_obj_balls):
        cue_s = [888, 339]
        obj_s_list = [[250, 267], [250, 304], [250, 341], [250, 378], [250, 415], [287, 285], [287, 322], [287, 359], [287, 396], [324, 303], [324, 340], [324, 377], [361, 321], [361, 358], [398, 339]]
        obj_s_list = obj_s_list[0:num_obj_balls]
        while(self.game.run):
            # Choose to take a random action or exploit best
            theta, power = self.choose_actions(cue_s, obj_s_list)
            # Take a step in the game
            cue_sp, obj_sp_list, reward = self.take_action(cue_s, obj_s_list, theta, power)
            # Update q-table
            self.q_learn(cue_s, obj_s_list, theta, power, reward, cue_sp, obj_sp_list)
            # Set current state to sp
            obj_s_list = obj_sp_list
            if cue_sp != [0, 0]:
                cue_s = cue_sp
            else:
                print ("Pocketed Cue Ball")
                # TODO: Need to add something to add cue to a random spot as long as not colliding with others
    
    def choose_actions(self, cue_s, obj_s_list):
        '''choose theta and power with eps-greedy approach'''
        if self.test:
            theta, power, max_q = self.Q.get_max_action_pair(cue_s, obj_s_list)
        else:
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
        _, _, max_Qp    = self.Q.get_max_action_pair(cue_sp, obj_sp_list)
        Q_new     = self.alpha*(reward + self.gamma*max_Qp - Q_current)
        #print(f"Q_current: {Q_current}, Max_Qp: {max_Qp}, Q_new: {Q_new}")
        self.Q.update_q_dict(cue_s, obj_s_list, theta, power, Q_new)

    def merge_Qs(dict_a, dict_b):
        dict_c = {**dict_a, **dict_b}
        for key, value in dict_c.items():
            if key in dict_a and key in dict_b:
               dict_c[key] = value+dict_a[key]
        return dict_c