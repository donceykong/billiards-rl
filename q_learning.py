import numpy as np
import pickle 
from datetime import datetime
import os
import time
from billiards_game_ql import BILLIARDS_GAME_COMPUTER
from billiards_game_dql import BILLIARDS_GAME_COMPUTER_DQL
from tqdm import tqdm
import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random

class DQ_LEARNING:
    def __init__(self, test=True, display=False, num_epochs=10000, num_obj_balls=15):
        self.test = test
        self.display = display
        self.num_epochs = num_epochs
        self.alpha = 0.20
        self.gamma = 0.99
        self.eps = 0.02
        self.memory = []
        self.memory_limit = 50000
        self.batch_size = 64
        self.powers = [0, 500, 2000, 5000, 8000, 10000, 15000, 20000]
        self.thetas = np.linspace(0, 359, 719)
        self.num_obj_balls = num_obj_balls
        self.model = self.create_model((1 + num_obj_balls) * 2, len(self.powers) * len(self.thetas))
        self.model_filename = "deep_q_model.h5"

    def create_model(self, input_size, output_size, learning_rate=0.001):
        model = Sequential()
        model.add(Dense(64, input_dim=input_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(output_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        return model

    def save_model(self, model, filename):
        model.save(filename)

    def load_model(self, filename):
        return keras.models.load_model(filename)

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.memory_limit:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def choose_actions(self, cue_s, obj_s_list, model):
        state = np.array([cue_s] + obj_s_list).flatten()
        if np.random.rand() < self.eps:
            action_values = model.predict(np.array([state]))[0]
            best_action_idx = np.argmax(action_values)
            theta, power = self.thetas[best_action_idx // len(self.powers)], self.powers[best_action_idx % len(self.powers)]
        else:
            power = np.random.choice(self.powers)
            theta = np.random.choice(self.thetas)
        return theta, power

    def replay(self, model, memory, batch_size=64):
        if len(memory) < batch_size:
            return

        minibatch = random.sample(memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(model.predict(np.array([next_state]))[0]))
            target_f = model.predict(np.array([state]))
            #print(f"target_f: {target_f}, target_f[0][int(action)]: {target_f[0][int(action)]}")
            target_f[0][int(action)] = target

            history = model.fit(np.array([state]), target_f, epochs=1, verbose=0)
            loss_values = history.history["loss"]
            print(f'Loss values at each epoch: {loss_values}')

    def begin(self):
        if self.test:
            print("\n**Testing Deep Q-Learning (only exploiting best actions)**\n")
            self.model = self.load_model(self.model_filename)
        else:
            print("\n**Training Deep Q-Learning**\n")

        for episode in tqdm(range(self.num_epochs)):
            self.game = BILLIARDS_GAME_COMPUTER_DQL(self.display, self.num_obj_balls)
            self.game.setup_game()
            cue_s = [888, 339]
            obj_s_list = [[250, 267], [250, 304], [250, 341], [250, 378], [250, 415], [287, 285], [287, 322], [287, 359], [287, 396], [324, 303], [324, 340], [324, 377], [361, 321], [361, 358], [398, 339]]
            state = np.array([cue_s] + obj_s_list).flatten()
            print(state)
            done = False

            while not done:
                theta, power = self.choose_actions(cue_s, obj_s_list,self.model)

                cue_sp, obj_sp_list, reward = self.game.step(theta, power, cue_s, obj_s_list)
                done = not self.game.run

                next_state = np.array([cue_sp] + obj_sp_list).flatten()
                #action = np.array([theta, power]).flatten()
                print(f"next_state: {next_state}, action: {theta * len(self.powers) + self.powers.index(power)}")

                self.remember(state, theta * len(self.powers) + self.powers.index(power), reward, next_state, done)

                if cue_sp == [0, 0] or (self.test and reward <= 0):
                    obj_s_list = obj_s_list
                    cue_s = cue_s
                else:
                    obj_s_list = obj_sp_list
                    cue_s = cue_sp
                    # TODO: Need to add something to add cue to a random spot as long as not colliding with others
                state = np.array([cue_s] + obj_s_list).flatten()

                self.replay(self.model, self.memory, self.batch_size)

                if done:
                    print(f"Episode: {episode}, Reward: {reward}")

            if (episode + 1) % 10 == 0:
                self.save_model(self.model, self.model_filename)
################################################################################################

################################################################################################
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
        self.thetas = np.linspace(0, 359, 719)
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
        self.load_Q_tables()
        #self.game = BILLIARDS_GAME_COMPUTER(self.display, self.num_obj_balls)
        #self.game.setup_game()
        if self.test:
            print("\n**Testing Q-Learning (only exploiting best actions)**\n")
            for episode in tqdm(range(self.num_epochs)):
                self.game = BILLIARDS_GAME_COMPUTER(self.display, self.num_obj_balls)
                self.game.setup_game()
                self.run_episode(self.num_obj_balls)
                self.save_Q_tables() # Save Q-tables every epoch
        else:
            print("\n**Training Q-Learning**\n")
            for episode in tqdm(range(self.num_epochs)):
                self.game = BILLIARDS_GAME_COMPUTER(self.display, self.num_obj_balls)
                self.game.setup_game()
                self.run_episode(self.num_obj_balls)
                self.save_Q_tables() # Save Q-tables every epoch

    def run_episode(self, num_obj_balls):
        cue_s = [888, 339]
        obj_s_list = [[250, 267], [250, 304], [250, 341], [250, 378], [250, 415], [287, 285], [287, 322], [287, 359], [287, 396], [324, 303], [324, 340], [324, 377], [361, 321], [361, 358], [398, 339]]
        obj_s_list = obj_s_list[0:num_obj_balls]
        while(self.game.run):
            print("1) Choosing Actions:")
            # Choose to take a random action or exploit best
            theta, power = self.choose_actions(cue_s, obj_s_list)
            print(" -> Done with 1.")
            # Take a step in the game
            print("2) Taking Actions:")
            cue_sp, obj_sp_list, reward = self.take_action(cue_s, obj_s_list, theta, power)
            print(" -> Done with 2.")
            # Update q-table
            print("3) Q-Learning:")
            self.q_learn(cue_s, obj_s_list, theta, power, reward, cue_sp, obj_sp_list)
            print(" -> Done with 3.")
            print(f"REWARD: {reward}")
            # Set current state to sp
            if cue_sp == [0, 0] or (self.test and reward <= 0):
                obj_s_list = obj_s_list
                cue_s = cue_s
            else:
                obj_s_list = obj_sp_list
                cue_s = cue_sp
                # TODO: Need to add something to add cue to a random spot as long as not colliding with others
    
    def choose_actions(self, cue_s, obj_s_list):
        '''choose theta and power with eps-greedy approach'''
        if self.test:
            if np.random.rand() < 1.0:
                theta, power, max_q = self.Q.get_max_action_pair(cue_s, obj_s_list)
            else:
                power = np.random.choice(self.powers)
                theta = np.random.choice(self.thetas)
        else:
            if np.random.rand() < self.eps:
                theta, power, max_q = self.Q.get_max_action_pair(cue_s, obj_s_list)
            else:
                power = np.random.choice(self.powers)
                theta = np.random.choice(self.thetas)
        print(f"Angle: {theta}, Power: {power}")
        return theta, power
    
    def take_action(self, cue_s, obj_s_list, theta, power):
        cue_sp, obj_sp_list, reward = self.game.step(theta, power, cue_s, obj_s_list)
        return cue_sp, obj_sp_list, reward
    
    def q_learn(self, cue_s, obj_s_list, theta, power, reward, cue_sp, obj_sp_list):
        Q_current = self.Q.get_q_val(cue_s, obj_s_list, theta, power)
        _, _, max_Qp = self.Q.get_max_action_pair(cue_sp, obj_sp_list)
        Q_new = self.alpha*(reward + self.gamma*max_Qp - Q_current)
        #print(f"Q_current: {Q_current}, Max_Qp: {max_Qp}, Q_new: {Q_new}")
        self.Q.update_q_dict(cue_s, obj_s_list, theta, power, Q_new)
    
################################################################################################

################################################################################################
class Q_DICT:
    ''' 
    A class that can be used to build a Q-table for Q-learning using Python Dictionary
    data structures.

    ...

    Attributes
    ----------
    Q_dict_0, Q_dict_1, ..., Q_dict_15 : dict
        The seperate q-dicts that will store a value for a given state-action pair. There are 16 different 
        q-dicts - one for each size of object balls remaining on table.
    Q_dict_list : list
        list of q_dicts
    mapper : MAPPER Object
        An object of the MAPPER class. 

    Methods
    -------
    convert_to_hexstring(cue_s, obj_s_list, angle, power):
        Converts state-action list to a string with states being in hex format.
    update_q_dict(cue_s, obj_s_list, angle, power, reward):
        Updates a q-dict for a specific SA pair.
    get_q_val(self, cue_s, obj_s_list, angle, power):
        Returns value for a given state-action pair.
    get_max_action_pair(cue_s, obj_s_list):
        Returns action set to maximize reward for a given state, as well as what that reward is.
    '''
    def __init__(self):
        self.mapper = MAPPER()
        cue_s_start = [95, 95]
        obj_s_start = []
        theta_start = 0.00
        power_start = 0
        reward      = 0.00
        self.powers = [0, 500, 2000, 5000, 8000, 10000, 15000, 20000]
        self.thetas = np.linspace(0, 360, 720)
        self.Q_dict_pointer = None
        self.Q_dict_list = [{} for _ in range(16)]
        for i, q_dict in enumerate(self.Q_dict_list):
            if i < 10:
                dict_name = f'Q_dict_0{i}'
            else:
                dict_name = f'Q_dict_{i}'
            setattr(self, dict_name, q_dict)
            
        for _ in range(16):
            SA_string = self.convert_to_hexstring(cue_s_start, obj_s_start, theta_start, power_start)
            self.Q_dict_pointer                = {SA_string: reward}
            self.Q_dict_list[len(obj_s_start)] = self.Q_dict_pointer
            obj_s_start.append([95, 95])
    
    def convert_to_hexstring(self, cue_s, obj_s_list, angle, power):
        cue_s = self.mapper.map_pos_to_1D(cue_s)
        obj_s_list = [self.mapper.map_pos_to_1D(obj_s) for obj_s in obj_s_list]
        ''' Converts state-action (1D-converted) list to a string with states being in hex format '''
        if len(obj_s_list) > 0:
            obj_s_list.sort()
            cue_s_hex      = hex(cue_s).lstrip("0x")
            obj_s_list_hex = [hex(obj_state).lstrip('0x') for obj_state in obj_s_list]
            SA_pair        = f"{cue_s_hex}|"
            for obj_s_hex in obj_s_list_hex:
                SA_pair += obj_s_hex + ","
            SA_pair = SA_pair.rstrip(",") + f"|{angle}|{power}"
            return SA_pair
    
    def update_q_dict(self, cue_s, obj_s_list, theta, power, reward):
        '''Updates a q-dict for a specific SA pair'''
        current_reward = self.get_q_val(cue_s, obj_s_list, theta, power)
        if current_reward is not None:
            updated_reward = current_reward + reward
        else:
            updated_reward = reward
        SA_string = self.convert_to_hexstring(cue_s, obj_s_list, theta, power)
        self.Q_dict_list[len(obj_s_list)].update({SA_string: updated_reward})  
        #self.Q_dict_pointer.update({SA_string: updated_reward})     
    
    def get_q_val(self, cue_s, obj_s_list, angle, power):
        SA_string = self.convert_to_hexstring(cue_s, obj_s_list, angle, power)
        #self.Q_dict_list[len(obj_s_list)] = self.Q_dict_pointer
        if SA_string in self.Q_dict_list[len(obj_s_list)]: 
            current_val = self.Q_dict_list[len(obj_s_list)][SA_string] #self.Q_dict_pointer[SA_string]
        else:
            current_val = 0.00
        return current_val
    
    def get_max_action_pair(self, cue_s, obj_s_list):
        max_q      = 0.00
        best_angle = np.random.choice(self.thetas)
        best_power = np.random.choice(self.powers)
        for theta in self.thetas:
            for power in self.powers:
                q_val = self.get_q_val(cue_s, obj_s_list, theta, power)
                if q_val > max_q:
                    max_q      = q_val
                    best_angle = theta
                    best_power = power                  

        return best_angle, best_power, max_q

################################################################################################

################################################################################################
class MAPPER:
    '''
    The MAPPER class is helper class to convert a 2D position array to a 1D value.

    ...

    Attributes
    ----------
    min_x : int
        The minimum position along the x-axis a ball may have.
    max_x : int
        The maximum position along the x-axis a ball may have.
    min_y : int
        The minimum position along the y-axis a ball may have.
    max_y : int
        The maximum position along the y-axis a ball may have.
    x : np array
        The nd array that helps with major-column expression of states.

    Methods
    -------
    build_map():
        This method builds the np.array for x given all possible states.
    map_pos_to_1D(pos):
        This method maps a 2D position to a 1D value. 
    '''
    def __init__(self):
        self.min_x = 0
        self.max_x = 1140
        self.min_y = 0
        self.max_y = 618
        self.x = self.build_map()

    def build_map(self):
        '''This method builds the np.array'''
        states_x_list = np.linspace(self.min_x, self.max_x, num = (self.max_x-self.min_x)+1)
        states_y_list = np.linspace(self.min_y, self.max_y, num = (self.max_y-self.min_y)+1)
        row_list = []
        num = 1
        for _ in states_y_list:
            row = []
            for _ in states_x_list:
                row.append(num)
                num += 1
            row_list.append(row)
        return np.array(row_list, np.int32, order='F')

    def map_pos_to_1D(self, pos):
        '''This method maps a 2D position to a 1D value'''
        if pos[0] < self.min_x or pos[0] > self.max_x or pos[1] < self.min_y or pos[1] > self.max_y:
            raise Exception(f"The position [{pos[0]}, {pos[1]}] is out of bounds!"
                            f" Please note the min pos is [{self.min_x}, {self.min_y}] and the max pos is [{self.max_x}, {self.max_y}].")
        else:
            return self.x[pos[1]-self.min_x, pos[0]-self.min_y]
