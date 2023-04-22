import numpy as np
import tqdm as tqdm

'''

Classes:

    Q_DICT
    MAPPER
'''
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
            setattr(self, f'Q_dict_{i}', q_dict)
            
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