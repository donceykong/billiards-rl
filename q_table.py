import numpy as np
import tqdm as tqdm

class Q_DICT:
    def __init__(self):
        cue_state_start=[0.00, 0.00]
        target_state_start=[0.00, 0.00]
        angle_start=0.00
        power_start=0.00
        reward=0.00
        self.Q_dict = {f'{cue_state_start}, {target_state_start}, {angle_start}, {power_start}': reward}
        self.angles = np.linspace(0, 360, num = 360*4) # List from 0 to 360, with theta_disc numbers
        max_power = 20000
        self.powers = np.linspace(100, max_power, num = 5) # List from 0 to max_power, with powers_disc numbers
        
    def init_q_dict(self, x_disc, y_disc, angles_disc, powers_disc):
        x_max = 10
        states_x = np.linspace(0, x_max, num = x_disc) # List from 0 to x_max, with x_discritized numbers

        y_max = 10
        states_y = np.linspace(0, y_max, num = y_disc) # List from 0 to y_max, with y_discritized numbers

        angles = np.linspace(0, 360, num = angles_disc) # List from 0 to 360, with theta_disc numbers

        max_power = 3600
        powers = np.linspace(0, max_power, num = powers_disc) # List from 0 to max_power, with powers_disc numbers
        print(powers)

        print(f"size dict list = {np.power(states_x.size*states_y.size, 2)}\n")

        for s_cue_x in tqdm(states_x):
            for s_cue_y in states_y:
                for s_target_x in states_x:
                    for s_target_y in states_y:
                        for angle in angles:
                            for power in powers:
                                cue_state = [s_cue_x, s_cue_y]
                                target_state = [s_target_x, s_target_y]
                                self.Q_dict.update({f'{cue_state}, {target_state}, {angle}, {power}': 0.00})

        print(len(self.Q_dict))

    def update_q_dict(self, cue_state, target_state, angle, power, reward):
        current_reward = self.get_q_val(cue_state, target_state, angle, power)
        if current_reward is not None:
            updated_reward = current_reward + reward
        else:
            updated_reward = reward       
        self.Q_dict.update({f'{cue_state}, {target_state}, {angle}, {power}': updated_reward})     

    def get_q_val(self, cue_state, target_state, angle, power):
        state_action_pair = f'{cue_state}, {target_state}, {angle}, {power}'
        if state_action_pair in self.Q_dict: 
            current_val = self.Q_dict[state_action_pair]
        else:
            current_val = None
        return current_val