from collections import deque
import math
import numpy as np

#SW-UCB
class SW_UCB():
    
    def __init__(self, c=1, actions=10, tau=100):
        
        self.c = c
        self.actions = actions
        self.tau = tau  # window size
        
        self.counts = [0 for col in range(self.actions)]
        
        self.values = [0.0 for col in range(self.actions)]
        self.rewards_window = [[] for _ in range(self.actions)]  # store recent rewards
        
        self.action_total_reward = [0.0 for _ in range(self.actions)]
        self.action_avg_reward = [[] for action in range(self.actions)]
        
        return
    
    def max_(self, values):
        max_index = 0
        maxv = values[max_index]
        for i in range(len(values)):
            if values[i] > maxv:
                maxv = values[i]
                max_index = i
        return max_index

    def select_action(self):
        actions = len(self.counts)
        for action in range(actions):
            if self.counts[action] == 0:
                return action
    
        ucb_values = [0.0 for action in range(actions)]
        total_counts = sum(self.counts)
        for action in range(actions):
            bonus =  self.c * (math.sqrt((2 * math.log(total_counts)) / float(self.counts[action])))
            ucb_values[action] = self.values[action] + bonus
        return self.max_(ucb_values)

    def update(self, chosen_act, reward):
        # Add the new reward to the window
        self.rewards_window[chosen_act].append(reward)
        
        # If the window size is exceeded, remove the oldest reward
        if len(self.rewards_window[chosen_act]) > self.tau:
            self.rewards_window[chosen_act].pop(0)
        
        # Update counts
        self.counts[chosen_act] = len(self.rewards_window[chosen_act])
        
        # Update average/mean value/reward for chosen action
        self.values[chosen_act] = sum(self.rewards_window[chosen_act]) / len(self.rewards_window[chosen_act])
        
        self.action_total_reward[chosen_act] += reward
        for a in range(self.actions):
            if self.counts[a]:
                self.action_avg_reward[a].append(self.action_total_reward[a]/self.counts[a])
            else:
                self.action_avg_reward[a].append(0)
        
        return
    
    
###
class UCB1_multiple_C_SW(SW_UCB):
    
    def __init__(self, 
                 add_points_function, 
                 remove_points_function,
                 c_list, 
                 actions=10, 
                 c_timestep_change=100,
                 verbose=False,
                 c_min=None, 
                 c_max=None
                 ):

        super().__init__(c=c_list[0], actions = actions, tau=200)
        self.c_timestep_change = c_timestep_change
        self.num_actions = actions
        self.timestep = 0
        self.c_list = c_list
        self.c_ind  = 0
        self.add_points_function = add_points_function
        self.remove_points_function = remove_points_function      
        
        # c is chosen to be the first in the c_list
        self.c = self.c_list[self.c_ind]
        self.c_history = []
        self.removed_points= []
        self.unique_c_history = []
        self.previous_c_values = []
        self.previous_c_scores = []
        self.verbose = verbose
        self.average_response_time = []
        
               
        
        self.c_min = c_min
        self.c_max = c_max
            
        # Additional attributes for response time tracking
        self.total_response_time = 0.0
        self.total_responses = 0
        c_score = 0.0  # Initialize c_score
        

        
        
        return
    
    def track_response_time(self, delay):
        """
        Tracks the response time for each action and calculates the average response time.
        """
        self.total_response_time += delay
        self.total_responses += 1
        self.average_response_time = self.total_response_time / self.total_responses if self.total_responses else 0

        # Directly assign the average response time to c_score
        self.c_score = self.average_response_time
        return self.c_score
    
    def update(self, chosen_act, reward, delay):
        self.c_history.append(self.c)
        
        
        
        
#     # Add the new reward to the window
        self.rewards_window[chosen_act].append(reward)
        
        # If the window size is exceeded, remove the oldest reward
        if len(self.rewards_window[chosen_act]) > self.tau:
            self.rewards_window[chosen_act].pop(0)
        
        # Update counts
        self.counts[chosen_act] = len(self.rewards_window[chosen_act])
        
        # Update average/mean value/reward for chosen action
        self.values[chosen_act] = sum(self.rewards_window[chosen_act]) / len(self.rewards_window[chosen_act])
        
        self.action_total_reward[chosen_act] += reward
        for a in range(self.actions):
            if self.counts[a]:
                self.action_avg_reward[a].append(self.action_total_reward[a]/self.counts[a])
            else:
                self.action_avg_reward[a].append(0)
        
        # if it's time to change c, then change c
        if self.timestep%self.c_timestep_change==0:
            #if self.verbose:
                #print("Selecting new c value - timestep ", self.timestep)
            
            #if len(self.previous_c_scores)==0:
                #c_score = np.sum(self.action_total_reward)
            c_score =self.track_response_time(delay)
            #else:
                #c_score = np.sum(self.action_total_reward)-self.previous_c_scores[-1]
            
            # this is the case the timestep is still using the list of values
            if self.timestep<(self.c_timestep_change*len(self.c_list)-1):
                self.previous_c_values.append(self.c)
                self.previous_c_scores.append(c_score)
            
                #print("cycling through list of c values entered")
                #if self.verbose:
                    #print("Using c value from input c values")
                    
                # go to the next c value
                self.c_ind+=1
                self.c = self.c_list[self.c_ind%len(self.c_list)]
                self.unique_c_history.append(self.c)
                #print("new c value: ")
                #print(self.c)
                
                self.points_next_round = self.previous_c_values
                self.scores_next_round = self.previous_c_scores
            else:
                
                self.points_next_round = np.concatenate((self.points_next_round, np.array([self.c])))
                self.scores_next_round = np.concatenate((self.scores_next_round, np.array([c_score])))
                
                #print("creating a new c values using the GP")
                #print("items going into the remove points function")
                #print(len(self.points_next_round), len(self.scores_next_round))
                points_next_round, scores_next_round = self.remove_points_function(
                                                                                    self.points_next_round,
                                                                                    self.scores_next_round,
                                                                                    actions=self.num_actions,
                                                                                    c_value=self.c,
                                                                                    current_cycle=self.timestep,
                                                                                    current_iteration=self.timestep
                                                                                    
                                                                                )
                #print("items coming out of the remove points function")
                #print(len(np.array(points_next_round)), len(np.array(scores_next_round)))
                
                #print("Before removing entries:")
                #print("points_next_round:", points_next_round)
                #print("self.c_to_action_map:", self.c_to_action_map)
                # Remove the entry for any c value that's no longer in points_next_round
                               
                self.points_next_round = points_next_round
                self.scores_next_round = scores_next_round
                
                #print("len of c_values: ", len(self.points_next_round))
                #print("len of s_scores: ", len(self.scores_next_round))  
                
                new_points = self.add_points_function(self.points_next_round,
                                                      self.scores_next_round,
                                                      actions=self.num_actions,
                                                      c_min=self.c_min, 
                                                      c_max=self.c_max)
                
                self.c = new_points[0]
                #print("new c value selected to be: ")  
                #print(self.c)
                
                self.unique_c_history.append(self.c)
                
        self.timestep += 1
        
            
        return

########## ## Discounted UCB

class Discounted_UCB():
    
    def __init__(self, c=1, actions=10, gamma=0.90):
        
        self.c = c
        self.actions = actions
        self.gamma = gamma  # discount factor
        
        self.counts = [0 for col in range(self.actions)]
        self.values = [0.0 for col in range(self.actions)]
        
        self.action_total_reward = [0.0 for _ in range(self.actions)]
        self.action_avg_reward = [[] for action in range(self.actions)]
        
        return
    
    def max_(self, values):
        max_index = 0
        maxv = values[max_index]
        for i in range(len(values)):
            if values[i] > maxv:
                maxv = values[i]
                max_index = i
        return max_index

    def select_action(self):
        actions = len(self.counts)
        for action in range(actions):
            if self.counts[action] == 0:
                return action
    
        ucb_values = [0.0 for action in range(actions)]
        total_counts = sum(self.counts)
        for action in range(actions):
            bonus =  self.c * (math.sqrt((2 * math.log(total_counts)) / float(self.counts[action])))
            ucb_values[action] = self.values[action] + bonus
        return self.max_(ucb_values)

    def update(self, chosen_act, reward):
        # Discount all previous rewards for the chosen action
        self.values[chosen_act] = self.gamma * self.values[chosen_act] + reward
        self.counts[chosen_act] += 1
        
        
        self.action_total_reward[chosen_act] += reward
        for a in range(self.actions):
            if self.counts[a]:
                self.action_avg_reward[a].append(self.action_total_reward[a]/self.counts[a])
            else:
                self.action_avg_reward[a].append(0)
        
        return

##
class Discounted_UCB_Pillar(Discounted_UCB):
    
    def __init__(self, 
                 add_points_function, 
                 remove_points_function,
                 c_list, 
                 actions=10, 
                 c_timestep_change=100,
                 gamma = 0.88,
                 verbose=False,
                 c_min=None, 
                 c_max=None
                 ):

        super().__init__(c=c_list[0], actions = actions, gamma=gamma)
        self.c_timestep_change = c_timestep_change
        self.num_actions = actions
        self.timestep = 0
        self.c_list = c_list
        self.c_ind  = 0
        self.add_points_function = add_points_function
        self.remove_points_function = remove_points_function      
        
        # c is chosen to be the first in the c_list
        self.c = self.c_list[self.c_ind]
        self.c_history = []
        self.removed_points= []
        self.unique_c_history = []
        self.previous_c_values = []
        self.previous_c_scores = []
        self.verbose = verbose
        self.average_response_time = []
      
        self.c_min = c_min
        self.c_max = c_max
            
        # Additional attributes for response time tracking
        self.total_response_time = 0.0
        self.total_responses = 0
        c_score = 0.0  # Initialize c_score
        
        
        
        return
    
    def track_response_time(self, delay):
        """
        Tracks the response time for each action and calculates the average response time.
        """
        self.total_response_time += delay
        self.total_responses += 1
        self.average_response_time = self.total_response_time / self.total_responses if self.total_responses else 0

        # Directly assign the average response time to c_score
        self.c_score = self.average_response_time
        return self.c_score
    
    def update(self, chosen_act, reward, delay):
        self.c_history.append(self.c)
        
        
        
        
#     # Discount all previous rewards for the chosen action
        self.values[chosen_act] = self.gamma * self.values[chosen_act] + reward
        self.counts[chosen_act] += 1
        
        self.action_total_reward[chosen_act] += reward
        for a in range(self.actions):
            if self.counts[a]:
                self.action_avg_reward[a].append(self.action_total_reward[a]/self.counts[a])
            else:
                self.action_avg_reward[a].append(0)
        
        # if it's time to change c, then change c
        if self.timestep%self.c_timestep_change==0:
            #if self.verbose:
                #print("Selecting new c value - timestep ", self.timestep)
            
            #if len(self.previous_c_scores)==0:
                #c_score = np.sum(self.action_total_reward)
            c_score =self.track_response_time(delay)
            #else:
                #c_score = np.sum(self.action_total_reward)-self.previous_c_scores[-1]
            
            # this is the case the timestep is still using the list of values
            if self.timestep<(self.c_timestep_change*len(self.c_list)-1):
                self.previous_c_values.append(self.c)
                self.previous_c_scores.append(c_score)
            
                #print("cycling through list of c values entered")
                #if self.verbose:
                    #print("Using c value from input c values")
                    
                # go to the next c value
                self.c_ind+=1
                self.c = self.c_list[self.c_ind%len(self.c_list)]
                self.unique_c_history.append(self.c)
                #print("new c value: ")
                #print(self.c)
                
                self.points_next_round = self.previous_c_values
                self.scores_next_round = self.previous_c_scores
            else:
                
                self.points_next_round = np.concatenate((self.points_next_round, np.array([self.c])))
                self.scores_next_round = np.concatenate((self.scores_next_round, np.array([c_score])))
                
                #print("creating a new c values using the GP")
                #print("items going into the remove points function")
                #print(len(self.points_next_round), len(self.scores_next_round))
                points_next_round, scores_next_round = self.remove_points_function(
                                                                                    self.points_next_round,
                                                                                    self.scores_next_round,
                                                                                    actions=self.num_actions,
                                                                                    c_value=self.c,
                                                                                    current_cycle=self.timestep,
                                                                                    current_iteration=self.timestep
                                                                                )
                #print("items coming out of the remove points function")
                #print(len(np.array(points_next_round)), len(np.array(scores_next_round)))
                
                #print("Before removing entries:")
                #print("points_next_round:", points_next_round)
                #print("self.c_to_action_map:", self.c_to_action_map)
                # Remove the entry for any c value that's no longer in points_next_round
                
                
                
                self.points_next_round = points_next_round
                self.scores_next_round = scores_next_round
                
                #print("len of c_values: ", len(self.points_next_round))
                #print("len of s_scores: ", len(self.scores_next_round))  
                
                new_points = self.add_points_function(self.points_next_round,
                                                      self.scores_next_round,
                                                      actions=self.num_actions,
                                                      c_min=self.c_min, 
                                                      c_max=self.c_max)
                
                self.c = new_points[0]
                #print("new c value selected to be: ")  
                #print(self.c)
                
                self.unique_c_history.append(self.c)
                
        self.timestep += 1
        
            
        return
########## SW-TS
class ThompsonSamplingSW:

    def __init__(self, actions=20, mu_0=0, lambda_0=1, alpha_0=1, beta_0=1, window_size=100):
        # Existing code for initialization
        self.actions = actions

        # Hyperparameters for the Normal-Gamma prior
        self.mu_0 = mu_0
        self.lambda_0 = lambda_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        self.counts = np.zeros(self.actions)
        self.sum_x = np.zeros(self.actions)
        self.sum_x_square = np.zeros(self.actions)
        
        
        self.window_size = window_size
        self.reward_windows = [deque(maxlen=window_size) for _ in range(actions)]
        
    def max_(self, values):
        max_index = 0
        maxv = values[max_index]
        for i in range(len(values)):
            if values[i] > maxv:
                maxv = values[i]
                max_index = i
        return max_index

    def select_action(self):
        samples = np.zeros(self.actions)
        for k in range(self.actions):
            window_rewards = self.reward_windows[k]
            recent_counts = len(window_rewards)

            # Calculate statistics based on window rewards
            recent_sum_x = sum(window_rewards)
            recent_sum_x_square = sum(r**2 for r in window_rewards)

            tau_k = np.random.gamma(self.alpha_0 + recent_counts / 2, 1 / (self.beta_0 + 0.5 * (recent_sum_x_square - (recent_sum_x ** 2) / (recent_counts + 1))))
            mu_k = np.random.normal(recent_sum_x / (self.lambda_0 + recent_counts), 1 / np.sqrt(tau_k * (self.lambda_0 + recent_counts)))
            samples[k] = mu_k
        return self.max_(samples)

    def update(self, chosen_act, reward):
        # Update counts and sums with new reward
        self.counts[chosen_act] += 1
        self.sum_x[chosen_act] += reward
        self.sum_x_square[chosen_act] += reward ** 2

        # Update the sliding window
        self.reward_windows[chosen_act].append(reward)

########## DS-TS
class ThompsonSamplingDS:

    def __init__(self, actions=20, mu_0=0, lambda_0=1, alpha_0=1, beta_0=1, discount_factor=0.9):
        self.actions = actions

        # Hyperparameters for the Normal-Gamma prior
        self.mu_0 = mu_0
        self.lambda_0 = lambda_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        self.counts = np.zeros(self.actions)
        self.sum_x = np.zeros(self.actions)
        self.sum_x_square = np.zeros(self.actions)

        # Discount factor for non-stationarity
        self.discount_factor = discount_factor

    def max_(self, values):
        max_index = 0
        maxv = values[max_index]
        for i in range(len(values)):
            if values[i] > maxv:
                maxv = values[i]
                max_index = i
        return max_index

    def select_action(self):
        samples = np.zeros(self.actions)
        for k in range(self.actions):
            # Discounted statistics for prior
            discounted_lambda = self.lambda_0 + self.discount_factor * self.counts[k]
            discounted_alpha = self.alpha_0 + self.discount_factor * self.counts[k] / 2
            discounted_sum_x = self.discount_factor * self.sum_x[k]
            discounted_sum_x_square = self.discount_factor * self.sum_x_square[k]

            # Sample mean and precision
            tau_k = np.random.gamma(discounted_alpha, 1 / (self.beta_0 + 0.5 * (discounted_sum_x_square - (discounted_sum_x ** 2) / (discounted_lambda + 1))))
            mu_k = np.random.normal(discounted_sum_x / discounted_lambda, 1 / np.sqrt(tau_k * discounted_lambda))
            samples[k] = mu_k

        return self.max_(samples)

    def update(self, chosen_act, reward):
        # Update counts and sums with discounted values
        self.counts[chosen_act] *= self.discount_factor
        self.sum_x[chosen_act] *= self.discount_factor
        self.sum_x_square[chosen_act] *= self.discount_factor

        # Update with new reward
        self.counts[chosen_act] += 1
        self.sum_x[chosen_act] += reward
        self.sum_x_square[chosen_act] += reward ** 2


##### ThompsonSampling_fDSW

class ThompsonSampling_fDSW:

    def __init__(self, actions=20, mu_0=0, lambda_0=1, alpha_0=1, beta_0=1, window_size=100, discount_factor=0.9):
        self.actions = actions

        # Hyperparameters for the Normal-Gamma prior
        self.mu_0 = mu_0
        self.lambda_0 = lambda_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        self.counts = np.zeros(self.actions)  # Counts with discounting
        self.sum_x = np.zeros(self.actions)  # Sum of rewards with discounting
        self.sum_x_square = np.zeros(self.actions)  # Sum of squared rewards with discounting

        self.window_size = window_size
        self.reward_windows = [deque(maxlen=window_size) for _ in range(actions)]
        self.discount_factor = discount_factor

    def select_action(self):
        samples = np.zeros(self.actions)
        for k in range(self.actions):
            # Discounted historical data
            historical_alpha = self.alpha_0 + self.counts[k] / 2
            historical_beta = self.beta_0 + 0.5 * (self.sum_x_square[k] - (self.sum_x[k] ** 2) / (self.counts[k] + 1))
            
            # Sliding window data
            window_rewards = np.array(self.reward_windows[k])
            window_alpha = self.alpha_0 + len(window_rewards) / 2
            window_beta = self.beta_0 + 0.5 * (np.sum(window_rewards**2) - (np.sum(window_rewards) ** 2) / (len(window_rewards) + 1))
            
            # Aggregate using mean 
            tau_k = np.random.gamma((historical_alpha + window_alpha) / 2, 1 / ((historical_beta + window_beta) / 2))
            mu_k = np.random.normal(np.mean([self.sum_x[k] / (self.lambda_0 + self.counts[k]), np.sum(window_rewards) / (self.lambda_0 + len(window_rewards))]), 1 / np.sqrt(tau_k * (self.lambda_0 + self.counts[k] + len(window_rewards))))
            
            samples[k] = mu_k
        return np.argmax(samples)
    
    def update(self, chosen_act, reward):
        # Apply discount factor to historical data
        self.counts[chosen_act] *= self.discount_factor
        self.sum_x[chosen_act] *= self.discount_factor
        self.sum_x_square[chosen_act] *= self.discount_factor

        # Update historical data with new reward
        self.counts[chosen_act] += 1
        self.sum_x[chosen_act] += reward
        self.sum_x_square[chosen_act] += reward ** 2

        # Update the sliding window with the new reward
        self.reward_windows[chosen_act].append(reward)