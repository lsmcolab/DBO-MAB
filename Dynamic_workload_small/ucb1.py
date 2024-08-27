from utils import max_
import math
import numpy as np


class UCB1():
    
    def __init__(self, c=1, actions=10):
        
        self.c = c
        self.actions = actions
        
        self.counts = [0 for col in range(self.actions)]
        
        self.values = [0.0 for col in range(self.actions)]
        
        self.action_total_reward = [0.0 for _ in range(self.actions)]
        self.action_avg_reward = [[] for action in range(self.actions)]
        
        return
    
    


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
        return max_(ucb_values)

    def update(self, chosen_act, reward):
        self.counts[chosen_act] = self.counts[chosen_act] + 1
        n = self.counts[chosen_act]
        
#     # Update average/mean value/reward for chosen action
        value = self.values[chosen_act]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        #new_value2 = value + (1/n * (reward - value))
        self.values[chosen_act] = new_value
        
        self.action_total_reward[chosen_act] += reward
        for a in range(self.actions):
            if self.counts[a]:
                self.action_avg_reward[a].append(self.action_total_reward[a]/self.counts[a])
            else:
                self.action_avg_reward[a].append(0)
        return
    
    
class UCB1_multiple_C(UCB1):
    
    def __init__(self, 
                 add_points_function, 
                 remove_points_function,
                 c_list, 
                 actions=10, 
                 c_timestep_change=100,
                 verbose=False, 
                 c_min=None, 
                 c_max=None,
                 ):

        super().__init__(c=c_list[0], actions = actions)
        self.c_timestep_change = c_timestep_change
        self.num_actions = actions
        self.timestep = 0
        self.c_list = c_list
        self.c_ind  = 0
        self.add_points_function = add_points_function
        self.remove_points_function = remove_points_function
        
        self.c_min = c_min
        self.c_max = c_max
        
        # c is chosen to be the first in the c_list
        self.c = self.c_list[self.c_ind]
        self.c_history = []
        self.removed_points= []
        self.unique_c_history = []
        self.previous_c_values = []
        self.previous_c_scores = []
        self.verbose = verbose
        self.average_response_time = []
        
        
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
    

    
    def update(self, chosen_act, reward,delay):
        self.c_history.append(self.c)
        self.counts[chosen_act] = self.counts[chosen_act] + 1
        n = self.counts[chosen_act]
        
        
        
        
#     # Update average/mean value/reward for chosen action
        value = self.values[chosen_act]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        #new_value2 = value + (1/n * (reward - value))
        self.values[chosen_act] = new_value
        
        self.action_total_reward[chosen_act] += reward
        for a in range(self.actions):
            if self.counts[a]:
                self.action_avg_reward[a].append(self.action_total_reward[a]/self.counts[a])
            else:
                self.action_avg_reward[a].append(0)
                
        #self.average_response_time.append(np.mean(self.action_avg_reward))
        
        # if it's time to change c, then change c
        # we only change c every x timesteps, e.g. x=10, then we change c when timestep = 0, 10, 20, 30
        # if timestep=20, and c_timestep_change=10, we change c
        if self.timestep%self.c_timestep_change==0:
            #if self.verbose:
                #print("Selecting new c value - timestep ", self.timestep)
            
            #if len(self.previous_c_scores) == 0:
            c_score =self.track_response_time(delay)
            # else:
            #     # [-1] means the last element from the list
            #     c_score = self.average_response_time[-1] - self.previous_c_scores[-1]
            
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
                points_next_round, scores_next_round = self.remove_points_function(self.points_next_round,
                                                                                     self.scores_next_round,
                                                                                     actions=self.num_actions,
                                                                                     c_value = self.c,
                                                                                    current_cycle=self.timestep,
                                                                                   current_iteration=self.timestep
                                                                                  )
                #print("items coming out of the remove points function")
                #print(len(np.array(points_next_round)), len(np.array(scores_next_round)))
                
                
                
                
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
    
    
class UCB1_():
    
    def __init__(self, c=1, actions=10):
        
        self.c = c
        self.actions = actions
        
        self.counts = [0 for col in range(self.actions)]
        
        self.values = [0.0 for col in range(self.actions)]
        
        self.action_total_reward = [0.0 for _ in range(self.actions)]
        self.action_avg_reward = [[] for action in range(self.actions)]
        
        return
    
    def max_(self,values):
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
            bonus =   (math.sqrt((2 * math.log(total_counts)) / float(self.counts[action])))
            ucb_values[action] = self.values[action] + bonus
        return self.max_(ucb_values)

    def update(self, chosen_act, reward):
        self.counts[chosen_act] = self.counts[chosen_act] + 1
        n = self.counts[chosen_act]
        
#     # Update average/mean value/reward for chosen action
        value = self.values[chosen_act]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        #new_value2 = value + (1/n * (reward - value))
        self.values[chosen_act] = new_value
        
        self.action_total_reward[chosen_act] += reward
        for a in range(self.actions):
            if self.counts[a]:
                self.action_avg_reward[a].append(self.action_total_reward[a]/self.counts[a])
            else:
                self.action_avg_reward[a].append(0)
        return