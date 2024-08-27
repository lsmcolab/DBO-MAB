from Dynamic_algo import *
import numpy as np
import time
from environments import Env, DynamicEnv, DynamicEnvOrdered
from simulation import do_run

def test_method_change_c_SW(mus_all, sigmas_all, method, start_size=20, init_method=None, change_every=2000, verbose=True):
    
    
    n_steps = 10000
    
    if len(mus_all.shape) == 1:
        mus_all = np.array([mus_all])
        sigmas_all = np.array([sigmas_all])
        change_every = n_steps + 1
        #print("mus_all.shape: ", mus_all.shape)
    
    if not init_method:
        new_points = np.random.uniform(low=0.0, high=10.0, size=start_size)
        #new_points = np.random.beta(100, 10, size=start_size)
        #new_points = np.random.beta(100, 10, size=start_size)
        #new_points = np.random.uniform(c_range[0], c_range[1], size=start_size)
        #new_points = 10**np.random.uniform(np.log10(c_range[0]), np.log10(c_range[1]),size=50)
    else:
        new_points = init_method(size=start_size)

    add_points_function    = method[0]
    remove_points_function = method[1]
        
    point_history = []
    score_history = []
    best_chosen_history = []
    all_points = np.array([])
    all_points_scores = np.array([])
    Avg_res = []
    Best_action = []
    start = time.time()
    
    if len(mus_all.shape)==1:
        num_actions = mus_all.shape[0]
    else:
        num_actions = mus_all.shape[1]
    
    #print(new_points)
    # select_action_class = UCB1(actions = num_actions)
    select_action_class = UCB1_multiple_C_SW(c_list = new_points, 
                                          actions = num_actions, 
                                          c_timestep_change=40,
                                          add_points_function=add_points_function,
                                          remove_points_function=remove_points_function,
                                          verbose=verbose)
   
    if len(mus_all.shape)==1:
        env = Env(mus_all, sigmas_all)
    else:
        env = DynamicEnvOrdered(mus_all, sigmas_all, change_every=change_every)
        
    results = do_run(env, 
                     select_action_class,
                     n_steps=n_steps)
    
    response_times = results["average_response_time"]
    best_chosen = results["best_chosen"]
    
    Avg_res.append(response_times)
    Best_action.append(best_chosen)
        
    #print(results.keys())
    
    return np.array(Avg_res), np.array(Best_action),select_action_class, env



def test_method_change_c_DIS(mus_all, sigmas_all, method, start_size=20, init_method=None, change_every=2000, verbose=True):
    
    
    n_steps = 10000
    
    if len(mus_all.shape) == 1:
        mus_all = np.array([mus_all])
        sigmas_all = np.array([sigmas_all])
        change_every = n_steps + 1
        #print("mus_all.shape: ", mus_all.shape)
    
    if not init_method:
        new_points = np.random.uniform(low=0.0, high=10.0, size=start_size)
        #new_points = np.random.beta(100, 10, size=start_size)
        #new_points = np.random.beta(100, 10, size=start_size)
        #new_points = np.random.uniform(c_range[0], c_range[1], size=start_size)
        #new_points = 10**np.random.uniform(np.log10(c_range[0]), np.log10(c_range[1]),size=50)
    else:
        new_points = init_method(size=start_size)

    add_points_function    = method[0]
    remove_points_function = method[1]
        
    point_history = []
    score_history = []
    best_chosen_history = []
    all_points = np.array([])
    all_points_scores = np.array([])
    Avg_res = []
    Best_action = []
    start = time.time()
    
    if len(mus_all.shape)==1:
        num_actions = mus_all.shape[0]
    else:
        num_actions = mus_all.shape[1]
    
    #print(new_points)
    # select_action_class = UCB1(actions = num_actions)
    select_action_class = Discounted_UCB_Pillar(c_list = new_points, 
                                          actions = num_actions, 
                                          c_timestep_change=40,
                                          add_points_function=add_points_function,
                                          remove_points_function=remove_points_function,
                                          verbose=verbose)
   
    if len(mus_all.shape)==1:
        env = Env(mus_all, sigmas_all)
    else:
        env = DynamicEnvOrdered(mus_all, sigmas_all, change_every=change_every)
        
    results = do_run(env, 
                     select_action_class,
                     n_steps=n_steps)
    
    response_times = results["average_response_time"]
    best_chosen = results["best_chosen"]
    
    Avg_res.append(response_times)
    Best_action.append(best_chosen)
        
    #print(results.keys())
    
    return np.array(Avg_res), np.array(Best_action),select_action_class, env



def test_method_n_times_S(n, **kwargs):
    #results = []
    response_times_means = []
    best_chosen_step_prob = []
    for i in range(n):
        # result, response_times_mean,best_chosen_step = test_method4(**kwargs)
        mean_response, best_s_action,s,e = test_method_change_c_SW(**kwargs)
        response_times_means.append(mean_response)
        best_chosen_step_prob.append(best_s_action)
        
    Avg_res = np.array(response_times_means)
    Best_act = np.array(best_chosen_step_prob)
    return Avg_res, Best_act


def test_method_n_times_D(n, **kwargs):
    #results = []
    response_times_means = []
    best_chosen_step_prob = []
    for i in range(n):
        # result, response_times_mean,best_chosen_step = test_method4(**kwargs)
        mean_response, best_s_action,s,e = test_method_change_c_DIS(**kwargs)
        response_times_means.append(mean_response)
        best_chosen_step_prob.append(best_s_action)
        
    Avg_res = np.array(response_times_means)
    Best_act = np.array(best_chosen_step_prob)
    return Avg_res, Best_act
