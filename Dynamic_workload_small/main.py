from ucb1 import UCB1, UCB1_multiple_C, UCB1_
from bootstrapped_ucb import BootstrappedUCB
from thompson_sampling import ThompsonSamplingUnknownMeanVariance
from environments import Env, DynamicEnv, DynamicEnvOrdered
from simulation import do_run, do_run_
from methods import remove_none, remove_oldest_point, RemoveOldestNonPillarClass, PillarHolder
from gp import *
from metrics import metrics
#from initial_and_pb2 import InitialiseSample,InitialisePB2

import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
# from ray import tune
# from ray.tune.schedulers.pb2 import PB2
# from ray.tune.suggest import ConcurrencyLimiter
# from ray.tune.suggest.bayesopt import BayesOptSearch
# from hyperopt import hp
# from ray.tune.suggest.hyperopt import HyperOptSearch
import numpy as np
import pandas as pd
import scipy.stats as stats

from workload import *
from Dynamic_algo import SW_UCB, Discounted_UCB, ThompsonSamplingSW, ThompsonSamplingDS, ThompsonSampling_fDSW
from test_methods import  *
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def test_method_change_c(mus_all, sigmas_all, method, start_size=20, init_method=None, change_every=2000, verbose=True):
    
    
    n_steps = 10000
    
    if len(mus_all.shape) == 1:
        mus_all = np.array([mus_all])
        sigmas_all = np.array([sigmas_all])
        change_every = n_steps + 1
        #print("mus_all.shape: ", mus_all.shape)
    
    if not init_method:
        new_points = np.random.uniform(low=0.0, high=0.09, size=start_size)
        #new_points = np.linspace(0.0, 80.0, 20)
        
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
    select_action_class = UCB1_multiple_C(c_list = new_points, 
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
    Selected_actions = [results["selected_actions"]]
    
    Avg_res.append(response_times)
    Best_action.append(best_chosen)
        
    #print(results.keys())
    
    return np.array(Avg_res), np.array(Best_action),np.array(Selected_actions),select_action_class

####### Bandit
#select_action_class = SW_UCB(actions=len(mus_all))
def test_method_Bandit_dynamic(mus_all, sigmas_all, method_type, change_every=2000, n_steps=10000, verbose=True):
    Avg_res = []
    Best_action = []
    
    if len(mus_all.shape)==1:
        num_actions = mus_all.shape[0]
    else:
        num_actions = mus_all.shape[1]
    
    if method_type == 'SW_UCB':
        
        select_action_class = SW_UCB(actions=num_actions)
    elif method_type == 'Discounted_UCB':
        select_action_class = Discounted_UCB(actions=num_actions)
    elif method_type == 'ThompsonSamplingSW':
        select_action_class = ThompsonSamplingSW(actions=num_actions)
        
    elif method_type == 'ThompsonSamplingDS':
        select_action_class = ThompsonSamplingDS(actions=num_actions)
        
    elif method_type == 'ThompsonSampling_fDSW':
        select_action_class = ThompsonSampling_fDSW(actions=num_actions)
    else:
        raise ValueError(f"Unknown method type: {method_type}")

    #env = DynamicEnvOrdered(mus_all, sigmas_all, change_every=change_every) if len(mus_all.shape) > 1 else Env(mus_all, sigmas_all)
    
    if len(mus_all.shape)==1:
        env = Env(mus_all, sigmas_all)
    else:
        env = DynamicEnvOrdered(mus_all, sigmas_all, change_every=change_every)

    # The `do_run` function should execute the simulation and return the results
    results = do_run_(env, select_action_class, n_steps)

    response_times = results["average_response_time"]
    best_chosen = results["best_chosen"]
    Selected_actions = [results["selected_actions"]]
    
    
    Avg_res.append(response_times)
    Best_action.append(best_chosen)
        
    #print(results.keys())
    
    return np.array(Avg_res), np.array(Best_action),np.array(Selected_actions),select_action_class



def test_method_n_times_nor(n, **kwargs):
    #results = []
    response_times_means = []
    best_chosen_step_prob = []
    selected_actions_all_runs = []
    for i in range(n):
        # result, response_times_mean,best_chosen_step = test_method4(**kwargs)
        mean_response, best_s_action,selected_actions,s = test_method_Bandit_dynamic(**kwargs)
        response_times_means.append(mean_response)
        best_chosen_step_prob.append(best_s_action)
        selected_actions_all_runs.append(selected_actions)
        
    Avg_res = np.array(response_times_means)
    Best_act = np.array(best_chosen_step_prob)
    selected_actions_aggregated = np.array(selected_actions_all_runs)
    return Avg_res, Best_act,selected_actions_aggregated


def test_method_n_times(n, **kwargs):
    #results = []
    response_times_means = []
    best_chosen_step_prob = []
    selected_actions_all_runs = []
    for i in range(n):
        # result, response_times_mean,best_chosen_step = test_method4(**kwargs)
        mean_response, best_s_action,selected_actions,s = test_method_change_c(**kwargs)
        response_times_means.append(mean_response)
        best_chosen_step_prob.append(best_s_action)
        selected_actions_all_runs.append(selected_actions)
        
    Avg_res = np.array(response_times_means)
    Best_act = np.array(best_chosen_step_prob)
    selected_actions_aggregated = np.array(selected_actions_all_runs)
    return Avg_res, Best_act,selected_actions_aggregated


def main():
    # Initialize methods
    Bays = BayesianOptimization(save_plots=False)
    pillar_holder = PillarHolder(num_pillar_points=5)
    gp_with_pillar_rerun = GPWithPillarReRun(pillar_holder)
    remove_oldest = RemoveOldestNonPillarClass(pillar_holder)
    gp_with_pillar_rerun2 = GPWithPillarReRun2(pillar_holder)

    methods = {
        'ucb_sl' : (gp_with_pillar_rerun2.select_points_GP, remove_oldest.remove_oldest_non_pillar),
        'add_1_remove_none': (Bays.select_points_GP, remove_none),
        'add_1_remove_oldest': (Bays.select_points_GP, remove_oldest_point),
        'add_1_keep_uni_dy': (gp_with_pillar_rerun.select_points_GP, remove_oldest.remove_oldest_non_pillar),
        'BO_UCB_non_dy': (gp_with_pillar_rerun2.select_points_GP, remove_oldest.remove_oldest_non_pillar),
        
        'Dis_ucb' : (gp_with_pillar_rerun2.select_points_GP, remove_oldest.remove_oldest_non_pillar)
        
    }

    num_actions_ordered = list(np.unique([key for key, _ in environments.items()]))
    num_actions_ordered.sort()

    # Directory for saving CSV files
    output_dir = 'my_data_d_s'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    for method_name, method in methods.items():
        print("method running:", method_name)
        mean_response_time_summary = []
        lower_bound_summary = []
        upper_bound_summary = []
        mean_best_action_chosen_summary = []
        lower_bound_best_action_summary = []
        upper_bound_best_action_summary = []

        for num_actions in num_actions_ordered:
            print("Num actions: ", num_actions)

            runs = environments[num_actions]
            means = []
            bests = []

            for mus, sigmas in runs:
                #print("mus=", mus)
                
                if method_name == 'ucb_sl':
                    mean_response_times, best_action_chosen= test_method_n_times_S(
                        10, mus_all=mus, sigmas_all=sigmas, method=method, verbose=False)
                    
                elif method_name == 'Dis_ucb':
                    mean_response_times, best_action_chosen= test_method_n_times_D(
                        10, mus_all=mus, sigmas_all=sigmas, method=method, verbose=False)
                    
                else:
                    mean_response_times, best_action_chosen, _= test_method_n_times(
                        10, mus_all=mus, sigmas_all=sigmas, method=method, verbose=False)
                means.append(mean_response_times)
                bests.append(best_action_chosen)

            mean_response_times = np.mean(np.array(means), 0)
            best_action_chosen = np.mean(np.array(bests), 0)

            mean_response_time_summary.append(mean_response_times.mean())
            std_error_summary = np.std(mean_response_times) / np.sqrt(len(mean_response_times))
            lower_bound_summary.append(mean_response_times.mean() - 1.96 * std_error_summary)
            upper_bound_summary.append(mean_response_times.mean() + 1.96 * std_error_summary)

            mean_best_action_chosen_summary.append(best_action_chosen.mean())
            std_error_best_action_summary = np.std(best_action_chosen) / np.sqrt(len(best_action_chosen))
            lower_bound_best_action_summary.append(best_action_chosen.mean() - 1.96 * std_error_best_action_summary)
            upper_bound_best_action_summary.append(best_action_chosen.mean() + 1.96 * std_error_best_action_summary)

        # Prepare data for CSV
        summary_data = pd.DataFrame({
            'num_actions': num_actions_ordered,
            'mean_response_time': mean_response_time_summary,
            'lower_bound_response_time': lower_bound_summary,
            'upper_bound_response_time': upper_bound_summary,
            'mean_best_action': mean_best_action_chosen_summary,
            'lower_bound_best_action': lower_bound_best_action_summary,
            'upper_bound_best_action': upper_bound_best_action_summary
        })

        # Save to CSV
        summary_data.to_csv(f'{output_dir}/{method_name}.csv', index=False)    
        
    bandit_algorithms = ['SW_UCB', 'Discounted_UCB', 'ThompsonSamplingSW', 'ThompsonSamplingDS', 'ThompsonSampling_fDSW']
    num_actions_ordered = sorted(np.unique([key for key, _ in environments.items()]))

    # Run tests for bandit algorithms
    for algorithm in bandit_algorithms:
        print(f"Running experiments for: {algorithm}")
        mean_response_time_summary = []
        lower_bound_summary = []
        upper_bound_summary = []
        mean_best_action_chosen_summary = []
        lower_bound_best_action_summary = []
        upper_bound_best_action_summary = []

        for num_actions in num_actions_ordered:
            print(f"Num actions: {num_actions}")
            runs = environments[num_actions]
            means = []
            bests = []

            for mus, sigmas in runs:
                #print("mus=", mus)
                mean_response_times, best_action_chosen, _ = test_method_n_times_nor(
                    n=10, mus_all=mus, sigmas_all=sigmas, method_type=algorithm, verbose=False
                )
                means.append(mean_response_times)
                bests.append(best_action_chosen)
                
            mean_response_times = np.mean(np.array(means), 0)
            best_action_chosen = np.mean(np.array(bests), 0)

            # Calculate statistics and append the results
            mean_response_time_summary.append(mean_response_times.mean())
            std_error_summary = np.std(mean_response_times) / np.sqrt(len(mean_response_times))
            lower_bound_summary.append(mean_response_times.mean() - 1.96 * std_error_summary)
            upper_bound_summary.append(mean_response_times.mean() + 1.96 * std_error_summary)

            mean_best_action_chosen_summary.append(best_action_chosen.mean())
            std_error_best_action_summary = np.std(best_action_chosen) / np.sqrt(len(best_action_chosen))
            lower_bound_best_action_summary.append(best_action_chosen.mean() - 1.96 * std_error_best_action_summary)
            upper_bound_best_action_summary.append(best_action_chosen.mean() + 1.96 * std_error_best_action_summary)

        # Prepare data for CSV
        summary_data = pd.DataFrame({
            'num_actions': num_actions_ordered,
            'mean_response_time': mean_response_time_summary,
            'lower_bound_response_time': lower_bound_summary,
            'upper_bound_response_time': upper_bound_summary,
            'mean_best_action': mean_best_action_chosen_summary,
            'lower_bound_best_action': lower_bound_best_action_summary,
            'upper_bound_best_action': upper_bound_best_action_summary
        })

        # Save to CSV
        summary_file = os.path.join(output_dir, f"{algorithm}.csv")
        summary_data.to_csv(summary_file, index=False)
        
        
        


if __name__ == "__main__":
    
    main()
        
            
            
        
            
            
            