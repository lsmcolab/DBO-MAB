import numpy as np
from scipy.stats import norm

                   
z_score_95 = norm.ppf(1 - (1 - 0.95) / 2)


num_actions_list = [20, 40, 60, 80, 100, 120, 140,160]


n_env = 5

# Store environments to be consistent across methods
environments = {}

for num_actions in num_actions_list:
    

    mus_group1 = np.random.uniform(low=0.2, high=0.3, size=(n_env, num_actions // 3))
    mus_group2 = np.random.uniform(low=0.4, high=0.5, size=(n_env, num_actions // 3))
    mus_group3 = np.random.uniform(low=0.6, high=0.7, size=(n_env, num_actions - (2 * num_actions // 3)))
    mus = np.hstack((mus_group1, mus_group2, mus_group3))

    sigmas_group1 = np.random.uniform(low=0.1, high=0.15, size=(n_env, num_actions // 3))
    sigmas_group2 = np.random.uniform(low=0.15, high=0.2, size=(n_env, num_actions // 3))
    sigmas_group3 = np.random.uniform(low=0.2, high=0.25, size=(n_env, num_actions - (2 * num_actions // 3)))
    sigmas = np.hstack((sigmas_group1, sigmas_group2, sigmas_group3))

    # Introducing a dominant action
    
    for env in range(n_env):
        dominant_action = np.random.randint(num_actions)
        #dominant_action = np.random.randint(len(mus))
        mus[env,dominant_action] = 0.1 
        sigmas[env,dominant_action] = 0.25 

        
        environments[num_actions] = [(mus, sigmas)]

