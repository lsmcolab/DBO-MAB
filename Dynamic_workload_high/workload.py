import numpy as np
from scipy.stats import norm

                   
z_score_95 = norm.ppf(1 - (1 - 0.95) / 2)


num_actions_list = [20, 40, 60, 80, 100, 120, 140,160]

#num_actions_list = [num_actions for num_actions in [21, 40, 60, 80, 100,120,140,160] for _ in range(10)]
#num_actions_list = [num_actions for num_actions in [140,160] for _ in range(10)]
n_env = 5

# Store environments to be consistent across methods
environments = {}

for num_actions in num_actions_list:
    

    mus_group1 = np.random.uniform(low=95, high=96, size=(n_env, num_actions // 3))
    mus_group2 = np.random.uniform(low=96, high=97, size=(n_env, num_actions // 3))
    mus_group3 = np.random.uniform(low=97, high=98, size=(n_env, num_actions - (2 * num_actions // 3)))
    mus = np.hstack((mus_group1, mus_group2, mus_group3))

    # Increasing the variability among the actions to add more uncertainty
    sigmas_group1 = np.random.uniform(low=70, high=80, size=(n_env, num_actions // 3))
    sigmas_group2 = np.random.uniform(low=80, high=90, size=(n_env, num_actions // 3))
    sigmas_group3 = np.random.uniform(low=90, high=95, size=(n_env, num_actions - (2 * num_actions // 3)))
    sigmas = np.hstack((sigmas_group1, sigmas_group2, sigmas_group3))

    # Introducing a dominant action
    
    for env in range(n_env):
        dominant_action = np.random.randint(num_actions)
        #dominant_action = np.random.randint(len(mus))
        mus[env,dominant_action] = 60.0 
        sigmas[env,dominant_action] = 95.0 

        #if num_actions in environments:
            #environments[num_actions].append((mus, sigmas))
        #else:
        environments[num_actions] = [(mus, sigmas)]

