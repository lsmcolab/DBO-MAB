def do_run(env, 
           select_action_class, 
           n_steps): 
    selected_actions, rewards, res, Nom_norm, cum_response_time, cum_cost, average_response_time, env_keys, best_chosen = [], [],[],[],[],[],[],[],[]
    
    total_response_time = 0
    for i in range(n_steps):
    
        action = select_action_class.select_action()
        best_action = env.best_action()
        best_action_chosen = action == best_action
        best_chosen.append(best_action_chosen)
        
        delay = env.sample(action)
        select_action_class.track_response_time(delay)
        reward = - delay
        
        total_response_time += delay
        
        selected_actions.append(action)
        rewards.append(reward)
        res.append(delay)
        #average_response_time.append(sum(res)/len(res))
        average_response_time.append(total_response_time / (i + 1))
        cum_response_time.append(sum(res))
        select_action_class.update(action, reward, delay)
        env.step()

    

    return {"rewards": rewards, 
            "cum_response_time": cum_response_time, 
            "average_response_time": average_response_time, 
            "res": res, 
            "best_chosen": best_chosen,
            "selected_actions": selected_actions,
            "env_keys": env_keys}
    
    
    
def do_run_(env, 
           select_action_class, 
           n_steps): 
    selected_actions, rewards, res, Nom_norm, cum_response_time, cum_cost, average_response_time, env_keys, best_chosen = [], [],[],[],[],[],[],[],[]
    
    total_response_time = 0
    for i in range(n_steps):
    
        action = select_action_class.select_action()
        #print("selected_action:", action)
        best_action = env.best_action()
        best_action_chosen = action == best_action
        best_chosen.append(best_action_chosen)
        
        delay = env.sample(action)
        
        reward = - delay
        # new
        total_response_time += delay
        
        selected_actions.append(action)
        rewards.append(reward)
        res.append(delay)
        #average_response_time.append(sum(res)/len(res))
        average_response_time.append(total_response_time / (i + 1))
        cum_response_time.append(sum(res))
        select_action_class.update(action, reward)
        env.step()

    

    return {"rewards": rewards, 
            "cum_response_time": cum_response_time, 
            "average_response_time": average_response_time, 
            "res": res, 
            "best_chosen": best_chosen,
            "selected_actions": selected_actions,
            "env_keys": env_keys}






            