import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_summary(directory, method_order, method_properties, output_folder, plot_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    mean_response_times = {}
    best_action_probabilities = {}
    num_actions = None

    for method in method_order:
        filename = f"{method}.csv"
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            summary_data = pd.read_csv(filepath)
            if num_actions is None:
                num_actions = summary_data['num_actions']
            mean_response_times[method] = (summary_data['mean_response_time'],
                                           summary_data['lower_bound_response_time'],
                                           summary_data['upper_bound_response_time'])
            best_action_probabilities[method] = (summary_data['mean_best_action'],
                                                 summary_data['lower_bound_best_action'],
                                                 summary_data['upper_bound_best_action'])

    plt.figure(figsize=(12, 6))
    for method in method_order:
        mean, lower, upper = mean_response_times.get(method, ([], [], []))
        color, line_marker = method_properties.get(method, ('black', '-'))
        plt.plot(num_actions, mean, line_marker, label=method, color=color)
        plt.fill_between(num_actions, lower, upper, color=color, alpha=0.1)
    plt.xlabel('Number of Actions')
    plt.ylabel('Mean Response Time')
    plt.title(f'Mean Response Time for {plot_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'combined_mean_response_time_{plot_name}.pdf'), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 6))
    for method in method_order:
        mean, lower, upper = best_action_probabilities.get(method, ([], [], []))
        color, line_marker = method_properties.get(method, ('black', '-'))
        plt.plot(num_actions, mean, line_marker, label=method, color=color)
        plt.fill_between(num_actions, lower, upper, color=color, alpha=0.1)
    plt.xlabel('Number of Actions')
    plt.ylabel('Best Action Chosen Probability')
    plt.title(f'Best Action Chosen Probability for {plot_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'combined_best_action_probability_{plot_name}.pdf'), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    directory = 'my_data_h_dyn'
    output_folder = 'plot_outputs2'
    
    # First group of methods
    method_order_group1 = [
        'add_1_remove_none', 'add_1_remove_oldest', 'add_1_keep_uni_dy', 'BO_UCB_non_dy'
    ]
    # Second group of methods
    method_order_group2 = [
        'ThompsonSampling_fDSW'
    ]
    
    method_order_group3 = [
        'SW_UCB','ucb_sl', 'Discounted_UCB','Dis_ucb', 'ThompsonSampling_fDSW'
    ]

    method_properties = {
        'add_1_remove_none': ('green', 's-'),
        'add_1_remove_oldest': ('red', '8-'),
        'add_1_keep_uni_dy': ('darkcyan', '*-.'),
        'BO_UCB_non_dy': ('purple', '*-'),
        'SW_UCB': ('#65d6cf', 'H-'),
        'ucb_sl': ('#65d6cf', 'H:'),
        'Discounted_UCB': ('#c29006', '>-'),
        'Dis_ucb': ('#c29006', '>:'),
        'ThompsonSampling_fDSW': ('black', 'o-')
    }

    plot_summary(directory, method_order_group1, method_properties, output_folder, 'Group1_Methods')
    plot_summary(directory, method_order_group2, method_properties, output_folder, 'Group2_Methods_Bandits')
    plot_summary(directory, method_order_group3, method_properties, output_folder, 'Group3_Methods_Bandits')
