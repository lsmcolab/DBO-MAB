from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import numpy as np
import matplotlib.pyplot as plt
import os



class BayesianOptimization:
    def __init__(self, noise_level=0.05, auto_plot=False, save_plots=False, plot_dir='plots_up44'):
        self.noise_level = noise_level
        self.model = None
        self.auto_plot = auto_plot
        self.save_plots = save_plots
        self.plot_dir = plot_dir
        self.plot_counter = 0
        self.next_sample_point = None
        self.acquisition_function_values = None
        
        self.iteration_count = 0  # Initialize the iteration count
        

    def fit_model(self, point_set, point_scores):
        m52 = ConstantKernel(1.1) * Matern(length_scale=0.5, nu=0.9)
        self.model = GaussianProcessRegressor(kernel=m52, alpha=self.noise_level**2)
        self.model.fit(point_set.reshape(-1, 1), point_scores)
        # print("data the model is being fit on:")
        # print("point set")
        # print(point_set)
        # print("point scores")
        # print(point_scores)
        self.x_observed = point_set
        self.y_observed = point_scores
        
    def select_points_GP(self, point_set, point_scores,xi=0.01,nu=0.5, delta=0.05, **kwargs):
        self.fit_model(point_set, point_scores)

        #self.c_min = kwargs.get("c_min", 0)  # Default value if not provided
        #self.c_max = kwargs.get("c_max", 1)  # Default value if not provided
        self.iteration_count += 1  # Increment the iteration count
        self.c_min = 0.0
        self.c_max = 0.09
        # Generate new points for prediction
        #new_points = np.linspace(self.c_min, self.c_max, 500)
        #new_points = np.random.beta( self.c_min, self.c_max, size=500)
        new_points = np.random.uniform( self.c_min, self.c_max, size=500)
        original_points_predictions, _ = self.model.predict(point_set.reshape(-1, 1), return_std=True)
        new_points_predictions, std = self.model.predict(new_points.reshape(-1, 1), return_std=True)
        
        # Check if point_set is one-dimensional and adjust accordingly
        # if point_set.ndim == 1:
        #     d = 1  # One-dimensional problem
        #     point_set = point_set.reshape(-1, 1)  # Reshape for compatibility with model
        # else:
        #     d = point_set.shape[1]  # Dimensionality of the problem
        # t = self.iteration_count  # Use the updated iteration count
        # kappa = np.sqrt(2 * nu * np.log(t * (d / 2.0) + 2 * np.pi**2 / (3 * delta)))

        # Calculate Lower Confidence Bound (LCB)
        kappa = 3.5  # Exploration-exploitation parameter
        lcb = new_points_predictions - kappa * std
        self.acquisition_function_values = lcb

       # Find the index of the next sample point by minimizing the acquisition function
        ind_next_sample = np.argmin(lcb)
        self.next_sample_point = new_points[ind_next_sample]

        new_point = new_points[ind_next_sample]
        new_point_score = new_points_predictions[ind_next_sample]

        if self.auto_plot:
            self.plot(point_set, point_scores, np.array([new_point]), np.array([new_point_score]))

        return np.array([new_point])

    def plot(self, X, y, X_new=None, y_new=None):
        if self.model is None:
            raise ValueError("The model has not been fitted yet!")

        X_all = np.linspace(self.c_min, self.c_max, 500).reshape(-1, 1)
        y_pred, sigma = self.model.predict(X_all, return_std=True)

        #plt.figure(figsize=(10, 5))
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 2, 1)
        plt.plot(X, y, 'r.', markersize=12, label='Observed Data')
        #plt.plot(self.x_observed, self.y_observed, 'g.', markersize=12, label='Observed Data 2')
        plt.plot(X_all, y_pred, 'b-', label='GP Prediction')
        plt.fill_between(X_all.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.1)

        # Add the vertical line for the next sample point
        if self.next_sample_point is not None:
            plt.axvline(x=self.next_sample_point, color='purple', linestyle='--', label='Next Sample Location')

        if X_new is not None and y_new is not None:
            plt.plot(X_new, y_new, 'gX', markersize=10, label='Next New Data Location')

        plt.xlabel('C_values')
        plt.ylabel('Mean')
        plt.title('Gaussian Process Regression')
        plt.xlim([min(X)-0.001, max(X)+0.001])
        plt.ylim([-0.3, 0.8])
        plt.legend()
        #plt.grid(True)
        
        # Plot Acquisition Function
        plt.subplot(1, 2, 2)
        if self.acquisition_function_values is not None:
            plt.plot(X_all, self.acquisition_function_values, 'k-', label='Acquisition Function')
            if self.next_sample_point is not None:
                plt.axvline(x=self.next_sample_point, color='purple', linestyle='--', label='Next Sample Location')
            plt.title('Acquisition Function (LCB)')
            plt.xlabel('C_values')
            plt.ylabel('LCB Value')
            #plt.xlim([min(X)-0.001, max(X)+0.001])
            #plt.ylim([min(X)-0.001, max(X)+0.001])
            plt.legend()


        plt.tight_layout()
        #plt.show()

        if self.save_plots:
            if not os.path.exists(self.plot_dir):
                os.makedirs(self.plot_dir)
            plot_filename = f"{self.plot_dir}/iteration_{self.plot_counter}.pdf"
            plt.savefig(plot_filename)
            print(f"Plot saved to {plot_filename}")
            self.plot_counter += 1

        plt.show()
        
        
#################################

class GPWithPillarReRun2:
    def __init__(self, pillar_holder, x_iteration=5, noise_level=0.05, num_pillar_points=3,inc=1.001):
        self.x_iteration = x_iteration
        self.noise_level = noise_level
        self.num_pillar_points = num_pillar_points
        self.current_iteration = 0
        self.pillar_point_index = 0
        self.pillar_points = None
        self.pillar_points_values = None
        self.model = None
        self.acquisition_function_values = None
        self.next_sample_point = None
        
        self.pillar_holder = pillar_holder
        
        self.c_min = 0.0
        self.c_max = 0.09
        self.inc = inc
            
    def fit_model(self, point_set, point_scores):
        # Define the kernel for the Gaussian Process
        m52 = ConstantKernel(1.1) * Matern(length_scale=0.5, nu=0.9)
        #m52 = RBF(length_scale=1.0) * RBF(length_scale=1.0)
        self.model = GaussianProcessRegressor(kernel=m52, alpha=self.noise_level ** 2)
        self.model.fit(point_set.reshape(-1, 1), point_scores)

    # def predict_pillar_point_values(self):
    #     """
    #     Predicts the values for the current set of pillar points using the Gaussian Process model.
    #     """
    #     if self.pillar_points is not None and self.model is not None:
    #         # Predict the values for the pillar points
    #         self.pillar_points_values, _ = self.model.predict(self.pillar_points.reshape(-1, 1), return_std=True)

    def select_points_GP(self, point_set, point_scores, **kwargs):
        self.fit_model(point_set, point_scores)
        
        #self.c_max *=self.inc
        
        if self.current_iteration % self.x_iteration == 0:
            if self.pillar_points is None:
                self.pillar_indices, self.non_pillar_indices = self.pillar_holder.resample_pillar_indices(point_set)
                self.pillar_points = point_set[self.pillar_indices]
                self.pillar_points_values = point_scores[self.pillar_indices]

            new_point = self.pillar_points[self.pillar_point_index]
            self.pillar_point_index = (self.pillar_point_index + 1) % self.num_pillar_points
        else:
            new_points = np.random.uniform(self.c_min, self.c_max, 500).reshape(-1, 1)
            original_points_predictions, _ = self.model.predict(point_set.reshape(-1, 1), return_std=True)
            new_points_predictions, std = self.model.predict(new_points, return_std=True)

            # Calculate Lower Confidence Bound (LCB)
            kappa = 3.5  # Exploration-exploitation parameter
           
            lcb = new_points_predictions - kappa * std
            self.acquisition_function_values = lcb

            ind_next_sample = np.argmin(lcb)
            self.next_sample_point = new_points[ind_next_sample]

            new_point = new_points[ind_next_sample]

        # Predict the values of the pillar points
        #self.predict_pillar_point_values()

        new_point_score, _ = self.model.predict(new_point.reshape(-1, 1), return_std=True)
        
        if self.pillar_points is None:
            self.pillar_points = point_set[self.pillar_indices]
            self.pillar_points_values = point_scores[self.pillar_indices]
        elif new_point in self.pillar_points:
            index = np.where(self.pillar_points == new_point)[0][0]
            self.pillar_points_values[index] = new_point_score
        
        self.current_iteration += 1
        return new_point.reshape(1,)
    
    ####################################
    
class GPWithPillarReRun:
    def __init__(self, pillar_holder, x_iteration=5, noise_level=0.05, num_pillar_points=3):
        self.x_iteration = x_iteration
        self.noise_level = noise_level
        self.num_pillar_points = num_pillar_points
        self.current_iteration = 0
        self.pillar_point_index = 0
        self.pillar_points = None
        self.pillar_points_values = None
        self.model = None
        self.acquisition_function_values = None
        self.next_sample_point = None
        
        self.pillar_holder = pillar_holder
        self.c_min = 0.0
        self.c_max = 0.09
        
        self.iteration_count = 0  # Initialize the iteration count
        self.expansion_factor_decay = 0.9
        
        
        ## new
        self.initial_expansion_factor = 0.4
        self.final_expansion_factor = 0.2
        
        
    def fit_model(self, point_set, point_scores):
        m52 = ConstantKernel(1.1) * Matern(length_scale=0.5, nu=0.9)
        self.model = GaussianProcessRegressor(kernel=m52, alpha=self.noise_level ** 2)
        self.model.fit(point_set.reshape(-1, 1), point_scores)

    # def predict_pillar_point_values(self):
    #     if self.pillar_points is not None and self.model is not None:
    #         self.pillar_points_values, _ = self.model.predict(self.pillar_points.reshape(-1, 1), return_std=True)

    def dynamic_range_adjustment(self, point_set, point_scores):
        # Calculate the first and third quartiles
        q1 = np.percentile(point_scores, 25)
        q3 = np.percentile(point_scores, 75)

        # Select points within the interquartile range
        iqr_points = point_set[(point_scores >= q1) & (point_scores <= q3)]

        if len(iqr_points) > 0:
            # Compute the IQM (mean of the points within the interquartile range)
            iqm = np.mean(iqr_points)

            # Adjust the range based on IQM
            expansion_factor1 = 0.1
            expansion_factor2 = 0.1
            expansion_factor = self.initial_expansion_factor * (self.expansion_factor_decay ** self.iteration_count) + self.final_expansion_factor * (1 - self.expansion_factor_decay ** self.iteration_count)
            #new_c_min = max(0, iqm - expansion_factor * np.ptp(iqr_points))
            #new_c_max = iqm + expansion_factor * np.ptp(iqr_points)
            
            
            new_c_min = max(0, iqm - expansion_factor1)
            new_c_max = iqm + expansion_factor2 * np.ptp(iqr_points)
            #print(" new_c_min=", new_c_min)
            #print(" new_c_max=", new_c_max)
        
        else:
            # If no points are found within the interquartile range, default to the original range
            new_c_min, new_c_max = self.c_min, self.c_max

        return new_c_min, new_c_max

    def select_points_GP(self, point_set, point_scores,nu=0.5, delta=0.05, **kwargs):
        self.fit_model(point_set, point_scores)
        self.iteration_count += 1  # Increment the iteration count
        self.c_min, self.c_max = self.dynamic_range_adjustment(point_set, point_scores)
        
        if self.current_iteration % self.x_iteration == 0:
            if self.pillar_points is None:
                self.pillar_indices, self.non_pillar_indices = self.pillar_holder.resample_pillar_indices(point_set)
                self.pillar_points = point_set[self.pillar_indices]
                self.pillar_points_values = point_scores[self.pillar_indices]

            new_point = self.pillar_points[self.pillar_point_index]
            self.pillar_point_index = (self.pillar_point_index + 1) % self.num_pillar_points
        else:
            new_points = np.random.uniform(self.c_min, self.c_max, 500).reshape(-1, 1)
            original_points_predictions, _ = self.model.predict(point_set.reshape(-1, 1), return_std=True)
            new_points_predictions, std = self.model.predict(new_points, return_std=True)

            
            # Check if point_set is one-dimensional and adjust accordingly
            # if point_set.ndim == 1:
            #     d = 1  # One-dimensional problem
            #     point_set = point_set.reshape(-1, 1)  # Reshape for compatibility with model
            # else:
            #     d = point_set.shape[1]  # Dimensionality of the problem
            # t = self.iteration_count  # Use the updated iteration count
            # kappa = np.sqrt(2 * nu * np.log(t * (d / 2.0) + 2 * np.pi**2 / (3 * delta)))
            
            # Adaptive kappa for LCB
            #kappa = 3.5 * (self.expansion_factor_decay ** self.iteration_count) + 1.0 * (1 - self.expansion_factor_decay ** self.iteration_count)
            
            kappa = 3.5
            lcb = new_points_predictions - kappa * std
            self.acquisition_function_values = lcb

            ind_next_sample = np.argmin(lcb)
            self.next_sample_point = new_points[ind_next_sample]

            new_point = new_points[ind_next_sample]

        #self.predict_pillar_point_values()

        new_point_score, _ = self.model.predict(new_point.reshape(-1, 1), return_std=True)
        
        if self.pillar_points is None:
            self.pillar_points = point_set[self.pillar_indices]
            self.pillar_points_values = point_scores[self.pillar_indices]
        elif new_point in self.pillar_points:
            index = np.where(self.pillar_points == new_point)[0][0]
            self.pillar_points_values[index] = new_point_score
        
        self.current_iteration += 1
        return new_point.reshape(1,)


    
