import numpy as np
from workload import mus, sigmas


def remove_none(point_set, all_points_scores, **kwargs):
    return point_set, all_points_scores

def remove_oldest_point(point_set, all_points_scores, **kwargs):
    return point_set[1:], all_points_scores[1:]



class RemoveOldestNonPillarClass:
    def __init__(self, pillar_holder):
        self.pillar_holder = pillar_holder
        
    def remove_oldest_non_pillar(self, point_set, all_points_scores, num_pillar_points=3, removal_interval=4, current_iteration=0, **kwargs):
        """
        Removes the oldest non-pillar point from the dataset every 'removal_interval' iterations.

        Args:
        - point_set (np.array): The current set of points.
        - all_points_scores (np.array): Scores corresponding to each point in point_set.
        - num_pillar_points (int): Number of points to retain as pillar points.
        - removal_interval (int): The interval at which the oldest non-pillar point is removed.
        - current_iteration (int): The current iteration number in the optimization process.

        Returns:
        - tuple: Updated point_set and all_points_scores with the oldest non-pillar point removed.
        """
        
        pillar_holder = self.pillar_holder
        
        if not pillar_holder.indices_ready:
            #print("first iteration, indices not ready")
            indices = np.arange(len(point_set))[:3]
            non_indices = np.arange(len(point_set))[3:]
            
        else:
            indices = pillar_holder.indices
            non_indices = pillar_holder.non_indices
            
        pillar_points = point_set[indices]
        non_pillar_points = point_set[non_indices]
        pillar_scores = all_points_scores[indices]
        non_pillar_scores = all_points_scores[non_indices]
        

        # Check if it's time to remove the oldest non-pillar point
        if current_iteration % removal_interval == 0 and len(non_pillar_points) > 0:
            #print("current_iteration is:...", current_iteration)
            # Remove the oldest non-pillar point
            non_pillar_points = non_pillar_points[1:]
            non_pillar_scores = non_pillar_scores[1:]

        new_point_set = np.concatenate([pillar_points, non_pillar_points])
        new_scores = np.concatenate([pillar_scores, non_pillar_scores])
        
        return new_point_set, new_scores
    
class PillarHolder:
    def __init__(self, num_pillar_points):
        self.num_pillar_points = num_pillar_points
        self.indices_ready = False

    def resample_pillar_indices(self, point_set):
        # Sort the points to ensure they are in order
        sorted_indices = np.argsort(point_set)
        sorted_point_set = point_set[sorted_indices]

        # Divide the sorted points into equal intervals
        interval_size = len(sorted_point_set) // self.num_pillar_points
        indices = []

        for i in range(self.num_pillar_points):
            # Select a point from each interval
            index_in_interval = i * interval_size
            if index_in_interval < len(sorted_point_set):
                indices.append(sorted_indices[index_in_interval])

        non_indices = [i for i in range(len(point_set)) if i not in indices]

        self.indices = np.array(indices)
        self.non_indices = np.array(non_indices)
        self.indices_ready = True
        return self.indices, self.non_indices
