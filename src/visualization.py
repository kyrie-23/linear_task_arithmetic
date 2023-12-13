import torch
import numpy as np

def compute_predictions(model, theta_0, alpha, tau, x):
    # Compute model predictions with modified parameters
    modified_params = [p + alpha * t for p, t in zip(theta_0, tau)]
    model.load_state_dict({name: param for name, param in zip(model.state_dict().keys(), modified_params)})
    return model(x)

def prediction_error(y_pred, y_true):
    # Prediction error as a distance metric
    return (y_pred.argmax(dim=-1) != y_true.argmax(dim=-1)).float().mean()

def compute_disentanglement_error(model, theta_0, alpha1, alpha2, tau1, tau2, dataset1, dataset2):
    xi = 0
    for (x1, y1), (x2, y2) in zip(dataset1, dataset2):
        pred1 = compute_predictions(model, theta_0, alpha1, tau1, x1)
        combined_pred1 = compute_predictions(model, theta_0, alpha1, tau1, x1) + compute_predictions(model, theta_0, alpha2, tau2, x1)
        
        pred2 = compute_predictions(model, theta_0, alpha2, tau2, x2)
        combined_pred2 = compute_predictions(model, theta_0, alpha1, tau1, x2) + compute_predictions(model, theta_0, alpha2, tau2, x2)
        
        xi += prediction_error(combined_pred1, y1) + prediction_error(combined_pred2, y2)
    
    xi /= (len(dataset1) + len(dataset2))
    return xi

# Example Usage
# Define your model, tasks, and datasets
model = LinearizedModel(MyModel())
theta_0 = [param.data for param in model.parameters()]  # Initial parameters
alpha1, alpha2 = 0.5, 0.5  # Scaling factors for task vectors
tau1, tau2 = [task_vector1], [task_vector2]  # Task vectors for each task
dataset1, dataset2 = [data_for_task1], [data_for_task2]  # Datasets for each task

# Compute disentanglement error
disentanglement_error = compute_disentanglement_error(model, theta_0, alpha1, alpha2, tau1, tau2, dataset1, dataset2)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Example: Visualize Task Vectors Using PCA
def visualize_task_vectors(task_vectors):
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(task_vectors)
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1])
    for i, txt in enumerate(range(len(task_vectors))):
        plt.annotate(txt, (reduced_vectors[i, 0], reduced_vectors[i, 1]))
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Task Vector Visualization')
    plt.show()

# Example: Visualize Eigenfunction Localization Using Heatmap
def visualize_eigenfunction_localization(eigenfunctions, points):
    # Assuming eigenfunctions is a 2D array where each row corresponds to a point
    plt.imshow(eigenfunctions, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Eigenfunction Localization')
    plt.show()

# Usage Example
# task_vectors = Your task vectors
# visualize_task_vectors(task_vectors)

# eigenfunctions = Computed eigenfunctions for certain points
# points = Points or regions of interest
# visualize_eigenfunction_localization(eigenfunctions, points)
