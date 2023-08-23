import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def visualize_weight(feature):
    # Normalize the feature values to range [0, 1]
    normalized_feature = (feature - feature.min()) / (feature.max() - feature.min())
    
    # Convert the feature tensor to a numpy array
    normalized_feature_np = normalized_feature.detach().numpy()
    
    # Create a heatmap using the normalized feature values
    heatmap = plt.imshow(normalized_feature_np, cmap='hot', interpolation='nearest')
    plt.colorbar(heatmap)
    plt.show()
