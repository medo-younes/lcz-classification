

import numpy as np

def predictor_stack(features):
    """Build a predictor stack from a 3D array with shape (n_features, height, width)
    
    Args:
        features (np.array): 3D array with shape (n_features, height, width)
       
        
    Returns:
        np.array: Predictor stack with shape (height*width, n_features)
    """
    return np.stack(features).reshape(features.shape[0],-1).T

