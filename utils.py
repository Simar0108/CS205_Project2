import numpy as np

def load_data(filename):
    """Load data from file. First column is class label, rest are features."""
    data = np.loadtxt(filename, dtype=float, delimiter=',')
    data[:, 0] = data[:, 0].astype(int)
    return data

def normalize_features(data):
    """Normalize all features (columns 1 onwards)."""
    for i in range(1, data.shape[1]):
        mean = np.mean(data[:, i])
        std = np.std(data[:, i])
        if std != 0:
            data[:, i] = (data[:, i] - mean) / std
    return data

def get_dataset_info(data):
    """Get basic information about the dataset."""
    num_features = data.shape[1] - 1
    num_instances = data.shape[0]
    return {
        'num_features': num_features,
        'num_instances': num_instances
    } 