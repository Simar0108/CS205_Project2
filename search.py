from math import sqrt
import numpy as np

def load_data(filename):
    """Load data from file. First column is class label (1 or 2), rest are features."""
    data = np.loadtxt(filename, dtype=float)
    data[:, 0] = data[:, 0].astype(int)
    return data

def nearest_neighbor(data, current_set_of_features, object_to_classify_idx):
    """Find nearest neighbor using specified features."""
    object_to_classify = data[object_to_classify_idx]
    min_distance = float('inf')
    nearest_neighbor_idx = -1
    
    for i in range(len(data)):
        if i != object_to_classify_idx:
            distance = sqrt(sum((data[i][j+1] - object_to_classify[j+1]) ** 2 
                              for j in current_set_of_features))
            if distance < min_distance:
                min_distance = distance
                nearest_neighbor_idx = i
    
    return nearest_neighbor_idx

def leave_one_out_cross_validation(data, current_set_of_features):
    """Perform leave-one-out cross validation."""
    correct_predictions = 0
    for i in range(len(data)):
        nearest_idx = nearest_neighbor(data, current_set_of_features, i)
        if data[nearest_idx][0] == data[i][0]:
            correct_predictions += 1
    return correct_predictions / len(data)

def forward_selection(data):
    """Implement forward selection."""
    num_features = data.shape[1] - 1
    current_set_of_features = []
    
    # Show default rate
    default_accuracy = leave_one_out_cross_validation(data, current_set_of_features)
    print(f"\nUsing no features (default rate), accuracy is {default_accuracy:.3f}")
    print("Beginning Forward Selection search...")
    
    best_overall_accuracy = default_accuracy
    best_feature_set = []
    
    for i in range(num_features):
        print(f"\nOn level {i+1} of the search tree")
        feature_to_add = -1
        best_so_far_accuracy = 0
        
        for k in range(num_features):
            if k not in current_set_of_features:
                print(f"--Considering adding feature {k+1}")
                accuracy = leave_one_out_cross_validation(data, current_set_of_features + [k])
                print(f"   Using feature(s) {[x+1 for x in current_set_of_features + [k]]} accuracy is {accuracy:.3f}")
                
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add = k
        
        if feature_to_add != -1:
            current_set_of_features.append(feature_to_add)
            print(f"\nOn level {i+1} I added feature {feature_to_add+1} to current set")
            
            if best_so_far_accuracy > best_overall_accuracy:
                best_overall_accuracy = best_so_far_accuracy
                best_feature_set = current_set_of_features.copy()
    
    return best_feature_set, best_overall_accuracy

def backward_elimination(data):
    """Implement backward elimination."""
    num_features = data.shape[1] - 1
    current_set_of_features = list(range(num_features))
    
    # Show baseline with all features
    baseline_accuracy = leave_one_out_cross_validation(data, current_set_of_features)
    print(f"\nUsing all {num_features} features, accuracy is {baseline_accuracy:.3f}")
    print("Beginning Backward Elimination search...")
    
    best_overall_accuracy = baseline_accuracy
    best_feature_set = current_set_of_features.copy()
    
    for i in range(num_features - 1):
        print(f"\nOn level {i+1} of the search tree")
        feature_to_remove = -1
        best_so_far_accuracy = 0
        
        for k in current_set_of_features:
            temp_set = current_set_of_features.copy()
            temp_set.remove(k)
            print(f"--Considering removing feature {k+1}")
            accuracy = leave_one_out_cross_validation(data, temp_set)
            print(f"   Using feature(s) {[x+1 for x in temp_set]} accuracy is {accuracy:.3f}")
            
            if accuracy > best_so_far_accuracy:
                best_so_far_accuracy = accuracy
                feature_to_remove = k
        
        if feature_to_remove != -1:
            current_set_of_features.remove(feature_to_remove)
            print(f"\nOn level {i+1} I removed feature {feature_to_remove+1} from current set")
            
            if best_so_far_accuracy > best_overall_accuracy:
                best_overall_accuracy = best_so_far_accuracy
                best_feature_set = current_set_of_features.copy()
    
    return best_feature_set, best_overall_accuracy

def main():
    print("Welcome to Simar's Feature Selection Algorithm.")
    filename = input("Type in the name of the file to test: ")
    
    # Load and normalize data
    data = load_data(filename)
    print(f"\nThis dataset has {data.shape[1]-1} features (not including the class attribute), with {data.shape[0]} instances.")
    print(f"Class labels present: {sorted(np.unique(data[:, 0]))}")
    
    # Normalize features
    for i in range(1, data.shape[1]):
        mean = np.mean(data[:, i])
        std = np.std(data[:, i])
        if std != 0:
            data[:, i] = (data[:, i] - mean) / std
    
    # Get algorithm choice
    while True:
        print("\nSelect the algorithm to use:")
        print("1) Forward Selection")
        print("2) Backward Elimination")
        print("3) Both algorithms")
        choice = input("Enter your choice (1-3): ")
        if choice in ['1', '2', '3']:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    # Run selected algorithm(s)
    if choice == '1' or choice == '3':
        print("\nRunning Forward Selection...")
        forward_features, forward_accuracy = forward_selection(data)
        print(f"\nForward Selection best feature set: {[x+1 for x in forward_features]}")
        print(f"Forward Selection best accuracy: {forward_accuracy:.3f}")
    
    if choice == '2' or choice == '3':
        print("\nRunning Backward Elimination...")
        backward_features, backward_accuracy = backward_elimination(data)
        print(f"\nBackward Elimination best feature set: {[x+1 for x in backward_features]}")
        print(f"Backward Elimination best accuracy: {backward_accuracy:.3f}")

if __name__ == "__main__":
    main()
    