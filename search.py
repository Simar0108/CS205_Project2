from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

def load_data(filename):
    """Load data from file. First column is class label (1 or 2), rest are features."""
    data = np.loadtxt(filename, dtype=float)
    data[:, 0] = data[:, 0].astype(int)
    return data

def nearest_neighbor(data, current_set_of_features, object_to_classify_idx):
    """Find nearest neighbor using only the specified features (vectorized)."""
    if not current_set_of_features:
        # If no features, return a random neighbor (or always the first, for consistency)
        return (object_to_classify_idx + 1) % len(data)
    features = np.array(current_set_of_features) + 1  # +1 to skip class label
    object_to_classify = data[object_to_classify_idx, features]
    others = np.delete(data, object_to_classify_idx, axis=0)
    others_features = others[:, features]
    dists = np.linalg.norm(others_features - object_to_classify, axis=1)
    min_idx = np.argmin(dists)
    # Adjust index because we removed one row
    if min_idx >= object_to_classify_idx:
        min_idx += 1
    return min_idx

def leave_one_out_cross_validation(data, current_set_of_features, cost_counter=None):
    """Perform leave-one-out cross validation and count cost if provided."""
    correct_predictions = 0
    for i in range(len(data)):
        if cost_counter is not None:
            cost_counter[0] += 1  # Increment for each NN search
        nearest_idx = nearest_neighbor(data, current_set_of_features, i)
        if data[nearest_idx][0] == data[i][0]:
            correct_predictions += 1
    return correct_predictions / len(data)

def plot_selection(history, filename='feature_selection.png', max_labels=6):
    plt.figure(figsize=(14, 6))
    accuracies = [a*100 for a in history['accuracies']]
    n = len(history['feature_sets'])

    # Decide which labels to show
    if n <= max_labels + 2:
        xtick_labels = [str(s) for s in history['feature_sets']]
    else:
        best_idx = int(np.argmax(accuracies))
        xtick_pos = [0, 1, best_idx, n-2, n-1]
        xtick_labels = []
        for i in range(n):
            if i in xtick_pos:
                xtick_labels.append(str(history['feature_sets'][i]))
            elif i == best_idx + 1 and n > max_labels + 2:
                xtick_labels.append('{omitted for space}')
            else:
                xtick_labels.append('')

    plt.bar(range(n), accuracies, color='gray')
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel('Step', fontsize=14)
    plt.ylim(0, 100)
    plt.xticks(range(n), xtick_labels, rotation=25, ha='right', fontsize=10)
    plt.title('Current Feature Set: Forward Selection' if 'forward' in filename else 'Current Feature Set: Backward Elimination', fontsize=15, pad=20)
    plt.tight_layout(rect=[0, 0.13, 1, 1])  # Add more bottom margin for labels
    plt.savefig(filename, dpi=300)
    plt.close()

def forward_selection(data):
    num_features = data.shape[1] - 1
    current_set_of_features = []
    default_accuracy = leave_one_out_cross_validation(data, current_set_of_features)
    print(f"\nUsing no features (default rate), accuracy is {default_accuracy:.3f}")
    print("Beginning Forward Selection search...")
    history = {
        'feature_sets': [set()],
        'accuracies': [default_accuracy]
    }
    best_overall_accuracy = default_accuracy
    best_feature_set = []
    cost_counter = [0]
    start_time = time.time()
    for i in range(num_features):
        print(f"\nOn level {i+1} of the search tree")
        feature_to_add = -1
        best_so_far_accuracy = 0
        for k in range(num_features):
            if k not in current_set_of_features:
                print(f"--Considering adding feature {k+1}")
                accuracy = leave_one_out_cross_validation(data, current_set_of_features + [k], cost_counter)
                current_features = [x+1 for x in current_set_of_features + [k]]
                print(f"   Using feature(s) {current_features} accuracy is {accuracy:.3f}")
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add = k
        if feature_to_add != -1:
            current_set_of_features.append(feature_to_add)
            print(f"\nOn level {i+1} I added feature {feature_to_add+1} to current set")
            history['feature_sets'].append(set([x+1 for x in current_set_of_features]))
            history['accuracies'].append(best_so_far_accuracy)
            if best_so_far_accuracy > best_overall_accuracy:
                best_overall_accuracy = best_so_far_accuracy
                best_feature_set = current_set_of_features.copy()
    elapsed = time.time() - start_time
    print(f"Total computational cost (NN searches): {cost_counter[0]}")
    print(f"Total time: {elapsed:.2f} seconds")
    history['cost'] = cost_counter[0]
    history['time'] = elapsed
    return best_feature_set, best_overall_accuracy, history

def backward_elimination(data):
    num_features = data.shape[1] - 1
    current_set_of_features = list(range(num_features))
    baseline_accuracy = leave_one_out_cross_validation(data, current_set_of_features)
    print(f"\nUsing all {num_features} features, accuracy is {baseline_accuracy:.3f}")
    print("Beginning Backward Elimination search...")
    history = {
        'feature_sets': [sorted([x+1 for x in current_set_of_features])],
        'accuracies': [baseline_accuracy]
    }
    best_overall_accuracy = baseline_accuracy
    best_feature_set = current_set_of_features.copy()
    cost_counter = [0]
    start_time = time.time()
    for i in range(num_features - 1):
        print(f"\nOn level {i+1} of the search tree")
        feature_to_remove = -1
        best_so_far_accuracy = 0
        for k in current_set_of_features:
            temp_set = current_set_of_features.copy()
            temp_set.remove(k)
            accuracy = leave_one_out_cross_validation(data, temp_set, cost_counter)
            if accuracy > best_so_far_accuracy:
                best_so_far_accuracy = accuracy
                feature_to_remove = k
        if feature_to_remove != -1:
            current_set_of_features.remove(feature_to_remove)
            history['feature_sets'].append(sorted([x+1 for x in current_set_of_features]))
            history['accuracies'].append(best_so_far_accuracy)
            if best_so_far_accuracy > best_overall_accuracy:
                best_overall_accuracy = best_so_far_accuracy
                best_feature_set = current_set_of_features.copy()
    elapsed = time.time() - start_time
    print(f"Total computational cost (NN searches): {cost_counter[0]}")
    print(f"Total time: {elapsed:.2f} seconds")
    history['cost'] = cost_counter[0]
    history['time'] = elapsed
    return best_feature_set, best_overall_accuracy, history

def main():
    print("Welcome to Simar's Feature Selection Algorithm.")
    filename = input("Type in the name of the file to test: ")
    data = load_data(filename)
    print(f"\nThis dataset has {data.shape[1]-1} features (not including the class attribute), with {data.shape[0]} instances.")
    print(f"Class labels present: {sorted(np.unique(data[:, 0]))}")
    for i in range(1, data.shape[1]):
        mean = np.mean(data[:, i])
        std = np.std(data[:, i])
        if std != 0:
            data[:, i] = (data[:, i] - mean) / std
    while True:
        print("\nSelect the algorithm to use:")
        print("1) Forward Selection")
        print("2) Backward Elimination")
        print("3) Both algorithms")
        choice = input("Enter your choice (1-3): ")
        if choice in ['1', '2', '3']:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    if choice == '1':
        print("\nRunning Forward Selection...")
        forward_features, forward_accuracy, history = forward_selection(data)
        print(f"\nForward Selection best feature set: {[x+1 for x in forward_features]}")
        print(f"Forward Selection best accuracy: {forward_accuracy:.3f}")
        plot_selection(history, filename='forward_selection.png')
        with open('forward_history.pkl', 'wb') as f:
            pickle.dump(history, f)
        print("Forward selection history saved as forward_history.pkl")
    elif choice == '2':
        print("\nRunning Backward Elimination...")
        backward_features, backward_accuracy, history = backward_elimination(data)
        print(f"\nBackward Elimination best feature set: {[x+1 for x in backward_features]}")
        print(f"Backward Elimination best accuracy: {backward_accuracy:.3f}")
        plot_selection(history, filename='backward_elimination.png')
        with open('backward_history.pkl', 'wb') as f:
            pickle.dump(history, f)
        print("Backward elimination history saved as backward_history.pkl")
    elif choice == '3':
        print("\nRunning Forward Selection...")
        forward_features, forward_accuracy, forward_history = forward_selection(data)
        print(f"\nForward Selection best feature set: {[x+1 for x in forward_features]}")
        print(f"Forward Selection best accuracy: {forward_accuracy:.3f}")
        plot_selection(forward_history, filename='forward_selection.png')
        with open('forward_history.pkl', 'wb') as f:
            pickle.dump(forward_history, f)
        print("Forward selection history saved as forward_history.pkl")
        print("\nRunning Backward Elimination...")
        backward_features, backward_accuracy, backward_history = backward_elimination(data)
        print(f"\nBackward Elimination best feature set: {[x+1 for x in backward_features]}")
        print(f"Backward Elimination best accuracy: {backward_accuracy:.3f}")
        plot_selection(backward_history, filename='backward_elimination.png')
        with open('backward_history.pkl', 'wb') as f:
            pickle.dump(backward_history, f)
        print("Backward elimination history saved as backward_history.pkl")

if __name__ == "__main__":
    main()
    