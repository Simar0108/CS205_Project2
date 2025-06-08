import pickle
import matplotlib.pyplot as plt

def plot_selection(history, filename='feature_selection.png', max_labels=6):
    plt.figure(figsize=(14, 6))
    accuracies = [a*100 for a in history['accuracies']]
    n = len(history['feature_sets'])

    if n <= max_labels + 2:
        xtick_labels = [str(s) for s in history['feature_sets']]
    else:
        best_idx = int(max(range(len(accuracies)), key=lambda i: accuracies[i]))
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
    plt.title('Current Feature Set: Forward Selection', fontsize=15, pad=20)
    plt.tight_layout(rect=[0, 0.13, 1, 1])
    plt.savefig(filename, dpi=300)
    plt.close()

if __name__ == "__main__":
    # Change the filename below to your saved history file
    with open('large_forward_history.pkl', 'rb') as f:
        history = pickle.load(f)
    plot_selection(history, filename='large_forward_selection.png')
    print("Plot saved as large_forward_selection.png") 