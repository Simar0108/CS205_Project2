import matplotlib.pyplot as plt
from utils import load_data, normalize_features, get_dataset_info
from search import forward_selection

def plot_feature_selection_history(history):
    """Plot the accuracy vs features added during forward selection."""
    plt.figure(figsize=(12, 8))
    plt.plot(range(len(history['accuracies'])), history['accuracies'], 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Features', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Feature Selection History', fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis limits from 0 to 1.0
    plt.ylim(0, 1.0)
    
    # Add feature labels with alternating positions to avoid overlap
    for i, feature in enumerate(history['features_added']):
        # Alternate between top and bottom positions
        if i % 2 == 0:
            y_offset = 0.02  # Above the point
        else:
            y_offset = -0.05  # Below the point
            
        plt.annotate(feature, 
                    (i+1, history['accuracies'][i+1]),
                    xytext=(0, y_offset * 1.0),  # Scale offset by 1.0 instead of y_max-y_min
                    textcoords='offset points',
                    ha='center',
                    va='bottom' if i % 2 == 0 else 'top',
                    fontsize=10,
                    bbox=dict(facecolor='white', 
                             edgecolor='none',
                             alpha=0.7,
                             pad=2))
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.savefig('wine_feature_selection.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Wine dataset feature names (using shorter abbreviations)
    feature_names = [
        "Alcohol",
        "Malic",
        "Ash",
        "Alcalinity",
        "Mg",
        "Phenols",
        "Flavanoids",
        "Nonflavanoid",
        "Proantho",
        "Color",
        "Hue",
        "OD280/315",
        "Proline"
    ]
    
    # Load and process data
    data = load_data('wine/wine.data')
    data = normalize_features(data)
    info = get_dataset_info(data)
    
    print(f"\nWine Dataset Information:")
    print(f"Number of features: {info['num_features']}")
    print(f"Number of instances: {info['num_instances']}")
    
    # Run forward selection
    best_features, best_accuracy, history = forward_selection(data, feature_names)
    
    # Plot results
    plot_feature_selection_history(history)
    
    # Print final results
    print("\nFinal Results:")
    print(f"Best feature set: {[feature_names[i] for i in best_features]}")
    print(f"Best accuracy: {best_accuracy:.3f}")

if __name__ == "__main__":
    main()
