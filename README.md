# Feature Selection Algorithm

This project implements feature selection algorithms to analyze and identify the most important features in datasets. It includes both a general-purpose feature selection tool and a specific analysis of the Wine dataset.

## Project Structure

- `search.py`: Core implementation of feature selection algorithms
  - Forward Selection
  - Nearest Neighbor classification
  - Leave-one-out cross validation

- `utils.py`: Utility functions for data handling
  - Data loading
  - Feature normalization
  - Dataset information

- `wine.py`: Specific analysis of the Wine dataset
  - Feature selection analysis
  - Visualization of results
  - Wine-specific feature names and descriptions

## Dataset Format

The code expects datasets in the following format:
- First column: Class labels (integers)
- Remaining columns: Features (numeric values)
- Comma-separated values (CSV) format

## Usage

### For General Datasets
```bash
python3 search.py
```
This will:
1. Prompt for a dataset filename
2. Ask which algorithm to use (Forward Selection, Backward Elimination, or both)
3. Display the results of the feature selection process

### For Wine Dataset Analysis
```bash
python3 wine.py
```
This will:
1. Load and analyze the Wine dataset
2. Perform forward selection
3. Generate a plot showing feature selection history
4. Save the plot as 'wine_feature_selection.png'
5. Display the best feature set and accuracy

## Wine Dataset Features

The Wine dataset includes 13 features:
1. Alcohol
2. Malic
3. Ash
4. Alcalinity
5. Mg (Magnesium)
6. Phenols
7. Flavanoids
8. Nonflavanoid
9. Proantho
10. Color
11. Hue
12. OD280/315
13. Proline

## Output

The program generates:
1. Console output showing the feature selection process
2. For the Wine dataset: a plot showing how accuracy changes as features are added
3. Final results showing the best feature set and its accuracy

## Requirements

- Python 3.x
- NumPy
- Matplotlib

## Note

The Wine dataset should be placed in a 'wine' directory as 'wine.data' for the wine analysis script to work properly. 