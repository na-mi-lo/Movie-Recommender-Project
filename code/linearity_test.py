import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
'''
"""
This program 1) reads in IMDB_MovieListData on kaggle (link: https://www.kaggle.com/datasets/shubhamchandra235/imdb-and-tmdb-movie-metadata-big-dataset-1m), 
2) visually evaluates the linearity of relationships between selected-features and revenue by scatter plots,
3) calculates the R^2 values to show how well each feature explains the variance in revenue lienarly.

### Outputs:
- A grid of scatter plots with feature values vs revenue.
- Titles of scatter plots include R^2 value for each feature to quantify its linear relationship.
"""
'''
def load_data(numerical_features):
    '''
    this function loads the dataset, removes rows with N/A values, and return the dadaset for analysis
    '''
    file_path = 'IMDB_MovieListData_Normalized.csv'
    previous_data = pd.read_csv(file_path)
    clean_data = previous_data.dropna()

    y = clean_data['Revenue ( USD, Adjusted for 2024 Inflation)']
    X = clean_data[numerical_features]

    return {'data': clean_data, 'features': X, 'target': y}


def linearity_validation(data, numerical_features, target_feature):
    '''
    this function creates a scatter plot and calculates the R^2 value against the revenue for each numerical 
    feature to quantify the strength of the linear relationship.
    '''
    num_cols = int(np.ceil(np.sqrt(len(numerical_features)))) 
    num_rows = int(np.ceil(len(numerical_features) / num_cols))

    fig, axis = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    axis = axis.flatten()  
    
    for i, feature in enumerate(numerical_features):
        ax = axis[i]
        x = data[feature]
        y = data[target_feature]

        # Compute simple linear regression line of feature vs revenue
        slope = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x))**2)
        intercept = np.mean(y) - slope * np.mean(x)
        predict_y = slope * x + intercept

        # Calculate R^2 of feature vs revenue
        r_total = np.sum((y - np.mean(y))**2)
        r_residual = np.sum((y - predict_y)**2)
        r_squared = 1 - (r_residual / r_total)

        # Draw the plot
        ax.scatter(x, y, alpha=0.5, label="Actual Revenue")
        ax.set_xlabel(feature)
        ax.set_ylabel(target_feature)
        ax.set_title(f"{feature} (R^2={r_squared:.2f})")
        ax.legend()

    for j in range(i + 1, len(axis)):
        axis[j].axis("off")

    plt.tight_layout()
    plt.savefig("combined_feature_plots.png")
    print("Plots saved as combined_feature_plots.png")

def main():
    numerical_feature = [
    'Vote Average', 'Vote Count', 'Runtime (mins)', 
    'Budget (USD, Adjusted for 2024 Inflation)', 
    'Release Year', 'Popularity', 
    'Average Rating', 'IMDB Rating', ]
    dataset = load_data(numerical_feature)
    linearity_validation(data=dataset['data'], numerical_features=numerical_feature, 
    target_feature='Revenue ( USD, Adjusted for 2024 Inflation)')

if __name__ == "__main__":
    main()