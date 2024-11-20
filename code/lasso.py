import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

'''
This file 1) reads in IMDB_MovieListData on kaggle (link: https://www.kaggle.com/datasets/shubhamchandra235/imdb-and-tmdb-movie-metadata-big-dataset-1m), 
2) trains machine learning models with features includind runtime, budget, etc., 
3) predicts the revenue of a movie  

The models includes:
1) Lasso (hyperparameter tuning: regularization strength (alpha))
2) Random Forest (hyperparameter tuning: number of trees, max depth)
3) Linear Regression (hyperparameter tuning: None)
3) Neural Network (hyperparameter tuning: number of hidden layers, number of neurons in each layer)

Note: Lasso, linear, and random forest models are used for feature selection. Nerual Network is used for revenue prediction.

'''
def load_data():
    '''
    The function loads the dataset, removes rows with N/A values, selects numerical and categorical columns for regression.
    Input: None
    Output: Train and test datasets for regression
    '''
    file_path = 'IMDB_MovieListData_Normalized.csv'
    previous_data = pd.read_csv(file_path)
    clean_data = previous_data.dropna()

    numerical_features = ['Vote Average', 'Vote Count', 'Runtime (mins)', 
                          'Budget (USD, Adjusted for 2024 Inflation)', 
                          'Release Year', 'Popularity', 
                          'Average Rating', 'IMDB Rating', 'Meta Score']
    categorical_features = ['Production Companies', 'Production Countries', 
                            'Spoken Languages', 'Genres List']
    y = clean_data['Revenue ( USD, Adjusted for 2024 Inflation)']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_features) 
        ]
    )

    X = preprocessor.fit_transform(clean_data).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    train_test_sets = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    return train_test_sets

def lasso_train(train_test_sets):
    """
    Train a Lasso regression model using the training set,
    tune hyperparameter alpha, and evaluate the final model on the test set.
    """
    X_train, X_test = train_test_sets['X_train'], train_test_sets['X_test']
    y_train, y_test = train_test_sets['y_train'], train_test_sets['y_test']

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    alpha_lst = np.logspace(-3, 1, 20)  
    parameter = {'alpha': alpha_lst}
    lasso = Lasso(max_iter=5000) 
    grid_search = GridSearchCV(lasso, parameter, scoring='r2', cv=5)
    grid_search.fit(X_train, y_train)
    alpha = grid_search.best_params_['alpha']
    print(f"Best alpha is: {alpha}")

    
    final_model = Lasso(alpha=alpha, max_iter=5000)
    final_model.fit(X_train, y_train)
    test_predictions = final_model.predict(X_test)
    test_r2 = r2_score(y_test, test_predictions)
    print(f"\nTest Set R^2 Score: {test_r2:.4f}")

    test_residuals = y_test - test_predictions
    plt.figure(figsize=(7, 6))
    plt.scatter(test_predictions, test_residuals, alpha=0.6, label='Residuals')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Lasso Regression Residuals')
    plt.xlabel('Predicted Revenue')
    plt.ylabel('Residuals')
    plt.legend() 
    plt.tight_layout()
    
    plt.savefig("lasso_residual_plot.png")
    print(f"Residual plot saved")

def main():
    train_test_sets = load_data()
    lasso_train(train_test_sets)

main()