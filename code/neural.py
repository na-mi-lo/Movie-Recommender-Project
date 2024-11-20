import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
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

def build_neural_network(input_dim):
    """
    Builds a simple feedforward neural network for regression.
    """
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    # model.add(Dropout(0.2))  # Dropout for regularization
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Output layer for regression

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


def train_and_evaluate_nn(train_test_sets, epochs=50, batch_size=32):
    """
    Trains a neural network on the training set, evaluates it on the test set,
    and outputs the R^2 score, residual plot, and performance metrics.
    """
    X_train, X_test = train_test_sets['X_train'], train_test_sets['X_test']
    y_train, y_test = train_test_sets['y_train'], train_test_sets['y_test']

    # Build the neural network
    model = build_neural_network(input_dim=X_train.shape[1])

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                        epochs=epochs, batch_size=batch_size, verbose=1)

    # Predict on test set
    y_test_pred = model.predict(X_test).flatten()

    # Calculate R^2 and other metrics
    test_r2 = r2_score(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    print(f"\nTest Set R^2 Score: {test_r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")

    # Residual plot for the test set
    test_residuals = y_test - y_test_pred
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test_pred, test_residuals, alpha=0.6, label='Residuals')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Neural Network Residuals')
    plt.xlabel('Predicted Revenue')
    plt.ylabel('Residuals')
    plt.legend()
    plt.tight_layout()

    # Save the residual plot
    plt.savefig("nn_residual_plot.png")
    print("Residual plot saved as 'nn_residual_plot.png'.")
    plt.show()

    # Plot training history
    plt.figure(figsize=(7, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Neural Network Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.tight_layout()

    # Save the training history plot
    plt.savefig("nn_training_history.png")
    print("Training history plot saved as 'nn_training_history.png'.")
    plt.show()

    return {
        "model": model,
        "test_R2": test_r2,
        "mse": mse
    }


def main():
    # Load the dataset
    train_test_sets = load_data()

    # Train and evaluate the neural network
    results = train_and_evaluate_nn(train_test_sets, epochs=100, batch_size=32)

    print(f"\nNeural Network Results:")
    print(f"Test R^2 Score: {results['test_R2']:.4f}")
    print(f"Mean Squared Error: {results['mse']:.4f}")


main()