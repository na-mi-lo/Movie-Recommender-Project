import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from tensorflow.keras.layers import BatchNormalization

def normalize(X):
    """
    This function adjusts all values to be in the range of 0 to 1 for each column.
    """         
    X = X - X.min() 
    normalizedX = X/X.max()
    return normalizedX

def load_data(num_classes):
    """
    Load and preprocess the dataset for training a neural network.
    Splits the data into training and testing sets.
    """
    file_path = 'IMDB_MovieListData_Normalized.csv'
    previous_data = pd.read_csv(file_path)

    # Assign classes based on the number of classes
    if num_classes == 2:
        previous_data.loc[previous_data['Revenue ( USD, Adjusted for 2024 Inflation)'] <= 50000000, 'Revenue Class'] = 0
        previous_data.loc[previous_data['Revenue ( USD, Adjusted for 2024 Inflation)'] > 50000000, 'Revenue Class'] = 1
    elif num_classes == 3:
        previous_data.loc[previous_data['Revenue ( USD, Adjusted for 2024 Inflation)'] <= 25000000, 'Revenue Class'] = 0
        previous_data.loc[(previous_data['Revenue ( USD, Adjusted for 2024 Inflation)'] > 25000000) & 
                          (previous_data['Revenue ( USD, Adjusted for 2024 Inflation)'] <= 120000000), 'Revenue Class'] = 1
        previous_data.loc[previous_data['Revenue ( USD, Adjusted for 2024 Inflation)'] > 120000000, 'Revenue Class'] = 2

    # Define numerical features
    numerical_features = [
        'Vote Average',
        'Vote Count',
        'Runtime (mins)',
        'Budget (USD, Adjusted for 2024 Inflation)',
        'Release Year',
        'Popularity'
    ]

    clean_data = previous_data[numerical_features + ['Revenue Class']]
    data = clean_data.dropna()
    print(data['Revenue Class'].value_counts())

    # Split and scale features and target
    X = data.drop(columns=['Revenue Class'])
    y = data['Revenue Class']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_one_hot = to_categorical(y, num_classes=num_classes)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_one_hot, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test

def train_neural_network(X_train, X_test, y_train, y_test, epochs, batch_size, num_classes):
    """
    Train a neural network to predict revenue and plot R^2 values over epochs.
    """
    model = Sequential([
        Input(shape=(X_train.shape[1],)),  # Define input shape
        Dense(512, activation='relu', kernel_regularizer=l2(0.02)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(256, activation='relu', kernel_regularizer=l2(0.02)),
        Dropout(0.2),
        Dense(128, activation='relu', kernel_regularizer=l2(0.02)),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = SGD(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Plot training and validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig("classification_accuracy_plot.png")

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    print("Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes))


def main():
    num_classes = 2  # Set to 2 for binary classification, 3 for multiclass classification

    # Load the data
    X_train, X_test, y_train, y_test = load_data(num_classes)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Train the neural network
    train_neural_network(X_train, X_test, y_train, y_test, epochs=300, batch_size=64, num_classes=num_classes)

if __name__ == "__main__":
    main()