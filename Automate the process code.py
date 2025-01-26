import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Define the data ingestion and preprocessing steps
def preprocess_data(data_path):
    """
    Reads data from the specified path, performs basic cleaning, and splits into train/test sets.

    Args:
        data_path: Path to the data file.

    Returns:
        X_train, X_test, y_train, y_test, preprocessor: Preprocessed data for training/testing.
    """
    data = pd.read_csv(data_path)

    # Handle missing values
    data.fillna(data.mean(), inplace=True)  # Replace NaN in numerical columns with the mean
    data['categorical_feature'].fillna(data['categorical_feature'].mode()[0], inplace=True)  # Replace NaN in categorical feature

    # Select features and target variable
    X = data[['feature1', 'feature2', 'feature3', 'categorical_feature']]  # Replace with actual feature names
    y = data['target_variable']

    # One-hot encode categorical features and standardize numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['feature1', 'feature2', 'feature3']),
            ('cat', OneHotEncoder(), ['categorical_feature'])
        ])

    X = preprocessor.fit_transform(X)

    # Convert target variable to one-hot encoding if it's categorical
    if y.nunique() > 2:  # Multi-class classification
        y = to_categorical(y)
    else:
        y = y.values  # Binary classification (no need for one-hot encoding)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessor

# Define the deep learning model
def build_model(input_dim, output_dim):
    """
    Builds a simple feedforward neural network model.

    Args:
        input_dim: Number of input features.
        output_dim: Number of output classes.

    Returns:
        model: Compiled Keras model.
    """
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(output_dim, activation='softmax' if output_dim > 1 else 'sigmoid')  # Multi-class or binary classification
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy' if output_dim > 1 else 'binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Train and evaluate the model
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """
    Trains and evaluates the deep learning model.

    Args:
        X_train, X_test, y_train, y_test: Preprocessed data for training/testing.

    Returns:
        model: The trained model.
        history: Training history.
    """
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1

    model = build_model(input_dim, output_dim)

    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train, y_train, 
                        validation_data=(X_test, y_test),
                        epochs=100, 
                        batch_size=32, 
                        callbacks=[early_stopping],
                        verbose=1)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    return model, history

# Main execution
if _name_ == "_main_":
    data_path = "path/to/your/data.csv"  # Replace with actual data path
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(data_path)
    model, history = train_and_evaluate_model(X_train, X_test, y_train, y_test)

    # Save the trained model
    model.save("trained_model.h5")
    print("Model saved as 'trained_model.h5'")