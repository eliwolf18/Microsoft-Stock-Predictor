#libraries
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Function to reshape data for LSTM input
def prepare_data_for_lstm(df, time_steps=60):
    """
    Reshapes the dataframe for LSTM input.

    Args:
        df (pd.DataFrame): Preprocessed stock data with 'Normalized_Close'.
        time_steps (int): Number of days to use as input for the model.

    Returns:
        np.array, np.array: X (features) and y (target).
    """
    data = df['Normalized_Close'].values
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    X, y = np.array(X), np.array(y)
    # Reshape X to (samples, time steps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

# Enhanced LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(units=1))  # Single predicted price
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Function to train the model
def train_lstm_model(model, X_train, y_train, epochs=20, batch_size=32):
    """
    Trains the LSTM model.

    Args:
        model (keras.Model): Compiled LSTM model.
        X_train (np.array): Training features.
        y_train (np.array): Training target.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        keras.callbacks.History: Training history object.
    """
    # Early stopping to avoid overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,  # Stops training if val_loss doesn't improve for 10 epochs
        restore_best_weights=True
    )

    history = model.fit(
    X_train, y_train,
    epochs=10,  # You can adjust epochs
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
    return history

# Function to calculate evaluation metrics new changes for metrics
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2    