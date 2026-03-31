"""
This module implements temporal (sequence-based) models like LSTMs for fall detection,
activity classification, and MET value regression.
"""
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, r2_score, mean_absolute_error
import joblib
from imblearn.over_sampling import SMOTE

def prepare_sequence_data(test_subject_id):
    """
    Loads windowed signal data and prepares it for sequence modeling.
    Splits data into training and testing sets based on the test_subject_id.
    Scales the features and handles class imbalance for fall detection.
    """
    print(f"Preparing sequence data for test subject: {test_subject_id}")

    # Load pre-processed data
    data = np.load('results/artifacts/windows_signals.npz', allow_pickle=True)
    signals = data['signals']
    metadata_df = pd.DataFrame(data['metadata'], columns=['subject_id', 'activity_id', 'fall_label', 'met_value'])

    # Separate features (X) and targets (y)
    X = signals
    y_fall = metadata_df['fall_label'].values
    y_activity = metadata_df['activity_id'].values
    y_met = metadata_df['met_value'].values.astype(float)
    groups = metadata_df['subject_id'].values

    # Split data into train and test sets
    train_indices = np.where(groups != test_subject_id)[0]
    test_indices = np.where(groups == test_subject_id)[0]

    X_train, X_test = X[train_indices], X[test_indices]
    y_fall_train, y_fall_test = y_fall[train_indices], y_fall[test_indices]
    y_activity_train, y_activity_test = y_activity[train_indices], y_activity[test_indices]
    y_met_train, y_met_test = y_met[train_indices], y_met[test_indices]

    # Scale the signal data
    # Reshape to 2D for scaler, then back to 3D
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_train_scaled_reshaped = scaler.fit_transform(X_train_reshaped)
    X_train = X_train_scaled_reshaped.reshape(X_train.shape)

    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled_reshaped = scaler.transform(X_test_reshaped)
    X_test = X_test_scaled_reshaped.reshape(X_test.shape)
    
    # Save the scaler for this fold
    scaler_path = f'results/artifacts/temporal_scaler_subject_{test_subject_id}.joblib'
    joblib.dump(scaler, scaler_path)

    # Handle imbalance for fall detection using SMOTE
    # Reshape 3D data to 2D for SMOTE
    X_train_smote_reshaped = X_train.reshape(X_train.shape[0], -1)
    smote = SMOTE(random_state=42)
    try:
        X_train_resampled_flat, y_fall_train_resampled = smote.fit_resample(X_train_smote_reshaped, y_fall_train)
        # Reshape back to 3D
        X_fall_train = X_train_resampled_flat.reshape(X_train_resampled_flat.shape[0], X_train.shape[1], X_train.shape[2])
        y_fall_train = y_fall_train_resampled
        print(f"SMOTE applied for fall detection. Original size: {len(y_fall_train)}, Resampled size: {len(y_fall_train_resampled)}")
    except ValueError as e:
        print(f"Could not apply SMOTE for subject {test_subject_id}, not enough samples. Error: {e}")
        # Use original data if SMOTE fails
        X_fall_train = X_train
        y_fall_train = y_fall_train

    # Encode activity labels
    label_encoder = LabelEncoder()
    y_activity_train_encoded = label_encoder.fit_transform(y_activity_train)
    y_activity_test_encoded = label_encoder.transform(y_activity_test)
    
    # One-hot encode for categorical cross-entropy
    y_activity_train_cat = to_categorical(y_activity_train_encoded)
    y_activity_test_cat = to_categorical(y_activity_test_encoded, num_classes=len(label_encoder.classes_))

    # Save the label encoder
    le_path = f'results/artifacts/temporal_activity_encoder_subject_{test_subject_id}.joblib'
    joblib.dump(label_encoder, le_path)

    return {
        "fall": (X_fall_train, y_fall_train, X_test, y_fall_test),
        "activity": (X_train, y_activity_train_cat, X_test, y_activity_test_cat, label_encoder),
        "met": (X_train, y_met_train, X_test, y_met_test)
    }


def build_lstm_fall_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm_activity_model(input_shape, num_classes):
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
        Dropout(0.5),
        Bidirectional(LSTM(64)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_lstm_met_regressor(input_shape):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.4),
        Bidirectional(LSTM(32)),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def run_temporal_analysis():
    """
    Main function to run the temporal modeling pipeline.
    It iterates through each subject, holding one out for testing,
    and trains/evaluates LSTM models for all three tasks.
    """
    print("Starting temporal modeling analysis...")
    
    # Load metadata to get subject list
    data = np.load('results/artifacts/windows_signals.npz', allow_pickle=True)
    metadata_df = pd.DataFrame(data['metadata'], columns=['subject_id', 'activity_id', 'fall_label', 'met_value'])
    all_subjects = np.unique(metadata_df['subject_id'])

    all_metrics = []

    for test_subject in all_subjects:
        print(f"\n{'='*20} Processing Fold: Test Subject {test_subject} {'='*20}")
        
        prepared_data = prepare_sequence_data(test_subject)
        
        # --- Task 1: Fall Detection ---
        X_fall_train, y_fall_train, X_fall_test, y_fall_test = prepared_data['fall']
        fall_model = build_lstm_fall_model(input_shape=(X_fall_train.shape[1], X_fall_train.shape[2]))
        fall_model.fit(X_fall_train, y_fall_train, epochs=20, batch_size=64, validation_split=0.1, verbose=0)
        y_pred_fall_prob = fall_model.predict(X_fall_test).ravel()
        y_pred_fall = (y_pred_fall_prob > 0.5).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_fall_test, y_pred_fall).ravel()
        fall_metrics = {
            'model': 'lstm_fall_detector', 'subject': test_subject,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'roc_auc': roc_auc_score(y_fall_test, y_pred_fall_prob)
        }
        all_metrics.append(fall_metrics)
        print(f"Fall Detector Metrics: {fall_metrics}")

        # --- Task 2: Activity Classification ---
        X_act_train, y_act_train, X_act_test, y_act_test, le = prepared_data['activity']
        activity_model = build_lstm_activity_model(
            input_shape=(X_act_train.shape[1], X_act_train.shape[2]),
            num_classes=len(le.classes_)
        )
        activity_model.fit(X_act_train, y_act_train, epochs=30, batch_size=64, validation_split=0.1, verbose=0)
        y_pred_act_cat = activity_model.predict(X_act_test)
        y_pred_act = np.argmax(y_pred_act_cat, axis=1)
        y_true_act = np.argmax(y_act_test, axis=1)
        
        report = classification_report(y_true_act, y_pred_act, target_names=le.classes_, output_dict=True)
        act_metrics = {
            'model': 'lstm_activity_classifier', 'subject': test_subject,
            'accuracy': report['accuracy'],
            'macro_f1': report['macro avg']['f1-score']
        }
        all_metrics.append(act_metrics)
        print(f"Activity Classifier Metrics: {act_metrics}")

        # --- Task 3: MET Value Regression ---
        X_met_train, y_met_train, X_met_test, y_met_test = prepared_data['met']
        met_model = build_lstm_met_regressor(input_shape=(X_met_train.shape[1], X_met_train.shape[2]))
        met_model.fit(X_met_train, y_met_train, epochs=30, batch_size=64, validation_split=0.1, verbose=0)
        y_pred_met = met_model.predict(X_met_test).ravel()

        met_metrics = {
            'model': 'lstm_met_regressor', 'subject': test_subject,
            'r2_score': r2_score(y_met_test, y_pred_met),
            'mae': mean_absolute_error(y_met_test, y_pred_met)
        }
        all_metrics.append(met_metrics)
        print(f"MET Regressor Metrics: {met_metrics}")

    # Save all metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv('results/artifacts/temporal_metrics.csv', index=False)
    print("\nTemporal modeling analysis complete. Metrics saved to results/artifacts/temporal_metrics.csv")

if __name__ == '__main__':
    # Create necessary directories if they don't exist
    os.makedirs('results/artifacts', exist_ok=True)
    run_temporal_analysis()
