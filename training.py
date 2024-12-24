import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Create models directory
MODEL_DIR = 'student_performance_models'
os.makedirs(MODEL_DIR, exist_ok=True)

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the student performance dataset
    """
    data = pd.read_csv(file_path)

    # Encode categorical variables
    categorical_columns = ['Gender', 'Residence', 'Disability']
    encoders = {}

    for col in categorical_columns:
        le = LabelEncoder()
        data[f'{col}_Encoded'] = le.fit_transform(data[col])
        encoders[col] = le

    # Create performance grade labels with more nuanced categorization
    def categorize_grade(percentage):
        if percentage >= 90:
            return 'A+'
        elif percentage >= 80:
            return 'A'
        elif percentage >= 70:
            return 'B+'
        elif percentage >= 60:
            return 'B'
        elif percentage >= 50:
            return 'C'
        elif percentage >= 40:
            return 'D'
        else:
            return 'F'

    data['Performance_Grade'] = data['Performance_Grade_Percentage'].apply(categorize_grade)

    return data, encoders

def prepare_features_and_target(data):
    """
    Prepare features and target variables
    """
    features = [
        'Internal_1', 'Internal_2', 'Internal_3',
        'Assignment_Marks', 'Other_Activities',
        'Total_Internal_Marks', 'Total_Marks',
        'Gender_Encoded', 'Residence_Encoded',
        'Disability_Encoded', 'Attendance_Percentage',
        'Study_Hours_Per_Week', 'Part_Time_Job'
    ]

    X = data[features]
    y = data['Performance_Grade']

    return X, y

def create_ml_pipeline():
    """
    Create machine learning pipeline with regularization
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,  # Limit tree depth
            min_samples_split=5,  # Prevent overfitting
            min_samples_leaf=2,
            max_features='sqrt',  # Reduce model complexity
            random_state=42
        ))
    ])

    # Hyperparameter tuning
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [5, 10, None],
        'classifier__min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='accuracy'
    )

    return grid_search

def train_and_evaluate_model(X, y):
    """
    Train the model with cross-validation and evaluation
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create and train the model
    model = create_ml_pipeline()
    model.fit(X_train, y_train)

    # Best model parameters
    print("Best Parameters:", model.best_params_)

    # Predictions
    y_pred = model.predict(X_test)

    # Detailed evaluation
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    return model

def save_model_components(model, encoders):
    """
    Save trained model and encoders
    """
    joblib.dump(model.best_estimator_, os.path.join(MODEL_DIR, 'performance_model.joblib'))
    joblib.dump(encoders, os.path.join(MODEL_DIR, 'label_encoders.joblib'))
    print(f"Model saved in {MODEL_DIR}")

def main():
    # Load and preprocess data
    data, encoders = load_and_preprocess_data('student_performance_dataset.csv')

    # Prepare features and target
    X, y = prepare_features_and_target(data)

    # Train and evaluate model
    model = train_and_evaluate_model(X, y)

    # Save model components
    save_model_components(model, encoders)

if _name_ == "_main_":
    main()
