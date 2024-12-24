import joblib
import pandas as pd
import os
import numpy as np

# Load saved model and components
MODEL_DIR = 'student_performance_models'

def load_model_components():
    """Load saved model and encoders"""
    model = joblib.load(os.path.join(MODEL_DIR, 'performance_model.joblib'))
    encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.joblib'))
    return model, encoders

def prepare_student_data(student_info, encoders):
    """
    Prepare student data for prediction with robust encoding
    """
    # Create a copy of student info
    student_data = student_info.copy()

    # Robust categorical encoding with fallback
    def safe_encode(encoder, value):
        try:
            return encoder.transform([value])[0]
        except ValueError:
            # If value not in original encoder, use the first category
            return 0

    # Encode categorical variables with fallback
    student_data['Gender_Encoded'] = safe_encode(encoders['Gender'], student_info['Gender'])
    student_data['Residence_Encoded'] = safe_encode(encoders['Residence'], student_info['Residence'])
    student_data['Disability_Encoded'] = safe_encode(encoders['Disability'], student_info.get('Disability', 'None'))

    # Calculate total marks
    max_internal_marks = 20 * 3  # 3 internal exams, each out of 20
    max_assignment_marks = 10
    max_activity_marks = 5
    max_total_marks = max_internal_marks + max_assignment_marks + max_activity_marks

    total_internal_marks = student_info['Internal_1'] + student_info['Internal_2'] + student_info['Internal_3']
    total_marks = (
        total_internal_marks +
        student_info['Assignment_Marks'] +
        student_info['Other_Activities']
    )

    # Add calculated fields
    student_data['Total_Internal_Marks'] = total_internal_marks
    student_data['Total_Marks'] = total_marks
    student_data['Performance_Grade_Percentage'] = (total_marks / max_total_marks) * 100

    # Default Part-Time Job and Study Hours if not provided
    student_data['Part_Time_Job'] = student_info.get('Part_Time_Job', 0)
    student_data['Study_Hours_Per_Week'] = student_info.get('Study_Hours_Per_Week', 10)
    student_data['Attendance_Percentage'] = student_info.get('Attendance_Percentage', 80)

    # Select features in the order expected by the model
    features = [
        'Internal_1', 'Internal_2', 'Internal_3',
        'Assignment_Marks', 'Other_Activities',
        'Total_Internal_Marks', 'Total_Marks',
        'Gender_Encoded', 'Residence_Encoded',
        'Disability_Encoded', 'Attendance_Percentage',
        'Study_Hours_Per_Week', 'Part_Time_Job'
    ]

    return pd.DataFrame([student_data])[features]

def predict_student_performance(student_info):
    """
    Predict student performance with error handling
    """
    try:
        # Load model and encoders
        model, encoders = load_model_components()

        # Prepare student data
        student_features = prepare_student_data(student_info, encoders)

        # Predict
        prediction = model.predict(student_features)
        prediction_proba = model.predict_proba(student_features)

        return prediction[0], prediction_proba[0]
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None

# Example usage
def main():
    # Sample student information
    student_info = {
        'Internal_1': 10,  # First internal exam mark (out of 20)
        'Internal_2': 10,  # Second internal exam mark (out of 20)
        'Internal_3': 7,  # Third internal exam mark (out of 20)
        'Assignment_Marks': 1,  # Assignment marks (out of 10)
        'Other_Activities': 2,  # Other activities marks (out of 5)
        'Gender': 'Male',
        'Residence': 'Urban',
        'Disability': 'None',
        'Attendance_Percentage': 10,
        'Study_Hours_Per_Week': 10,
        'Part_Time_Job': 0
    }

    # Predict performance
    predicted_grade, grade_probabilities = predict_student_performance(student_info)

    if predicted_grade is not None:
        print(f"Predicted Grade: {predicted_grade}")
        print("Grade Probabilities:")
        model, _ = load_model_components()
        for grade, prob in zip(model.classes_, grade_probabilities):
            print(f"{grade}: {prob*100:.2f}%")

if _name_ == "_main_":
    main()
