import streamlit as st
import joblib
import pandas as pd
import os
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import numpy as np
import plotly.express as px


class StudentPerformanceDashboard:
    def __init__(self):
        self.MODEL_DIR = 'student_performance_models'
        self.setup_page()

    def setup_page(self):
        """Configure Streamlit page layout and styling"""
        st.set_page_config(
            page_title="Student Performance Intelligence Dashboard",
            page_icon="🎓",
            layout="wide"
        )
        st.markdown("""
        <style>
        .stApp {
            background-color: #f4f6f9;
            font-family: 'Arial', sans-serif;
        }
        </style>
        """, unsafe_allow_html=True)

    def create_fallback_encoders(self, df):
        """Create fallback encoders for categorical columns if pre-trained ones are unavailable"""
        encoders = {}
        categorical_columns = {
            'Gender': None,
            'Residence': None,
            'Disability': ['None', 'Physical', 'Learning']
        }
        for col, predefined_categories in categorical_columns.items():
            encoder = LabelEncoder()
            if predefined_categories:
                encoder.fit(predefined_categories)
            else:
                encoder.fit(df[col].fillna('Unknown'))
            encoders[col] = encoder
        return encoders

    def load_model_components(self, df):
        """Load the trained machine learning model and encoders"""
        try:
            model = joblib.load(os.path.join(self.MODEL_DIR, 'performance_model.joblib'))
            try:
                encoders = joblib.load(os.path.join(self.MODEL_DIR, 'label_encoders.joblib'))
            except:
                st.warning("Couldn't load pre-trained encoders. Creating fallback encoders.")
                encoders = self.create_fallback_encoders(df)
            return model, encoders
        except FileNotFoundError:
            st.error("Model or encoders not found. Please ensure training is complete.")
            st.stop()
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.stop()

    def prepare_student_data(self, student_info, encoders):
        """Prepare student data for prediction"""
        student_data = student_info.copy()

        # Handle missing values for numeric columns
        numeric_columns = [
            'Internal_1', 'Internal_2', 'Internal_3', 'Assignment_Marks',
            'Other_Activities', 'Attendance_Percentage', 'Study_Hours_Per_Week'
        ]
        for col in numeric_columns:
            student_data[col] = student_info.get(col, 0)  # Replace NaN with 0

        # Handle missing or invalid categorical values
        categorical_columns = ['Gender', 'Residence', 'Disability', 'Part_Time_Job']
        for col in categorical_columns:
            if col == 'Part_Time_Job':
                mapping = {'No': 0, 'Yes': 1}
                student_data[col] = mapping.get(student_info.get(col, '').lower(), 0)
            else:
                encoder = encoders[col]
                student_data[f'{col}_Encoded'] = encoder.transform(
                    [student_info.get(col, 'Unknown')]
                )[0]

        # Calculate derived features
        student_data['Total_Internal_Marks'] = (
            student_data['Internal_1'] + student_data['Internal_2'] + student_data['Internal_3']
        )
        student_data['Total_Marks'] = (
            student_data['Total_Internal_Marks'] + 
            student_data['Assignment_Marks'] + 
            student_data['Other_Activities']
        )

        # Select features expected by the model
        features = [
            'Internal_1', 'Internal_2', 'Internal_3', 'Assignment_Marks',
            'Other_Activities', 'Total_Internal_Marks', 'Total_Marks',
            'Gender_Encoded', 'Residence_Encoded', 'Disability_Encoded',
            'Attendance_Percentage', 'Study_Hours_Per_Week', 'Part_Time_Job'
        ]
        return pd.DataFrame([student_data])[features]


    def load_and_validate_student_data(self):
        """Load and validate uploaded student data"""
        uploaded_file = st.file_uploader("📁 Upload Student Database (CSV)", type=['csv'], help="Upload a CSV file containing student performance data")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, low_memory=False)
                required_columns = [
                    'Student_ID', 'Internal_1', 'Internal_2', 'Internal_3',
                    'Assignment_Marks', 'Other_Activities', 'Gender', 'Residence',
                    'Disability', 'Attendance_Percentage', 'Study_Hours_Per_Week',
                    'Part_Time_Job', 'Subject_Code'
                ]
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    st.error(f"Missing columns: {', '.join(missing_columns)}")
                    return None
                df['Student_ID'] = df['Student_ID'].astype(str)
                return df
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
        return None

    def create_performance_chart(self, df):
        """Create a chart comparing predicted grades across subjects"""
        fig = go.Figure()

        # Check if Predicted_Grade exists in the DataFrame
        if 'Predicted_Grade' in df.columns:
            # Group by 'Subject_Code' and aggregate grades into lists
            predictions = df.groupby('Subject_Code')['Predicted_Grade'].agg(list).reset_index()
            predictions['hover_text'] = predictions['Subject_Code'] + ': ' + predictions['Predicted_Grade'].astype(str)

            # Define the grade mapping to numeric values for standard deviation calculation
            def grade_to_numeric(grade):
                """Map grades to numeric values for standard deviation calculation"""
                grade_map = {
                    'A+': 4, 'A': 3.7, 'B+': 3, 'B': 2.7, 'C+': 2, 'C': 1.7,
                    'D': 1, 'F': 0
                }
                return grade_map.get(grade, -1)  # Default to -1 for unknown grades

            # Convert grades to numeric values for calculation
            predictions['Predicted_Grade_Numeric'] = predictions['Predicted_Grade'].apply(lambda x: [grade_to_numeric(grade) for grade in x])

            # Calculate the standard deviation for each subject's numeric grades
            predictions['Std_Dev'] = predictions['Predicted_Grade_Numeric'].apply(lambda x: np.std(x) if len(x) > 1 else 0)

            # Plot the bar chart
            fig.add_trace(go.Bar(
                x=predictions['Subject_Code'],
                y=[np.mean(grade_list) for grade_list in predictions['Predicted_Grade_Numeric']],  # Use mean for y-axis
                error_y=dict(type='data', array=predictions['Std_Dev']),
                marker_color=px.colors.qualitative.Set1,
                hoverinfo='text',
                hovertext=predictions['hover_text']
            ))

            # Customize the layout
            fig.update_layout(
                title="Predicted Student Grades Across Subjects",
                xaxis_title="Subjects",
                yaxis_title="Predicted Grade (Numeric)",
                barmode='group'
            )
        else:
            st.warning("No predicted grades available. Please run the prediction model first.")

        return fig




    def main(self):
        """Main application logic"""
        st.title("🎓 Student Performance Intelligence Dashboard")
        df = self.load_and_validate_student_data()
        if df is not None:
            st.sidebar.header("🔍 HOD Student Search")
            search_query = st.sidebar.text_input("Enter Student ID", help="Search for a student by their unique ID")
            if search_query:
                student_records = df[df['Student_ID'] == search_query]
                if not student_records.empty:
                    st.header(f"Performance Analysis for Student ID: {search_query}")
                    model, encoders = self.load_model_components(df)

                    # Process each subject for predictions
                    predictions = []
                    for _, row in student_records.iterrows():
                        student_data = row.to_dict()
                        subject = student_data.pop('Subject_Code')  # Exclude Subject_Code from model input
                        prepared_data = self.prepare_student_data(student_data, encoders)
                        predicted_grade = model.predict(prepared_data)[0]
                        predictions.append({'Subject_Code': subject, 'Predicted_Grade': predicted_grade})
                    
                    prediction_df = pd.DataFrame(predictions)
                    st.dataframe(prediction_df)

                    # Create performance chart
                    if not prediction_df.empty:
                        performance_chart = self.create_performance_chart(prediction_df)
                        st.plotly_chart(performance_chart)
                    else:
                        st.warning("No predictions available for this student.")
                else:
                    st.warning("No records found for the given Student ID.")
            else:
                st.info("Enter a Student ID in the search box to begin.")

if __name__ == "__main__":
    dashboard = StudentPerformanceDashboard()
    dashboard.main()
