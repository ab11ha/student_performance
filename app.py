import streamlit as st
import joblib
import pandas as pd
import os
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

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
        .recommendation-card {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 15px;
            margin-bottom: 10px;
            transition: transform 0.3s ease;
        }
        .recommendation-card:hover {
            transform: scale(1.02);
        }
        </style>
        """, unsafe_allow_html=True)

    def create_fallback_encoders(self, df):
        """Create fallback label encoders if model encoders are not available"""
        encoders = {}
        
        # Explicitly define the expected categories
        categorical_columns = {
            'Gender': None,
            'Residence': None,
            'Disability': ['None', 'Physical', 'Learning']
        }
        
        for col, predefined_categories in categorical_columns.items():
            encoder = LabelEncoder()
            
            # If predefined categories are provided, use them
            if predefined_categories:
                encoder.fit(predefined_categories)
            else:
                # Otherwise, fit on the column's unique values
                encoder.fit(df[col].fillna('Unknown'))
            
            encoders[col] = encoder
        
        return encoders

    def load_model_components(self, df):
        """Load saved machine learning model and encoders with fallback mechanism"""
        try:
            # Try to load pre-trained model and encoders
            model = joblib.load(os.path.join(self.MODEL_DIR, 'performance_model.joblib'))
            try:
                encoders = joblib.load(os.path.join(self.MODEL_DIR, 'label_encoders.joblib'))
            except:
                # If encoders can't be loaded, create fallback encoders
                st.warning("Couldn't load pre-trained encoders. Creating fallback encoders.")
                encoders = self.create_fallback_encoders(df)
            
            return model, encoders
        
        except FileNotFoundError:
            st.error("Model files not found. Please ensure model training is complete.")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error loading models: {str(e)}")
            st.stop()

    def prepare_student_data(self, student_info, encoders):
        """Prepare student data for model prediction with robust encoding"""
        def safe_encode(encoder, value, column_name):
            """Safely encode categorical values"""
            # Mapping for specific columns
            special_mappings = {
                'Disability': {
                    'None':'', 'Physical':'Physical', 'Learning':'Learning'
                }
            }
            
            # Apply special mapping if it exists for the column
            if column_name in special_mappings:
                value = special_mappings[column_name].get(value, 'None')
            
            try:
                # If value is not in encoder's classes, use the first (default) class
                if value not in encoder.classes_:
                    # st.warning(f"Unexpected value for {column_name}: {value}. Using default: {encoder.classes_[0]}")
                    value = encoder.classes_[0]
                
                return encoder.transform([value])[0]
            except Exception as e:
                st.error(f"Encoding error for {column_name}: {e}")
                return 0

        student_data = student_info.copy()
        
        # Robust encoding for categorical columns
        categorical_columns = ['Gender', 'Residence', 'Disability']
        for col in categorical_columns:
            student_data[f'{col}_Encoded'] = safe_encode(
                encoders[col], 
                student_info.get(col, 'None'),
                col
            )
        
        # Performance calculation
        max_internal_marks = 20 * 3
        max_assignment_marks = 10
        max_activity_marks = 5
        max_total_marks = max_internal_marks + max_assignment_marks + max_activity_marks
        
        total_internal_marks = sum([
            student_info.get('Internal_1', 0), 
            student_info.get('Internal_2', 0), 
            student_info.get('Internal_3', 0)
        ])
        total_marks = total_internal_marks + student_info.get('Assignment_Marks', 0) + student_info.get('Other_Activities', 0)
        
        student_data['Total_Internal_Marks'] = total_internal_marks
        student_data['Total_Marks'] = total_marks
        student_data['Performance_Grade_Percentage'] = (total_marks / max_total_marks) * 100
        
        # Consistent feature selection
        features = [
            'Internal_1', 'Internal_2', 'Internal_3', 
            'Assignment_Marks', 'Other_Activities', 
            'Total_Internal_Marks', 'Total_Marks',
            'Gender_Encoded', 'Residence_Encoded', 
            'Disability_Encoded', 'Attendance_Percentage', 
            'Study_Hours_Per_Week', 'Part_Time_Job'
        ]
        
        return pd.DataFrame([student_data])[features]

    def load_and_validate_student_data(self):
        """Enhanced data loading with comprehensive validation"""
        uploaded_file = st.file_uploader(
            "📁 Upload Student Database (CSV)", 
            type=['csv'], 
            help="Upload a CSV file containing student performance data"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV with more robust parsing
                df = pd.read_csv(uploaded_file, low_memory=False)
                
                # Enhanced column validation
                required_columns = [
                    'Student_ID', 'Internal_1', 'Internal_2', 'Internal_3',
                    'Assignment_Marks', 'Other_Activities', 'Gender', 'Residence',
                    'Disability', 'Attendance_Percentage', 'Study_Hours_Per_Week',
                    'Part_Time_Job'
                ]
                
                # Comprehensive column check
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    st.error(f"Missing columns: {', '.join(missing_columns)}")
                    return None
                
                # Type checking and data cleaning
                df['Student_ID'] = df['Student_ID'].astype(str)
                
                return df
            
            except pd.errors.EmptyDataError:
                st.error("The uploaded CSV file is empty.")
            except pd.errors.ParserError:
                st.error("Error parsing the CSV file. Please check the file format.")
            except Exception as e:
                st.error(f"Unexpected error loading CSV: {str(e)}")
        
        return None

    def generate_detailed_recommendations(self, student_info, predicted_grade):
        """Generate comprehensive, personalized recommendations"""
        recommendations = {
            'grade_specific': [],
            'internal_analysis': [],
            'improvement_strategies': []
        }
        
        # Grade-Specific Recommendations
        grade_recommendations = {
            'A+': [
                {'type': 'Excellence', 'message': 'Pursue Advanced Academic Challenges', 'icon': '🏆', 'color': 'green'},
                {'type': 'Opportunity', 'message': 'Consider Research or Mentorship Programs', 'icon': '🔬', 'color': 'green'}
            ],
            'A': [
                {'type': 'Consistent', 'message': 'Maintain High Performance, Explore Depth', 'icon': '📈', 'color': 'darkgreen'},
                {'type': 'Growth', 'message': 'Develop Interdisciplinary Skills', 'icon': '🌱', 'color': 'darkgreen'}
            ],
            'B': [
                {'type': 'Potential', 'message': 'Focus on Targeted Academic Improvement', 'icon': '🎯', 'color': 'blue'},
                {'type': 'Strategy', 'message': 'Develop Advanced Study Techniques', 'icon': '📚', 'color': 'blue'}
            ],
            'C': [
                {'type': 'Alert', 'message': 'Requires Comprehensive Academic Support', 'icon': '⚠️', 'color': 'orange'},
                {'type': 'Action', 'message': 'Implement Structured Learning Plan', 'icon': '📋', 'color': 'orange'}
            ],
            'D': [
                {'type': 'Critical', 'message': 'Immediate Academic Intervention Needed', 'icon': '🚨', 'color': 'red'},
                {'type': 'Support', 'message': 'Seek Personalized Tutoring', 'icon': '🤝', 'color': 'red'}
            ]
        }
        
        recommendations['grade_specific'] = grade_recommendations.get(predicted_grade, [])
        
        # Internal Exam Analysis
        internals = [
            ('Internal_1', student_info['Internal_1']),
            ('Internal_2', student_info['Internal_2']),
            ('Internal_3', student_info['Internal_3'])
        ]
        
        for internal_name, marks in internals:
            if marks < 10:
                recommendations['internal_analysis'].append({
                    'type': 'Weakness',
                    'message': f'Critical Improvement Needed in {internal_name}',
                    'icon': '🔍',
                    'color': 'red'
                })
        
        # General Improvement Strategies
        recommendations['improvement_strategies'] = [
            {'type': 'Skill', 'message': 'Enhance Time Management', 'icon': '⏰', 'color': 'purple'},
            {'type': 'Learning', 'message': 'Develop Active Study Techniques', 'icon': '💡', 'color': 'teal'}
        ]
        
        return recommendations

    def create_performance_radar(self, student_info):
        """Create interactive radar chart of student performance"""
        categories = [
            'Internal 1', 'Internal 2', 'Internal 3', 
            'Assignment', 'Activities', 
            'Attendance', 'Study Hours'
        ]
        
        values = [
            student_info['Internal_1'] / 20 * 100,
            student_info['Internal_2'] / 20 * 100,
            student_info['Internal_3'] / 20 * 100,
            student_info['Assignment_Marks'] / 10 * 100,
            student_info['Other_Activities'] / 5 * 100,
            student_info['Attendance_Percentage'],
            min(student_info['Study_Hours_Per_Week'] * 2.5, 100)
        ]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            line_color='blue',
            fillcolor='rgba(0,100,255,0.3)'
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            title='Student Performance Radar'
        )
        
        return fig

    def main(self):
        """Main application flow"""
        st.title("🎓 Student Performance Intelligence Dashboard")
        
        # Load student database
        df = self.load_and_validate_student_data()
        
        if df is not None:
            # Student Search Section
            st.sidebar.header("🔍 Student Search")
            search_query = st.sidebar.text_input(
                "Enter Register Number", 
                help="Search for a student by their unique Student ID"
            )
            
            if search_query:
                # Find student record
                student_record = df[df['Student_ID'].astype(str) == search_query]
                
                if not student_record.empty:
                    # Convert to dictionary for easier processing
                    selected_student = student_record.iloc[0].to_dict()
                    
                    # Display student details
                    st.markdown("""
                        <div style='padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 20px;'>
                            <h3 style='margin: 0;'>🔍 Student Details</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # More comprehensive student details
                    detail_columns = [
                        ('Student ID', 'Student_ID'),
                        ('Gender', 'Gender'),
                        ('Residence', 'Residence'),
                        ('Disability', 'Disability'),
                        ('Attendance', 'Attendance_Percentage'),
                        ('Study Hours/Week', 'Study_Hours_Per_Week')
                    ]
                    
                    cols = st.columns(3)
                    for i, (label, key) in enumerate(detail_columns):
                        with cols[i % 3]:
                            st.info(f"{label}: {selected_student[key]}")
                    
                    # Process Analysis Button
                    if st.sidebar.button("🔄 Process Student Analysis"):
                        # Load model components with fallback
                        model, encoders = self.load_model_components(df)
                        
                        # Prepare data for prediction
                        student_features = self.prepare_student_data(selected_student, encoders)
                        predicted_grade = model.predict(student_features)[0]
                        
                        # Performance Visualization
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.header("🕸️ Performance Radar")
                            spider_chart = self.create_performance_radar(selected_student)
                            st.plotly_chart(spider_chart, use_container_width=True)
                        
                        with col2:
                            st.header("📊 Performance Summary")
                            grade_colors = {
                                'A+': 'darkgreen', 'A': 'green', 
                                'B': 'blue', 'C': 'orange', 
                                'D': 'red', 'F': 'darkred'
                            }
                            st.markdown(f"""
                            <div style="background-color:{grade_colors.get(predicted_grade, 'gray')};
                                        color:white; 
                                        padding:20px; 
                                        border-radius:10px; 
                                        text-align:center; 
                                        font-size:24px;">
                                Predicted Grade: {predicted_grade}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Recommendations Section
                        st.header("🚀 Personalized Recommendations")
                        # Generate recommendations
                        recommendations = self.generate_detailed_recommendations(selected_student, predicted_grade)
                        
                        # Display recommendations
                        recommendation_sections = [
                            ('🌟 Grade-Specific Insights', 'grade_specific'),
                            ('🎯 Internal Exam Analysis', 'internal_analysis'),
                            ('💡 Improvement Strategies', 'improvement_strategies')
                        ]
                        
                        for section_title, section_key in recommendation_sections:
                            if recommendations.get(section_key):
                                st.subheader(section_title)
                                for rec in recommendations[section_key]:
                                    st.markdown(f"""
                                    <div class="recommendation-card" style="border-left: 5px solid {rec['color']};">
                                        <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                            <span style="font-size: 24px; margin-right: 10px;">{rec['icon']}</span>
                                            <strong style="color: {rec['color']}; font-size: 16px;">{rec['type']}</strong>
                                        </div>
                                        <p style="margin: 0; color: #333;">{rec['message']}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                
                else:
                    st.warning(f"No student found with ID: {search_query}")
        
if __name__ == "__main__":
    dashboard = StudentPerformanceDashboard()
    dashboard.main()
