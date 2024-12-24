import streamlit as st
import joblib
import pandas as pd
import os
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

class StudentPerformanceDashboard:
    def __init__(self):  # Fixed initialization method name
        self.MODEL_DIR = 'student_performance_models'
        self.setup_page()

    # Rest of the code remains the same...
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
            background: linear-gradient(135deg, #f8f9fe 0%, #f1f4fd 100%);
            font-family: 'Inter', sans-serif;
        }
        
        .dashboard-header {
            background: linear-gradient(120deg, #2b3de7 0%, #4158d0 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 8px 20px rgba(43, 61, 231, 0.15);
            margin-bottom: 2rem;
        }
        
        .header-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin: 0;
            padding: 0;
        }
        
        .header-subtitle {
            font-size: 1rem;
            opacity: 0.9;
            margin-top: 0.5rem;
        }
        
        .info-card {
            background: white;
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
            border: 1px solid rgba(0,0,0,0.05);
        }
        
        .info-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        }
        
        .metric-card {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            margin: 0.5rem 0;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #2b3de7;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #666;
            margin-top: 0.5rem;
        }
        
        .recommendation-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            border-left: 5px solid;
            transition: transform 0.3s ease;
        }
        
        .recommendation-card:hover {
            transform: translateX(5px);
        }
        
        .sidebar-search {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        }
        
        .stButton>button {
            width: 100%;
            background: linear-gradient(120deg, #2b3de7 0%, #4158d0 100%);
            color: white;
            border: none;
            padding: 0.8rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(43, 61, 231, 0.2);
        }
        
        .chart-container {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        }
        
        .upload-section {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        }
        </style>
        """, unsafe_allow_html=True)

    def create_fallback_encoders(self, df):
        """Create fallback label encoders for categorical variables"""
        encoders = {}
        categorical_columns = {
            'Gender': ['Male', 'Female', 'Other'],
            'Residence': ['Urban', 'Rural'],
            'Disability': ['None', 'Physical', 'Learning']
        }
        
        for col, categories in categorical_columns.items():
            encoder = LabelEncoder()
            if categories:
                encoder.fit(categories)
            else:
                encoder.fit(df[col].fillna('Unknown').unique())
            encoders[col] = encoder
        
        return encoders

    def load_model_components(self, df):
        """Load ML model and encoders with fallback mechanism"""
        try:
            model = joblib.load(os.path.join(self.MODEL_DIR, 'performance_model.joblib'))
            try:
                encoders = joblib.load(os.path.join(self.MODEL_DIR, 'label_encoders.joblib'))
            except:
                st.warning("Using fallback encoders for categorical variables.")
                encoders = self.create_fallback_encoders(df)
            return model, encoders
        except Exception as e:
            st.error(f"Error loading model components: {str(e)}")
            st.stop()

    def prepare_student_data(self, student_info, encoders):
        """Prepare student data for prediction"""
        student_data = student_info.copy()
        
        # Encode categorical variables
        for col in ['Gender', 'Residence', 'Disability']:
            try:
                value = student_info.get(col, 'Unknown')
                if value not in encoders[col].classes_:
                    value = encoders[col].classes_[0]
                student_data[f'{col}_Encoded'] = encoders[col].transform([value])[0]
            except Exception as e:
                st.error(f"Error encoding {col}: {str(e)}")
                student_data[f'{col}_Encoded'] = 0

        # Calculate performance metrics
        max_internal_marks = 20 * 3
        max_assignment_marks = 10
        max_activity_marks = 5
        
        total_internal_marks = sum([
            student_info.get('Internal_1', 0),
            student_info.get('Internal_2', 0),
            student_info.get('Internal_3', 0)
        ])
        total_marks = (total_internal_marks + 
                      student_info.get('Assignment_Marks', 0) + 
                      student_info.get('Other_Activities', 0))
        
        student_data.update({
            'Total_Internal_Marks': total_internal_marks,
            'Total_Marks': total_marks,
            'Performance_Grade_Percentage': (total_marks / (max_internal_marks + max_assignment_marks + max_activity_marks)) * 100
        })
        
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
        """Load and validate student data from CSV"""
        uploaded_file = st.file_uploader(
            "📁 Upload Student Database (CSV)",
            type=['csv'],
            help="Upload a CSV file containing student performance data"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                required_columns = [
                    'Student_ID', 'Internal_1', 'Internal_2', 'Internal_3',
                    'Assignment_Marks', 'Other_Activities', 'Gender', 'Residence',
                    'Disability', 'Attendance_Percentage', 'Study_Hours_Per_Week',
                    'Part_Time_Job'
                ]
                
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    st.error(f"Missing columns: {', '.join(missing_columns)}")
                    return None
                
                # Data cleaning and type conversion
                df['Student_ID'] = df['Student_ID'].astype(str)
                numeric_columns = [
                    'Internal_1', 'Internal_2', 'Internal_3',
                    'Assignment_Marks', 'Other_Activities',
                    'Attendance_Percentage', 'Study_Hours_Per_Week',
                    'Part_Time_Job'
                ]
                
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                return df
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                return None
        
        return None

    def create_performance_radar(self, student_info):
        """Create radar chart for student performance"""
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
            line=dict(color='#2b3de7', width=2),
            fillcolor='rgba(43, 61, 231, 0.2)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    showline=False,
                    gridcolor='rgba(0,0,0,0.1)',
                ),
                angularaxis=dict(
                    gridcolor='rgba(0,0,0,0.1)',
                    linecolor='rgba(0,0,0,0.1)',
                )
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            margin=dict(l=80, r=80, t=40, b=40)
        )
        
        return fig

    def generate_recommendations(self, student_info, predicted_grade):
        """Generate personalized recommendations"""
        recommendations = []
        
        # Grade-based recommendations
        grade_recommendations = {
            'A+': [
                ('Excellence', 'Consider advanced coursework and research opportunities', '🏆'),
                ('Leadership', 'Take on peer tutoring or mentoring roles', '👥')
            ],
            'A': [
                ('Growth', 'Explore additional challenging projects', '📈'),
                ('Balance', 'Maintain current study habits while exploring new areas', '⚖')
            ],
            'B': [
                ('Focus', 'Identify specific areas for improvement', '🎯'),
                ('Practice', 'Increase practice problem solving', '📝')
            ],
            'C': [
                ('Support', 'Seek additional academic support', '🤝'),
                ('Structure', 'Develop a more structured study routine', '📅')
            ],
            'D': [
                ('Urgent', 'Immediate academic intervention required', '⚠'),
                ('Foundation', 'Focus on fundamental concepts', '🔨')
            ]
        }
        
        if predicted_grade in grade_recommendations:
            recommendations.extend(grade_recommendations[predicted_grade])
        
        # Performance-based recommendations
        if student_info['Attendance_Percentage'] < 75:
            recommendations.append(
                ('Attendance', 'Improve class attendance to enhance learning', '📊')
            )
        
        if student_info['Study_Hours_Per_Week'] < 10:
            recommendations.append(
                ('Study Time', 'Increase weekly study hours', '⏰')
            )
        
        return recommendations

    def calculate_total_marks(self, student_info):
        """Calculate total marks and related metrics"""
        total_internal_marks = sum([
            float(student_info.get('Internal_1', 0)),
            float(student_info.get('Internal_2', 0)),
            float(student_info.get('Internal_3', 0))
        ])
        
        assignment_marks = float(student_info.get('Assignment_Marks', 0))
        activity_marks = float(student_info.get('Other_Activities', 0))
        
        return {
            'Total_Internal_Marks': total_internal_marks,
            'Total_Marks': total_internal_marks + assignment_marks + activity_marks,
            'Internal_Percentage': (total_internal_marks / 60) * 100 if total_internal_marks > 0 else 0
        }

    def main(self):
        """Main dashboard interface"""
        st.markdown("""
            <div class="dashboard-header">
                <h1 class="header-title">Student Performance Analytics</h1>
                <p class="header-subtitle">Comprehensive academic performance analysis and insights</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Main content area
        st.markdown("""
            <div class="upload-section">
                <h3>🔍 Student Search</h3>
            </div>
        """, unsafe_allow_html=True)
        
        df = self.load_and_validate_student_data()
        
        if df is not None:
            # Create two columns for search and basic info
            search_col, info_col = st.columns([1, 2])
            
            with search_col:
                student_id = st.text_input(
                    "Enter Student ID",
                    help="Enter the student's unique identifier"
                )
            
            if student_id:
                student_record = df[df['Student_ID'].astype(str) == student_id]
                
                if not student_record.empty:
                    # Get basic student info
                    student_info = student_record.iloc[0].to_dict()
                    
                    # Calculate total marks and add to student_info
                    marks_data = self.calculate_total_marks(student_info)
                    student_info.update(marks_data)
                    
                    # Display basic info in the info column
                    with info_col:
                        st.markdown("""
                            <div class="info-card">
                                <h4>Basic Student Information</h4>
                                <div style="display: flex; justify-content: space-between;">
                                """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"""
                                <p><strong>ID:</strong> {student_info['Student_ID']}</p>
                                <p><strong>Gender:</strong> {student_info['Gender']}</p>
                            """, unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"""
                                <p><strong>Residence:</strong> {student_info['Residence']}</p>
                                <p><strong>Attendance:</strong> {student_info['Attendance_Percentage']}%</p>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("</div></div>", unsafe_allow_html=True)
                    
                    # Center the analyze button
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        analyze_button = st.button("Generate Analysis", key="analyze_button", use_container_width=True)
                    
                    if analyze_button:
                        try:
                            model, encoders = self.load_model_components(df)
                            features = self.prepare_student_data(student_info, encoders)
                            predicted_grade = model.predict(features)[0]
                            
                            # Performance Analysis Section
                            st.markdown("<h2>Performance Analysis</h2>", unsafe_allow_html=True)
                            
                            # Create three columns for metrics
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            
                            # Grade colors (same as before)
                            grade_colors = {
                                'A+': '#28a745', 'A': '#20c997',
                                'B': '#17a2b8', 'C': '#ffc107',
                                'D': '#dc3545', 'F': '#6c757d'
                            }
                            
                            with metric_col1:
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <h3>Predicted Grade</h3>
                                        <div class="metric-value" style="color: {grade_colors.get(predicted_grade, '#6c757d')}">
                                            {predicted_grade}
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with metric_col2:
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-value">{student_info['Total_Internal_Marks']:.1f}/60</div>
                                        <div class="metric-label">Internal Marks</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with metric_col3:
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-value">{student_info['Internal_Percentage']:.1f}%</div>
                                        <div class="metric-label">Internal Percentage</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            # Performance Radar Chart
                            st.markdown("""
                                <div class="chart-container">
                                    <h3>Performance Radar</h3>
                                </div>
                            """, unsafe_allow_html=True)
                            radar_chart = self.create_performance_radar(student_info)
                            st.plotly_chart(radar_chart, use_container_width=True)
                            
                            # Recommendations Section
                            st.markdown("<h2>Personalized Recommendations</h2>", unsafe_allow_html=True)
                            recommendations = self.generate_recommendations(student_info, predicted_grade)
                            
                            # Display recommendations in two columns
                            rec_cols = st.columns(2)
                            for idx, (rec_type, message, icon) in enumerate(recommendations):
                                with rec_cols[idx % 2]:
                                    st.markdown(f"""
                                        <div class="recommendation-card" style="border-left-color: {grade_colors.get(predicted_grade, '#6c757d')}">
                                            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                                                <span style="font-size: 1.5rem; margin-right: 0.8rem;">{icon}</span>
                                                <strong>{rec_type}</strong>
                                            </div>
                                            <p style="margin: 0; color: #444; line-height: 1.5;">{message}</p>
                                        </div>
                                    """, unsafe_allow_html=True)
                            
                            # Additional Insights Section
                            st.markdown("<h2>Additional Insights</h2>", unsafe_allow_html=True)
                            insights_cols = st.columns(2)
                            
                            with insights_cols[0]:
                                self.display_study_pattern_analysis(student_info)
                            
                            with insights_cols[1]:
                                self.display_performance_trend(student_info)
                                
                        except Exception as e:
                            st.error(f"Error in analysis: {str(e)}")
                    
                else:
                    st.error("No student found with the provided ID")
                    st.markdown("""
                        <div class="info-card">
                            <p>Please check the student ID and try again.</p>
                        </div>
                    """, unsafe_allow_html=True)

    def display_study_pattern_analysis(self, student_info):
        st.markdown("""
            <div class="info-card">
                <h4>Study Pattern Analysis</h4>
            """, unsafe_allow_html=True)
        
        study_hours = float(student_info.get('Study_Hours_Per_Week', 0))
        if study_hours < 10:
            st.warning("⚠ Study hours are below recommended levels")
        elif study_hours > 20:
            st.success("✅ Excellent study commitment")
        else:
            st.info("ℹ Adequate study hours")
        
        st.markdown("</div>", unsafe_allow_html=True)

    def display_performance_trend(self, student_info):
        st.markdown("""
            <div class="info-card">
                <h4>Performance Trend</h4>
            """, unsafe_allow_html=True)
        
        try:
            internals = [
                float(student_info['Internal_1']),
                float(student_info['Internal_2']),
                float(student_info['Internal_3'])
            ]
            
            if internals[-1] > internals[0]:
                st.success("📈 Improving performance trend")
            elif internals[-1] < internals[0]:
                st.warning("📉 Declining performance trend")
            else:
                st.info("➡ Stable performance trend")
        except Exception as e:
            st.error("Unable to calculate performance trend")
        
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    dashboard = StudentPerformanceDashboard()
    dashboard.main()
