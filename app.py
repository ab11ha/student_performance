import streamlit as st
import joblib
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px

# Model and Data Preparation Functions
MODEL_DIR = 'student_performance_models'

def load_model_components():
    """Load saved machine learning model and encoders"""
    try:
        model = joblib.load(os.path.join(MODEL_DIR, 'performance_model.joblib'))
        encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.joblib'))
        return model, encoders
    except FileNotFoundError:
        st.error("Model files not found. Please ensure model training is complete.")
        return None, None

def prepare_student_data(student_info, encoders):
    """Prepare student data for model prediction"""
    def safe_encode(encoder, value):
        try:
            return encoder.transform([value])[0]
        except ValueError:
            return 0

    student_data = student_info.copy()
    
    # Encode categorical variables
    student_data['Gender_Encoded'] = safe_encode(encoders['Gender'], student_info['Gender'])
    student_data['Residence_Encoded'] = safe_encode(encoders['Residence'], student_info['Residence'])
    student_data['Disability_Encoded'] = safe_encode(encoders['Disability'], student_info.get('Disability', 'None'))
    
    # Calculate performance metrics
    max_internal_marks = 20 * 3
    max_assignment_marks = 10
    max_activity_marks = 5
    max_total_marks = max_internal_marks + max_assignment_marks + max_activity_marks
    
    total_internal_marks = student_info['Internal_1'] + student_info['Internal_2'] + student_info['Internal_3']
    total_marks = total_internal_marks + student_info['Assignment_Marks'] + student_info['Other_Activities']
    
    student_data['Total_Internal_Marks'] = total_internal_marks
    student_data['Total_Marks'] = total_marks
    student_data['Performance_Grade_Percentage'] = (total_marks / max_total_marks) * 100
    
    # Select features for prediction
    features = [
        'Internal_1', 'Internal_2', 'Internal_3', 
        'Assignment_Marks', 'Other_Activities', 
        'Total_Internal_Marks', 'Total_Marks',
        'Gender_Encoded', 'Residence_Encoded', 
        'Disability_Encoded', 'Attendance_Percentage', 
        'Study_Hours_Per_Week', 'Part_Time_Job'
    ]
    
    return pd.DataFrame([student_data])[features]

def generate_detailed_recommendations(student_info, predicted_grade):
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

def create_performance_radar(student_info):
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

def main():
    st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
    
    # Custom Styling
    st.markdown("""
    <style>
    .stApp { background-color: #f4f6f9; }
    .recommendation-card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 15px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎓 Student Performance Intelligence Dashboard")
    
    # Sidebar Input Section
    with st.sidebar:
        st.header("📝 Student Information")
        
        # Performance Inputs
        internal_1 = st.number_input("Internal 1 Marks (out of 20)", min_value=0.0, max_value=20.0, step=0.1, key='internal_1')
        internal_2 = st.number_input("Internal 2 Marks (out of 20)", min_value=0.0, max_value=20.0, step=0.1, key='internal_2')
        internal_3 = st.number_input("Internal 3 Marks (out of 20)", min_value=0.0, max_value=20.0, step=0.1, key='internal_3')
        
        assignment_marks = st.number_input("Assignment Marks (out of 10)", min_value=0.0, max_value=10.0, step=0.1, key='assignment_marks')
        other_activities = st.number_input("Other Activities Marks (out of 5)", min_value=0.0, max_value=5.0, step=0.1, key='other_activities')
        
        gender = st.selectbox("Gender", ['Male', 'Female', 'Other'], key='gender')
        residence = st.selectbox("Residence", ['Urban', 'Rural', 'Suburban'], key='residence')
        disability = st.selectbox("Disability", ['None', 'Physical', 'Learning'], key='disability')
        
        attendance = st.slider("Attendance Percentage", min_value=0, max_value=100, value=80, key='attendance')
        study_hours = st.number_input("Study Hours Per Week", min_value=0.0, max_value=40.0, step=0.5, key='study_hours')
        part_time_job = st.selectbox("Part-Time Job", [0, 1], key='part_time_job')
        
        predict_button = st.button("🔮 Analyze Performance")
    
    # Main Analysis Section
    if predict_button:
        # Collect Student Information
        student_info = {
            'Internal_1': internal_1,
            'Internal_2': internal_2,
            'Internal_3': internal_3,
            'Assignment_Marks': assignment_marks,
            'Other_Activities': other_activities,
            'Gender': gender,
            'Residence': residence,
            'Disability': disability,
            'Attendance_Percentage': attendance,
            'Study_Hours_Per_Week': study_hours,
            'Part_Time_Job': part_time_job
        }
        
        # Load Model and Predict
        model, encoders = load_model_components()
        student_features = prepare_student_data(student_info, encoders)
        predicted_grade = model.predict(student_features)[0]
        
        # Performance Visualization
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("🕸️ Performance Radar")
            spider_chart = create_performance_radar(student_info)
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
                        padding:10px; 
                        border-radius:10px; 
                        text-align:center; 
                        font-size:24px;">
                Predicted Grade: {predicted_grade}
            </div>
            """, unsafe_allow_html=True)
        
        # Recommendations Section
        st.header("🚀 Personalized Recommendations")
        recommendations = generate_detailed_recommendations(student_info, predicted_grade)
        
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
                        <div style="color:{rec['color']}; font-weight:bold;">
                            {rec['icon']} {rec['type']}: {rec['message']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
