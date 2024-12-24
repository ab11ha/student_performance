import streamlit as st
import joblib
import pandas as pd
import os
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import numpy as np
import plotly.express as px


class ModernStudentDashboard:
    def __init__(self):
        self.MODEL_DIR = 'student_performance_models'
        self.setup_page()

    def setup_page(self):
        st.set_page_config(
            page_title="Student Performance Analytics",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.markdown("""
        <style>
            .main { background-color: #1a1a1a; color: #ffffff; }
            .stHeader {
                background: linear-gradient(90deg, #2c3e50, #3498db);
                padding: 2rem;
                border-radius: 10px;
                margin-bottom: 2rem;
            }
            .stCard {
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 1.5rem;
                margin: 1rem 0;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            .stMetric {
                background: linear-gradient(135deg, #00b4db, #0083b0);
                padding: 1rem;
                border-radius: 10px;
                color: white;
            }
            .plot-container {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 15px;
                padding: 1rem;
                margin: 1rem 0;
            }
        </style>
        """, unsafe_allow_html=True)

    def load_and_validate_student_data(self, uploaded_file):
        """Load and validate uploaded student data with proper type handling"""
        try:
            df = pd.read_csv(uploaded_file)
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

            numeric_columns = [
                'Internal_1', 'Internal_2', 'Internal_3',
                'Assignment_Marks', 'Other_Activities',
                'Attendance_Percentage', 'Study_Hours_Per_Week'
            ]

            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            return df

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

    def create_performance_chart(self, student_data):
        """Create comprehensive performance visualization"""
        fig = go.Figure()
        internals = ['Internal_1', 'Internal_2', 'Internal_3']

        for internal in internals:
            fig.add_trace(go.Scatter(
                x=student_data['Subject_Code'],
                y=student_data[internal],
                name=f'{internal}',
                line=dict(width=3),
                mode='lines+markers'
            ))

        fig.update_layout(
            title="Performance Trends Across Subjects",
            xaxis_title="Subjects",
            yaxis_title="Marks",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
            font=dict(color="white"),
            legend=dict(
                bgcolor='rgba(255, 255, 255, 0.1)',
                bordercolor='rgba(255, 255, 255, 0.1)',
                borderwidth=1
            )
        )
        return fig

    def create_attendance_chart(self, attendance_percentage):
        """Create gauge chart for attendance"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=attendance_percentage,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#00b4db"},
                'steps': [
                    {'range': [0, 75], 'color': "red"},
                    {'range': [75, 85], 'color': "yellow"},
                    {'range': [85, 100], 'color': "green"}
                ]
            }
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white")
        )
        return fig

    def display_metrics(self, student_record):
        """Display key performance metrics"""
        cols = st.columns(4)

        metrics = [
            ("Overall Grade", f"{np.mean([student_record['Internal_1'], student_record['Internal_2'], student_record['Internal_3']]):.1f}", "📈"),
            ("Attendance", f"{student_record['Attendance_Percentage']:.1f}%", "📅"),
            ("Study Hours", f"{student_record['Study_Hours_Per_Week']:.1f}h/week", "⏰"),
            ("Activities", f"{student_record['Other_Activities']:.1f}", "🎯")
        ]

        for col, (label, value, icon) in zip(cols, metrics):
            col.markdown(f"""
            <div class="stMetric">
                <h3>{icon} {label}</h3>
                <h2>{value}</h2>
            </div>
            """, unsafe_allow_html=True)

    def main(self):
        st.markdown('<div class="stHeader">', unsafe_allow_html=True)
        st.title("🎓 Student Performance Analytics")
        st.markdown('</div>', unsafe_allow_html=True)

        with st.sidebar:
            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            st.header("🔍 Student Search")
            search_query = st.text_input(
                "Enter Student ID",
                placeholder="e.g., 1",
                help="Enter the student's unique identifier"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "📁 Upload Student Database (CSV)",
            type=['csv'],
            help="Upload student performance data"
        )

        if uploaded_file is not None:
            df = self.load_and_validate_student_data(uploaded_file)

            if df is not None and search_query:
                search_query = str(search_query)
                student_records = df[df['Student_ID'] == search_query]

                if not student_records.empty:
                    st.markdown('<div class="stCard">', unsafe_allow_html=True)
                    st.header(f"Analysis for Student: {search_query}")

                    self.display_metrics(student_records.iloc[0])

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### Performance Trends")
                        perf_chart = self.create_performance_chart(student_records)
                        st.plotly_chart(perf_chart, use_container_width=True)

                    with col2:
                        st.markdown("### Attendance Overview")
                        attendance_chart = self.create_attendance_chart(
                            student_records['Attendance_Percentage'].iloc[0]
                        )
                        st.plotly_chart(attendance_chart, use_container_width=True)

                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning(f"No records found for Student ID: {search_query}")
                    st.write("Available Student IDs:", df['Student_ID'].unique())


if __name__ == "__main__":
    dashboard = ModernStudentDashboard()
    dashboard.main()
