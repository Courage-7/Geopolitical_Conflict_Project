import os
import json
import re
import sys
import io
import time
import contextlib
import warnings
from typing import Optional, List, Any, Tuple
from PIL import Image
import streamlit as st
import pandas as pd
import base64
from io import BytesIO
from together import Together
from e2b_code_interpreter import Sandbox

# Retry decorator
def retry(retries=3, exceptions=(Exception,), delay=5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt < retries - 1:
                        st.warning(f"Retrying... ({attempt + 1}/{retries})")
                        time.sleep(delay)
                    else:
                        raise e
            return wrapper
        return decorator
    return decorator

# File upload function with preview
def upload_and_preview_dataset(code_interpreter: Sandbox, uploaded_file) -> Tuple[str, pd.DataFrame]:
    dataset_path = f"./{uploaded_file.name}"
    try:
        # Read file in chunks
        chunk_size = 2024 * 2024  # 2 MB
        file_content = uploaded_file.read()
        for i in range(0, len(file_content), chunk_size):
            code_interpreter.files.write(dataset_path, file_content[i:i + chunk_size])
        
        # Read the data for preview
        df = pd.read_csv(BytesIO(file_content))
        return dataset_path, df
    except Exception as error:
        st.error(f"Error during file upload: {error}")
        raise error

# Data visualization function
def code_interpret(e2b_code_interpreter: Sandbox, code: str) -> Optional[List[Any]]:
    with st.spinner('Executing code in E2B sandbox...'):
        stdout_capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_capture):
                exec_results = e2b_code_interpreter.run_code(code)
            return exec_results.results
        except Exception as error:
            st.error(f"Error during code execution: {error}")
            return None

# Data analysis LLM interaction
def chat_for_analysis(e2b_code_interpreter: Sandbox, user_message: str, dataset_path: str) -> Tuple[str, str]:
    system_prompt = f"""
    You're a Python data scientist and visualization expert.
    Dataset located at '{dataset_path}'.
    Provide your response in two parts:
    1. INSIGHTS: Write a clear analysis of the data addressing the user's query
    2. VISUALIZATION: Generate Python code to create relevant visualizations
    
    Separate the parts with '---VISUALIZATION_CODE---'
    Focus on creating clear visualizations and statistical insights.
    Use pandas, matplotlib, or seaborn for visualizations.
    Ensure all visualizations have proper titles, labels, and legends.
    """
    client = Together(api_key=st.session_state.together_api_key)
    response = client.chat.completions.create(
        model=st.session_state.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    
    # Split response into insights and visualization code
    content = response.choices[0].message.content
    parts = content.split('---VISUALIZATION_CODE---')
    
    insights = parts[0].strip() if len(parts) > 0 else "No insights generated."
    visualization_code = parts[1].strip() if len(parts) > 1 else ""
    
    return insights, visualization_code

# Conflict mitigation LLM interaction
def chat_for_mitigation(e2b_code_interpreter: Sandbox, user_message: str, dataset_path: str, region: str):
    system_prompt = f"""
    You are an expert in geopolitical conflict resolution and data analysis.
    Dataset at path '{dataset_path}' contains information about conflicts.
    Analyze the trends and suggest mitigation strategies based on historical patterns,
    international frameworks, and UN guidelines for the {region} region.
    Provide specific, actionable recommendations supported by data.
    Include both short-term and long-term strategies.
    """
    client = Together(api_key=st.session_state.together_api_key)
    response = client.chat.completions.create(
        model=st.session_state.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return response.choices[0].message.content

# Predefined frameworks for mitigation
def get_mitigation_frameworks(df: pd.DataFrame, region: str) -> dict:
    try:
        region_data = df[df['Region'].str.contains(region, case=False, na=False)]
        
        frameworks = {
            "statistical_summary": {
                "total_incidents": len(region_data),
                "avg_fatalities": region_data['fatalities'].mean() if 'fatalities' in region_data.columns else None,
                "total_fatalities": region_data['fatalities'].sum() if 'fatalities' in region_data.columns else None,
                "timespan": f"{region_data['date'].min()} to {region_data['date'].max()}" if 'date' in region_data.columns else None
            },
            "high_risk_areas": region_data.groupby('country')['fatalities'].sum().nlargest(5).to_dict() if 'country' in region_data.columns else {},
            "conflict_types": region_data['event_type'].value_counts().to_dict() if 'event_type' in region_data.columns else {}
        }
        return frameworks
    except Exception as e:
        st.error(f"Error in framework analysis: {e}")
        return {}

def main():
    st.title("Geopolitical Conflict Analysis & Mitigation AI Agent") 

    # Sidebar configuration
    with st.sidebar:
        # Navigation Tips
        with st.expander("📌 Navigation Tips", expanded=False):
            st.markdown("""
                ### How to Use This App
                1. **Configuration**
                   - Enter your API keys in the sidebar
                   - Select your preferred model type
                
                2. **Data Upload**
                   - Upload your CSV file with conflict data
                   - Ensure it contains Region, date, event_type and fatalities columns
                
                3. **Analysis Tabs**
                   - **Data Analysis**: Generates analytical insights & visualizations
                   - **Mitigation**: Provides AI-driven conflict resolution strategies
                
                4. **Best Practices**
                   - Start with a free model for testing
                   - Use specific queries for better results
                   - Review the dataset preview before analysis
            """)

        st.header("Configuration")
        if "together_api_key" not in st.session_state:
            st.session_state.together_api_key = ""
        if "e2b_api_key" not in st.session_state:
            st.session_state.e2b_api_key = ""
        if "model_name" not in st.session_state:
            st.session_state.model_name = ""

        st.session_state.together_api_key = st.text_input("Together AI API Key", type="password")
        st.markdown("[Get Together AI API Key](https://www.together.ai/signup)")
        
        st.session_state.e2b_api_key = st.text_input("E2B API Key", type="password")
        st.markdown("[Get E2B API Key](https://e2b.dev/docs/getting-started/api-key)")

        # Model selection with pricing information
        st.subheader("Model Selection")
        
        # Free Models Section
        st.markdown("**🆓 Free Models**")
        free_models = {
            "DeepSeek V3": "deepseek-ai/DeepSeek-V3",
            "Qwen 2.5 7B": "Qwen/Qwen2.5-7B-Instruct-Turbo"
        }
        
        # Paid Models Section
        st.markdown("**💰 Paid Models**")
        paid_models = {
            "Meta-Llama 3.1 405B": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "Meta-Llama 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        }
        
        # Two-step model selection
        model_category = st.radio("Select Model Category", ["Free Models", "Paid Models"])
        
        if model_category == "Free Models":
            selected_model = st.selectbox(
                "Select Free Model",
                options=list(free_models.keys()),
                help="These models are available for use without additional charges"
            )
            st.session_state.model_name = free_models[selected_model]
        else:
            selected_model = st.selectbox(
                "Select Paid Model",
                options=list(paid_models.keys()),
                help="These models require additional credits/payment on TogetherAI"
            )
            st.session_state.model_name = paid_models[selected_model]
            st.warning("⚠️ This is a paid model. Make sure you have sufficient credits on TogetherAI.")

        st.markdown("---")
        st.markdown("[Check TogetherAI Pricing](https://www.together.ai/pricing)")
        
        # About Section
        with st.expander("ℹ️ About", expanded=False):
            st.markdown("""
                ### Geopolitical Conflict Analysis & Mitigation AI Agent
                
                This application combines advanced AI models with data analysis to help:
                - Analyze conflict patterns and trends
                - Generate data-driven visualizations
                - Provide actionable mitigation strategies
                - Support evidence-based decision making
                
                **Technologies Used:**
                - TogetherAI for Large Language Models
                - E2B for Code Interpretation
                - Streamlit for User Interface
                - Python Data Science Stack
                
                **Version:** 1.0.0
                
                **Created by:** Courage Siameh
                
                For support or questions, please contact: courage336@outlook.com
            """)

    # Main application
    st.header("Dataset Upload")
    uploaded_file = st.file_uploader("Upload your conflict dataset", type=["csv"])
    
    if not st.session_state.together_api_key or not st.session_state.e2b_api_key:
        st.warning("Please provide both API keys in the sidebar to proceed.")
        return
    
    if uploaded_file:
        with Sandbox(api_key=st.session_state.e2b_api_key) as code_interpreter:
            try:
                dataset_path, df = upload_and_preview_dataset(code_interpreter, uploaded_file)
                st.success("File uploaded successfully!")
                
                # Dataset Preview Section
                st.header("Dataset Preview")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("First few rows of the dataset:")
                    st.dataframe(df.head())
                
                with col2:
                    st.write("Dataset Information:")
                    st.write(f"Total Rows: {len(df)}")
                    st.write(f"Total Columns: {len(df.columns)}")
                    st.write("Columns:", ", ".join(df.columns.tolist()))
                
                # Tabs for different functionalities
                tab1, tab2 = st.tabs(["Data Analysis & Visualization", "Conflict Mitigation Strategies"])
                
                with tab1:
                    st.header("Data Analysis & Visualization")
                    analysis_query = st.text_area(
                        "What would you like to analyze in your data?",
                        "Show me the trend of conflicts over time and create a visualization."
                    )
                    
                    if st.button("Generate Analysis", key="analysis_button"):
                        with st.spinner("Generating analysis..."):
                            insights, visualization_code = chat_for_analysis(code_interpreter, analysis_query, dataset_path)
                            
                            # Display insights first
                            st.subheader("Analysis Insights")
                            st.write(insights)
                            
                            # Generate and display visualizations
                            if visualization_code.strip():
                                st.subheader("Visualizations")
                                results = code_interpret(code_interpreter, visualization_code)
                                if results:
                                    for result in results:
                                        if hasattr(result, 'png'):
                                            st.image(Image.open(BytesIO(base64.b64decode(result.png))))
                                        elif hasattr(result, 'figure'):
                                            st.pyplot(result.figure)
                                        elif hasattr(result, 'show'):
                                            st.plotly_chart(result)
                                        else:
                                            st.write(result)
                                else:
                                    st.warning("No visualizations could be generated for this query.")
                
                with tab2:
                    st.header("Conflict Mitigation Analysis")
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        region = st.text_input("Region of Interest", "Africa")
                    
                    with col2:
                        timeframe = st.selectbox(
                            "Analysis Timeframe",
                            ["All Time", "Last Year", "Last 5 Years"]
                        )
                    
                    mitigation_query = st.text_area(
                        "What conflict patterns would you like to analyze for mitigation?",
                        "Analyze conflict patterns and suggest de-escalation strategies."
                    )
                    
                    if st.button("Generate Mitigation Strategies", key="mitigation_button"):
                        with st.spinner("Analyzing patterns and generating strategies..."):
                            # Get framework analysis
                            frameworks = get_mitigation_frameworks(df, region)
                            
                            # Get LLM analysis
                            mitigation_response = chat_for_mitigation(
                                code_interpreter, 
                                mitigation_query, 
                                dataset_path, 
                                region
                            )
                            
                            # Display results
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.subheader("AI-Generated Mitigation Strategies")
                                st.write(mitigation_response)
                            
                            with col2:
                                st.subheader("Statistical Overview")
                                if frameworks.get("statistical_summary"):
                                    stats = frameworks["statistical_summary"]
                                    st.metric("Total Incidents", stats["total_incidents"])
                                    if stats["avg_fatalities"]:
                                        st.metric("Average Fatalities", f"{stats['avg_fatalities']:.2f}")
                                    if stats["timespan"]:
                                        st.metric("Time Period", stats["timespan"])
                                
                                if frameworks.get("high_risk_areas"):
                                    st.subheader("High-Risk Areas")
                                    for area, value in frameworks["high_risk_areas"].items():
                                        st.metric(area, f"{value:.0f} fatalities")
                
            except Exception as error:
                st.error(f"Error during processing: {error}")
                st.info("Please check your API keys and try again.")

if __name__ == "__main__":
    main()