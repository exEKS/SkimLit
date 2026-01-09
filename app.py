"""Streamlit web application for SkimLit."""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="SkimLit - RCT Abstract Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API URL from environment or default
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Color mapping for labels
LABEL_COLORS = {
    'BACKGROUND': '#2E86AB',
    'OBJECTIVE': '#F77F00',
    'METHODS': '#6A4C93',
    'RESULTS': '#06A77D',
    'CONCLUSIONS': '#D62828'
}

LABEL_BG_COLORS = {
    'BACKGROUND': '#E8F4F8',
    'OBJECTIVE': '#FFF4E6',
    'METHODS': '#F0F4FF',
    'RESULTS': '#E8F5E9',
    'CONCLUSIONS': '#FFF0F5'
}


def check_api_health():
    """Check if API is available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def call_api(text: str, return_probabilities: bool = False):
    """Call the SkimLit API."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={
                "text": text,
                "return_probabilities": return_probabilities
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code}"
            
    except requests.exceptions.Timeout:
        return None, "Request timeout. Please try again."
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Please check if it's running."
    except Exception as e:
        return None, str(e)


def display_structured_abstract(predictions: List[Dict]):
    """Display the structured abstract with color coding."""
    st.markdown("### 📋 Structured Abstract")
    
    for pred in predictions:
        label = pred['label']
        text = pred['text']
        confidence = pred['confidence']
        
        # Create colored box
        st.markdown(
            f"""
            <div style='
                background-color: {LABEL_BG_COLORS[label]};
                padding: 15px;
                border-radius: 8px;
                border-left: 4px solid {LABEL_COLORS[label]};
                margin-bottom: 12px;
            '>
                <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                    <strong style='color: {LABEL_COLORS[label]};'>{label}</strong>
                    <span style='color: #666; font-size: 0.9em;'>{confidence:.1%}</span>
                </div>
                <div style='color: #1f2937; line-height: 1.6;'>{text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def create_confidence_chart(predictions: List[Dict]):
    """Create a confidence distribution chart."""
    df = pd.DataFrame(predictions)
    
    fig = px.bar(
        df,
        x='line_number',
        y='confidence',
        color='label',
        title='Prediction Confidence by Sentence',
        labels={'line_number': 'Sentence Number', 'confidence': 'Confidence'},
        color_discrete_map=LABEL_COLORS,
        height=400
    )
    
    fig.update_layout(
        xaxis_title="Sentence Number",
        yaxis_title="Confidence",
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig


def create_label_distribution(predictions: List[Dict]):
    """Create label distribution pie chart."""
    df = pd.DataFrame(predictions)
    label_counts = df['label'].value_counts()
    
    fig = px.pie(
        values=label_counts.values,
        names=label_counts.index,
        title='Label Distribution',
        color=label_counts.index,
        color_discrete_map=LABEL_COLORS,
        height=400
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("📄 SkimLit: AI-Powered Medical Abstract Analyzer")
    st.markdown("""
    This tool uses deep learning to automatically structure unformatted medical research abstracts,
    making it easier for researchers to quickly understand and navigate scientific literature.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **SkimLit** analyzes RCT abstracts and classifies each sentence into:
        - 🔵 **BACKGROUND**: Context and motivation
        - 🟠 **OBJECTIVE**: Research goals
        - 🟣 **METHODS**: Study methodology
        - 🟢 **RESULTS**: Key findings
        - 🔴 **CONCLUSIONS**: Implications
        """)
        
        st.header("⚙️ Settings")
        show_confidence = st.checkbox("Show confidence scores", value=True)
        show_charts = st.checkbox("Show visualizations", value=True)
        return_probabilities = st.checkbox("Include all probabilities", value=False)
        
        st.header("📊 API Status")
        if check_api_health():
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")
            st.info(f"API URL: {API_URL}")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📝 Analyze", "📚 Examples", "ℹ️ Help"])
    
    with tab1:
        st.header("Paste your abstract")
        
        # Example abstracts
        example_abstracts = {
            "Diabetes Treatment": """This study examined the effects of a new drug on diabetes management. 
            A total of 200 patients were randomly assigned to treatment or control groups. 
            The treatment group received the drug daily for 12 weeks. 
            Blood glucose levels were measured weekly. 
            Results showed a 25% reduction in blood glucose in the treatment group. 
            The new drug appears to be effective for diabetes management.""",
            
            "Cancer Research": """The objective was to evaluate the efficacy of immunotherapy in breast cancer patients.
            We conducted a randomized controlled trial with 500 participants.
            Patients received either immunotherapy or standard chemotherapy.
            The primary outcome was progression-free survival.
            Immunotherapy showed a 40% improvement in progression-free survival.
            These findings suggest immunotherapy as a promising treatment option.""",
            
            "Custom": ""
        }
        
        selected_example = st.selectbox(
            "Load example:",
            options=list(example_abstracts.keys())
        )
        
        default_text = example_abstracts[selected_example]
        
        user_input = st.text_area(
            "Enter abstract text:",
            value=default_text,
            height=200,
            help="Paste an unstructured RCT abstract"
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
        
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.rerun()
        
        if analyze_button and user_input:
            with st.spinner("Analyzing abstract..."):
                result, error = call_api(user_input, return_probabilities)
                
                if error:
                    st.error(f"Error: {error}")
                elif result:
                    # Success metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sentences", result['total_sentences'])
                    with col2:
                        st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                    with col3:
                        avg_conf = sum(s['confidence'] for s in result['sentences']) / len(result['sentences'])
                        st.metric("Avg Confidence", f"{avg_conf:.1%}")
                    
                    st.success("✅ Abstract structured successfully!")
                    
                    # Display structured abstract
                    display_structured_abstract(result['sentences'])
                    
                    # Show visualizations
                    if show_charts:
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.plotly_chart(
                                create_confidence_chart(result['sentences']),
                                use_container_width=True
                            )
                        
                        with col2:
                            st.plotly_chart(
                                create_label_distribution(result['sentences']),
                                use_container_width=True
                            )
                    
                    # Download options
                    st.markdown("---")
                    st.markdown("### 💾 Export Results")
                    
                    # Create formatted text
                    formatted_text = "\n\n".join([
                        f"{pred['label']}: {pred['text']}"
                        for pred in result['sentences']
                    ])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            "📄 Download as Text",
                            formatted_text,
                            file_name="structured_abstract.txt",
                            mime="text/plain"
                        )
                    
                    with col2:
                        df = pd.DataFrame(result['sentences'])
                        st.download_button(
                            "📊 Download as CSV",
                            df.to_csv(index=False),
                            file_name="structured_abstract.csv",
                            mime="text/csv"
                        )
    
    with tab2:
        st.header("Example Abstracts")
        st.markdown("Try these example abstracts to see how SkimLit works:")
        
        for title, abstract in list(example_abstracts.items())[:-1]:  # Exclude "Custom"
            with st.expander(title):
                st.text(abstract)
                if st.button(f"Analyze {title}", key=f"btn_{title}"):
                    st.session_state.example_text = abstract
                    st.rerun()
    
    with tab3:
        st.header("How to Use")
        
        st.markdown("""
        ### Quick Start
        
        1. **Paste Abstract**: Copy an RCT abstract from PubMed
        2. **Click Analyze**: Press the 🔍 Analyze button
        3. **View Results**: See structured, color-coded sentences
        
        ### Understanding Labels
        
        - **BACKGROUND** 🔵: Provides context and motivation for the study
        - **OBJECTIVE** 🟠: States the research question or hypothesis
        - **METHODS** 🟣: Describes how the study was conducted
        - **RESULTS** 🟢: Presents key findings and data
        - **CONCLUSIONS** 🔴: Summarizes implications and significance
        
        ### Tips
        
        - Longer abstracts work better (5+ sentences)
        - Use complete sentences
        - Medical/scientific text gives best results
        - Try the examples to see it in action
        
        ### API Configuration
        
        The app connects to: `{API_URL}`
        
        To change this, set the `API_URL` environment variable.
        """)
        
        st.header("About the Model")
        st.markdown("""
        This tool uses a **tribrid neural network** combining:
        - Token embeddings (Universal Sentence Encoder)
        - Character-level embeddings (Bi-LSTM)
        - Positional embeddings (sentence position)
        
        **Performance**:
        - Accuracy: ~87%
        - Processing: <100ms per abstract
        - Trained on 200k+ medical abstracts
        
        **Citation**: Based on [PubMed 200k RCT Dataset](https://arxiv.org/abs/1710.06071)
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Built with ❤️ using TensorFlow and FastAPI | 
        <a href='https://github.com/yourusername/skimlit'>GitHub</a> | 
        <a href='https://arxiv.org/abs/1710.06071'>Paper</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()