"""
IEI Diagnostic Chatbot - Streamlit Interface
Interactive diagnostic tool using Shannon's Information Theory
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from iei_diagnostic_engine import (
    IEIDiagnosticEngine,
    QUESTIONS,
    IEI_CATEGORIES,
    calculate_entropy
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="IEI Diagnostic Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .question-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .pattern-alert {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .diagnosis-box {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #155a8a;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'engine' not in st.session_state:
    st.session_state.engine = IEIDiagnosticEngine()
    st.session_state.history = []
    st.session_state.current_result = None
    st.session_state.diagnosis_complete = False
    st.session_state.question_count = 0

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<div class="main-header">🧬 IEI Diagnostic Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Powered by Shannon\'s Information Theory & Clinical Pattern Recognition</div>', 
    unsafe_allow_html=True
)

# ============================================================================
# SIDEBAR - Patient Info & Statistics
# ============================================================================

with st.sidebar:
    st.header("📊 Diagnostic Dashboard")
    
    # Patient information
    st.subheader("Patient Information")
    patient_id = st.text_input("Patient ID (optional)", placeholder="e.g., IEI-2024-001")
    
    st.divider()
    
    # Current statistics
    st.subheader("Current Session")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Questions Asked", st.session_state.question_count)
    with col2:
        current_entropy = calculate_entropy(st.session_state.engine.current_probs)
        st.metric("Entropy", f"{current_entropy:.2f} bits")
    
    # Top 3 leading diagnoses
    st.subheader("Leading Categories")
    top_3 = st.session_state.engine.get_top_diagnoses(n=3)
    
    for i, (category, prob) in enumerate(top_3, 1):
        st.write(f"{i}. **{category.replace('_', ' ')}**")
        st.progress(prob)
        st.caption(f"{prob*100:.1f}%")
    
    st.divider()
    
    # Reset button
    if st.button("🔄 Reset & Start New Case"):
        st.session_state.engine.reset()
        st.session_state.history = []
        st.session_state.current_result = None
        st.session_state.diagnosis_complete = False
        st.session_state.question_count = 0
        st.rerun()
    
    # About section
    st.divider()
    st.subheader("ℹ️ About")
    st.caption("""
    This diagnostic assistant uses:
    - **Shannon's Entropy** for information gain
    - **Bayesian Updates** for probability refinement
    - **Pattern Recognition** for pathognomonic findings
    - **Weighted Question Selection** for optimal diagnostic paths
    """)

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["🎯 Diagnostic Interview", "📈 Probability Analysis", "📋 Case Summary"])

# ============================================================================
# TAB 1: DIAGNOSTIC INTERVIEW
# ============================================================================

with tab1:
    # Show diagnosis if complete
    if st.session_state.diagnosis_complete and st.session_state.current_result is not None:
        result = st.session_state.current_result
        
        st.markdown('<div class="diagnosis-box">', unsafe_allow_html=True)
        st.success("### ✅ Diagnostic Assessment Complete")
        
        if result.get('status') == 'pattern_detected':
            st.write(f"**Suspected Diagnosis:** {result.get('suspected_diagnosis', 'Unknown')}")
            st.write(f"**Confidence:** {result.get('confidence', 0)*100:.1f}%")
            st.write(f"**Category:** {result.get('category', 'Unknown').replace('_', ' ')}")
            
            confirm_with = result.get('confirm_with', [])
            if confirm_with:
                st.info("**Recommended confirmatory questions:**")
                for q_id in confirm_with:
                    if q_id in QUESTIONS:
                        st.write(f"- {QUESTIONS[q_id].text}")
        
        elif result.get('status') == 'diagnosis_reached':
            top_dx = result.get('top_diagnosis')
            if top_dx:
                st.write(f"**Primary Category:** {top_dx[0].replace('_', ' ')}")
                st.write(f"**Confidence:** {top_dx[1]*100:.1f}%")
            
            differential = result.get('differential', [])
            if differential:
                st.write("**Differential Diagnosis (Top 5):**")
                for i, (cat, prob) in enumerate(differential, 1):
                    st.write(f"{i}. {cat.replace('_', ' ')}: {prob*100:.1f}%")
        
        elif result.get('status') == 'questions_exhausted':
            st.write("**Diagnostic Assessment:**")
            st.write("All available questions have been asked.")
            differential = result.get('differential', [])
            if differential:
                st.write("**Top Differential Diagnosis:**")
                for i, (cat, prob) in enumerate(differential, 1):
                    st.write(f"{i}. {cat.replace('_', ' ')}: {prob*100:.1f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Option to continue or restart
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 Export Case Report"):
                st.info("Case report export feature coming soon!")
        with col2:
            if st.button("🔄 Start New Case"):
                st.session_state.engine.reset()
                st.session_state.history = []
                st.session_state.current_result = None
                st.session_state.diagnosis_complete = False
                st.session_state.question_count = 0
                st.rerun()
    
    # Show current question
    else:
        # Start with initial question if no history
        if not st.session_state.history:
            st.markdown('<div class="question-box">', unsafe_allow_html=True)
            st.write("### Let's begin the diagnostic interview")
            st.write("I'll ask you strategic questions to narrow down the diagnosis efficiently.")
            st.write("**Ready to start?**")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show first question (highest IG - Q15)
            first_q_id = "Q15"
            first_q = QUESTIONS[first_q_id]
            
            st.markdown('<div class="question-box">', unsafe_allow_html=True)
            st.write(f"### Question {st.session_state.question_count + 1}")
            st.write(f"**{first_q.text}**")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Answer options
            answer = st.radio(
                "Select answer:",
                options=first_q.answer_options,
                key=f"q_{first_q_id}"
            )
            
            if st.button("Submit Answer", key=f"submit_{first_q_id}"):
                # Process answer
                result = st.session_state.engine.process_answer(first_q_id, answer)
                st.session_state.history.append({
                    'question': first_q.text,
                    'answer': answer,
                    'question_id': first_q_id
                })
                st.session_state.current_result = result
                st.session_state.question_count += 1
                
                # Check if diagnosis complete
                if result['status'] in ['pattern_detected', 'diagnosis_reached', 'questions_exhausted']:
                    st.session_state.diagnosis_complete = True
                
                st.rerun()
        
        # Continue with next question
        elif st.session_state.current_result and st.session_state.current_result['status'] == 'continue':
            result = st.session_state.current_result
            
            # Show pattern alert if detected but not conclusive
            if st.session_state.engine.pathognomonic_match:
                pattern = st.session_state.engine.pathognomonic_match
                st.markdown('<div class="pattern-alert">', unsafe_allow_html=True)
                st.warning(f"⚠️ **Pattern Detected:** {pattern.name} (Confidence: {pattern.probability*100:.0f}%)")
                st.write("Asking confirmatory questions...")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Show current question
            next_q_id = result['next_question']
            next_q = QUESTIONS[next_q_id]
            
            st.markdown('<div class="question-box">', unsafe_allow_html=True)
            st.write(f"### Question {st.session_state.question_count + 1}")
            st.write(f"**{next_q.text}**")
            
            # Show why this question matters
            with st.expander("ℹ️ Why this question?"):
                st.write(f"**Information Gain:** {next_q.base_information_gain:.2f} bits")
                if next_q.is_nodal:
                    st.write("🔑 **Key Decision Point** - This question routes to major diagnostic categories")
                st.write(f"**Current Entropy:** {result['current_entropy']:.2f} bits")
                st.write("This question maximizes information gain based on current probabilities.")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Answer options
            answer = st.radio(
                "Select answer:",
                options=next_q.answer_options,
                key=f"q_{next_q_id}"
            )
            
            if st.button("Submit Answer", key=f"submit_{next_q_id}"):
                # Process answer
                result = st.session_state.engine.process_answer(next_q_id, answer)
                st.session_state.history.append({
                    'question': next_q.text,
                    'answer': answer,
                    'question_id': next_q_id
                })
                st.session_state.current_result = result
                st.session_state.question_count += 1
                
                # Check if diagnosis complete
                if result['status'] in ['pattern_detected', 'diagnosis_reached', 'questions_exhausted']:
                    st.session_state.diagnosis_complete = True
                
                st.rerun()
    
    # Show history
    if st.session_state.history:
        st.divider()
        st.subheader("📜 Interview History")
        
        for i, item in enumerate(st.session_state.history, 1):
            with st.expander(f"Q{i}: {item['question'][:50]}..."):
                st.write(f"**Question:** {item['question']}")
                st.write(f"**Answer:** {item['answer']}")

# ============================================================================
# TAB 2: PROBABILITY ANALYSIS
# ============================================================================

with tab2:
    st.header("📈 Real-Time Probability Analysis")
    
    # Current probability distribution
    st.subheader("Current Probability Distribution")
    
    probs_df = pd.DataFrame([
        {'Category': cat.replace('_', ' '), 'Probability': prob}
        for cat, prob in st.session_state.engine.current_probs.items()
    ]).sort_values('Probability', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        probs_df,
        x='Category',
        y='Probability',
        title='Probability Distribution Across IEI Categories',
        color='Probability',
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        xaxis_title="IEI Category",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Entropy evolution over time
    if st.session_state.history:
        st.subheader("🎲 Entropy Evolution")
        st.write("Watch how uncertainty decreases with each question:")
        
        # We'd need to track entropy at each step for this
        # For now, show current entropy
        current_entropy = calculate_entropy(st.session_state.engine.current_probs)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Initial Entropy", "2.43 bits")
        with col2:
            st.metric("Current Entropy", f"{current_entropy:.2f} bits")
        with col3:
            reduction = ((2.43 - current_entropy) / 2.43) * 100
            st.metric("Entropy Reduction", f"{reduction:.1f}%")
        
        st.info("💡 Lower entropy = Higher diagnostic certainty")
    
    # Detailed probability table
    st.subheader("📊 Detailed Probabilities")
    
    # Color-coded probability display
    for idx, row in probs_df.iterrows():
        cat = row['Category']
        prob = row['Probability']
        
        # Color code based on probability
        if prob > 0.5:
            color = "🔴"  # High probability
        elif prob > 0.2:
            color = "🟡"  # Medium probability
        elif prob > 0.05:
            color = "🟢"  # Low probability
        else:
            color = "⚪"  # Very low probability
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"{color} **{cat}**")
        with col2:
            st.write(f"{prob*100:.1f}%")
        st.progress(prob)
        
        if idx < len(probs_df) - 1:  # Don't add divider after last item
            st.divider()

# ============================================================================
# TAB 3: CASE SUMMARY
# ============================================================================

with tab3:
    st.header("📋 Case Summary")
    
    if patient_id:
        st.write(f"**Patient ID:** {patient_id}")
    
    st.write(f"**Questions Asked:** {st.session_state.question_count}")
    st.write(f"**Status:** {'Diagnosis Complete' if st.session_state.diagnosis_complete else 'In Progress'}")
    
    st.divider()
    
    # Clinical findings summary
    st.subheader("Clinical Findings")
    
    if st.session_state.history:
        findings_df = pd.DataFrame([
            {
                'Question': item['question'],
                'Finding': item['answer']
            }
            for item in st.session_state.history
        ])
        st.dataframe(findings_df, use_container_width=True, hide_index=True)
    else:
        st.info("No findings recorded yet. Start the diagnostic interview!")
    
    st.divider()
    
    # Diagnostic reasoning
    st.subheader("Diagnostic Reasoning")
    
    if st.session_state.diagnosis_complete and st.session_state.current_result is not None:
        result = st.session_state.current_result
        
        st.write("**Final Assessment:**")
        
        if result.get('status') == 'pattern_detected':
            st.write(f"- Pattern-based diagnosis: **{result.get('suspected_diagnosis', 'Unknown')}**")
            st.write(f"- Confidence level: **{result.get('confidence', 0)*100:.1f}%**")
            st.write(f"- Primary category: **{result.get('category', 'Unknown').replace('_', ' ')}**")
        
        elif result.get('status') == 'diagnosis_reached':
            st.write("**Top Differential Diagnosis:**")
            differential = result.get('differential', [])
            if differential:
                for i, (cat, prob) in enumerate(differential, 1):
                    st.write(f"{i}. {cat.replace('_', ' ')}: **{prob*100:.1f}%**")
            else:
                st.warning("No differential diagnosis available.")
        
        st.divider()
        
        st.write("**Recommended Next Steps:**")
        st.write("1. Review patient's immunological workup")
        st.write("2. Consider genetic testing for suspected category")
        st.write("3. Consult with IEI specialist")
        st.write("4. Review IUIS classification for specific diagnoses")
    
    else:
        st.info("Complete the diagnostic interview to see the final assessment.")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🧬 Based on IUIS 2024 Classification")
with col2:
    st.caption("📊 Powered by Shannon's Information Theory")
with col3:
    st.caption("🏥 National Institute of Pediatrics, Mexico")

st.caption("⚠️ This is a clinical decision support tool. Always confirm with comprehensive evaluation and expert consultation.")
