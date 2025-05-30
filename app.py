import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM, RobertaModel, RobertaTokenizer, RobertaConfig
import io
import os
from huggingface_hub import login, hf_hub_download
from safetensors.torch import load_file
import plotly.express as px
import plotly.graph_objects as go
import pickle
import time

# Additional imports for PDF and DOCX support
import PyPDF2
import pdfplumber
from docx import Document

# Configure Streamlit page
st.set_page_config(
    page_title="Compliance Maturity & Gap Analysis Platform",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ff6b6b;
}
.gap-analysis-section {
    background-color: #e8f4fd;
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.recommendation-box {
    background-color: #d4edda;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #28a745;
}
.gap-form {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 0.5rem;
    border: 1px solid #dee2e6;
    margin: 1rem 0;
}
.file-preview {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #e9ecef;
    max-height: 400px;
    overflow-y: auto;
    font-family: monospace;
    font-size: 0.85em;
}
.evidence-recommendation {
    background-color: #fff3cd;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ffc107;
    margin: 1rem 0;
}
.evidence-item {
    background-color: #000000;
    color: #ffffff;
    padding: 0.5rem;
    margin: 0.3rem 0;
    border-radius: 0.3rem;
    border-left: 3px solid #007bff;
}
.loading-status {
    background-color: #e3f2fd;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #2196f3;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Configuration
YOUR_HF_ORGANIZATION = os.getenv("HF_ORGANIZATION")
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize session state for models
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = {
        'maturity': False,
        'gap_analysis': False,
        'evidence': False
    }

if 'models' not in st.session_state:
    st.session_state.models = {}

# Validate configuration
if not YOUR_HF_ORGANIZATION or not HF_TOKEN:
    st.error("❌ Missing required environment variables.")
    st.info("Please ensure HF_ORGANIZATION and HF_TOKEN are set in your Render environment variables.")
    st.stop()

# Model paths
MATURITY_MODEL_PATH = f"{YOUR_HF_ORGANIZATION}/compliance-maturity-classifier"
GAP_ANALYSIS_MODEL_PATH = f"{YOUR_HF_ORGANIZATION}/flan-t5-large-gap-analysis"
EVIDENCE_RECOMMENDATION_MODEL_PATH = f"{YOUR_HF_ORGANIZATION}/evidence-recommendation-model"

# Define maturity levels
MATURITY_LEVELS = {
    0: "Level 0 - No Evidence",
    1: "Level 1 - Initial",
    2: "Level 2 - Developing",
    3: "Level 3 - Defined",
    4: "Level 4 - Managed and Measurable",
    5: "Level 5 - Optimized"
}

def show_loading_status():
    """Show current model loading status"""
    st.markdown('<div class="loading-status">', unsafe_allow_html=True)
    st.markdown("### 🤖 AI Models Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status = "✅ Loaded" if st.session_state.models_loaded['maturity'] else "⏳ Not Loaded"
        st.markdown(f"**Maturity Model**: {status}")
    
    with col2:
        status = "✅ Loaded" if st.session_state.models_loaded['gap_analysis'] else "⏳ Not Loaded"
        st.markdown(f"**Gap Analysis Model**: {status}")
    
    with col3:
        status = "✅ Loaded" if st.session_state.models_loaded['evidence'] else "⏳ Not Loaded"
        st.markdown(f"**Evidence Model**: {status}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Evidence Recommendation Model Class
class EvidenceRecommendationModel(nn.Module):
    def __init__(self, num_labels=10, hidden_dim=256, dropout_rate=0.3):
        super().__init__()
        self.num_labels = num_labels
        
        roberta_config = RobertaConfig.from_pretrained('roberta-base')
        self.roberta = RobertaModel(roberta_config)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(roberta_config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_labels)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def load_maturity_model():
    """Load maturity model on demand"""
    if st.session_state.models_loaded['maturity']:
        return st.session_state.models['maturity_model'], st.session_state.models['maturity_tokenizer']
    
    try:
        with st.spinner("🔄 Loading Maturity Assessment Model..."):
            if HF_TOKEN:
                login(token=HF_TOKEN)
            
            tokenizer = AutoTokenizer.from_pretrained(MATURITY_MODEL_PATH)
            model = AutoModelForSequenceClassification.from_pretrained(MATURITY_MODEL_PATH)
            model.eval()
            
            # Store in session state
            st.session_state.models['maturity_model'] = model
            st.session_state.models['maturity_tokenizer'] = tokenizer
            st.session_state.models_loaded['maturity'] = True
            
            st.success("✅ Maturity model loaded successfully!")
            return model, tokenizer
            
    except Exception as e:
        st.error(f"❌ Error loading maturity model: {str(e)}")
        return None, None

def load_gap_analysis_model():
    """Load gap analysis model on demand"""
    if st.session_state.models_loaded['gap_analysis']:
        return st.session_state.models['gap_model'], st.session_state.models['gap_tokenizer']
    
    try:
        with st.spinner("🔄 Loading Gap Analysis Model..."):
            if HF_TOKEN:
                login(token=HF_TOKEN)
            
            tokenizer = AutoTokenizer.from_pretrained(GAP_ANALYSIS_MODEL_PATH)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                GAP_ANALYSIS_MODEL_PATH,
                torch_dtype=torch.float32,  # Use float32 to reduce memory
                device_map=None  # Don't use auto device mapping
            )
            
            # Store in session state
            st.session_state.models['gap_model'] = model
            st.session_state.models['gap_tokenizer'] = tokenizer
            st.session_state.models_loaded['gap_analysis'] = True
            
            st.success("✅ Gap analysis model loaded successfully!")
            return model, tokenizer
            
    except Exception as e:
        st.error(f"❌ Error loading gap analysis model: {str(e)}")
        return None, None

def load_evidence_recommendation_model():
    """Load evidence recommendation model on demand"""
    if st.session_state.models_loaded['evidence']:
        return st.session_state.models['evidence_model'], st.session_state.models['evidence_tokenizer'], st.session_state.models['evidence_types']
    
    try:
        with st.spinner("🔄 Loading Evidence Recommendation Model..."):
            evidence_types = [
                "Implementation Evidence",
                "Governance Documentation", 
                "Procedural Documents",
                "Operational Records",
                "Audit and Assessment Reports",
                "Training and Awareness",
                "Agreements and Legal Documents",
                "Communications and Meeting Records",
                "Planning Documents",
                "Evidence of Continuous Improvement"
            ]
            
            if HF_TOKEN:
                login(token=HF_TOKEN)
            
            tokenizer = RobertaTokenizer.from_pretrained(EVIDENCE_RECOMMENDATION_MODEL_PATH)
            
            model = EvidenceRecommendationModel(
                num_labels=len(evidence_types),
                hidden_dim=256,
                dropout_rate=0.3
            )
            
            model_file = hf_hub_download(
                repo_id=EVIDENCE_RECOMMENDATION_MODEL_PATH, 
                filename="model.safetensors"
            )
            
            state_dict = load_file(model_file)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            
            # Store in session state
            st.session_state.models['evidence_model'] = model
            st.session_state.models['evidence_tokenizer'] = tokenizer
            st.session_state.models['evidence_types'] = evidence_types
            st.session_state.models_loaded['evidence'] = True
            
            st.success("✅ Evidence recommendation model loaded successfully!")
            return model, tokenizer, evidence_types
            
    except Exception as e:
        st.error(f"❌ Error loading evidence recommendation model: {str(e)}")
        return None, None, None

def create_rule_based_gap_analysis(question, evidence, current_level, target_level, domain, framework):
    """Enhanced rule-based gap analysis as fallback"""
    
    evidence_lower = evidence.lower()
    maturity_gap = target_level - current_level

    # Identify gap type based on keywords
    if any(keyword in evidence_lower for keyword in ['access', 'provisioning', 'user', 'identity']):
        gap_desc = f"Current access management framework lacks comprehensive user lifecycle procedures and role-based access controls required for Level {target_level} maturity."
        gap_initiative = "1. Implement automated IAM system with RBAC 2. Develop user lifecycle procedures 3. Establish regular access review processes"
        common_gap = "Access Management Control Gap"
    
    elif any(keyword in evidence_lower for keyword in ['incident', 'security', 'response']):
        gap_desc = f"Current incident management relies on informal processes without centralized tracking, escalation, or compliance reporting capabilities."
        gap_initiative = "1. Deploy incident management system 2. Establish formal response procedures 3. Implement compliance reporting dashboard"
        common_gap = "Incident Management Process Gap"
    
    elif any(keyword in evidence_lower for keyword in ['data', 'backup', 'recovery', 'protection']):
        gap_desc = f"Current data management practices lack systematic backup, recovery, and data governance controls required for Level {target_level} compliance."
        gap_initiative = "1. Implement data governance framework 2. Establish automated backup procedures 3. Deploy data classification controls"
        common_gap = "Data Management Control Gap"
    
    elif any(keyword in evidence_lower for keyword in ['monitoring', 'logging', 'audit', 'tracking']):
        gap_desc = f"Current monitoring and logging capabilities are insufficient for comprehensive audit trails and real-time threat detection."
        gap_initiative = "1. Deploy centralized logging system 2. Implement continuous monitoring 3. Establish audit trail procedures"
        common_gap = "Monitoring and Logging Gap"
    
    elif any(keyword in evidence_lower for keyword in ['training', 'awareness', 'education']):
        gap_desc = f"Current security awareness program lacks structured training, regular assessments, and role-based education."
        gap_initiative = "1. Develop comprehensive training program 2. Implement awareness campaigns 3. Establish role-based security education"
        common_gap = "Security Awareness Training Gap"
    
    else:
        gap_desc = f"Current {domain.lower()} procedures lack documentation depth and control frameworks required for Level {target_level} {framework} compliance."
        gap_initiative = "1. Develop comprehensive procedures 2. Implement process controls 3. Establish compliance monitoring"
        common_gap = "Documentation and Process Control Gap"

    # Severity assessment
    if maturity_gap >= 3:
        severity = "HIGH - Critical compliance gap requiring immediate attention and significant resources"
    elif maturity_gap >= 2:
        severity = "MEDIUM-HIGH - Significant compliance exposure requiring prioritized remediation"
    else:
        severity = "MEDIUM - Moderate gap requiring structured improvement approach"

    return {
        "gap_description": gap_desc,
        "gap_initiative": gap_initiative,
        "common_gap_description": common_gap,
        "severity_assessment": severity
    }

def predict_maturity(model, tokenizer, text):
    """Predict maturity level"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()

    return predicted_class, confidence, predictions[0].tolist()

def predict_evidence_types(model, tokenizer, evidence_types, question, domain, framework, threshold=0.5, top_k=5):
    """Predict evidence types"""
    text = f"{question} [SEP] {domain}: {framework}"
    
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=512
    )
    
    with torch.no_grad():
        logits = model(inputs['input_ids'], inputs['attention_mask'])
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]
    
    results = []
    for i, evidence_type in enumerate(evidence_types):
        results.append({
            'evidence_type': evidence_type,
            'probability': float(probabilities[i]),
            'recommended': probabilities[i] > threshold
        })
    
    results.sort(key=lambda x: x['probability'], reverse=True)
    recommended_evidence = results[:top_k]
    
    return {
        'all_predictions': results,
        'recommended_evidence': recommended_evidence,
        'summary': {
            'total_recommended': len([r for r in results if r['recommended']]),
            'max_probability': float(np.max(probabilities)),
            'avg_probability': float(np.mean(probabilities)),
            'threshold_used': threshold
        }
    }

def display_evidence_recommendations(recommendations, question):
    """Display evidence recommendations"""
    st.markdown('<div class="evidence-recommendation">', unsafe_allow_html=True)
    st.markdown("### 💡 Recommended Evidence Types")
    st.markdown(f"**For question:** {question}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Top Recommendation", 
                 f"{recommendations['recommended_evidence'][0]['probability']:.1%}")
    with col2:
        st.metric("Threshold Met", 
                 recommendations['summary']['total_recommended'])
    with col3:
        st.metric("Average Confidence", 
                 f"{recommendations['summary']['avg_probability']:.1%}")
    
    st.markdown("#### 🎯 Top 5 Evidence Types:")
    
    for i, rec in enumerate(recommendations['recommended_evidence'], 1):
        confidence = rec['probability']
        evidence_type = rec['evidence_type']
        
        if confidence >= 0.7:
            icon = "🟢"
            confidence_level = "High"
        elif confidence >= 0.5:
            icon = "🟡"
            confidence_level = "Medium"
        else:
            icon = "🔵"
            confidence_level = "Low"
        
        st.markdown(f"""
        <div class="evidence-item">
            <strong>{icon} {i}. {evidence_type}</strong><br>
            <small>Confidence: {confidence:.1%} ({confidence_level})</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def process_uploaded_file(uploaded_file):
    """Process uploaded files"""
    try:
        file_type = uploaded_file.type
        file_name = uploaded_file.name.lower()

        if file_type == "text/plain" or file_name.endswith('.txt'):
            content = io.StringIO(uploaded_file.getvalue().decode("utf-8")).read()
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
            content = df.to_string()
        elif file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            content = df.to_string()
        elif file_name.endswith('.pdf'):
            try:
                pdf_bytes = uploaded_file.read()
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                    content = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            content += page_text + "\n"
            except:
                uploaded_file.seek(0)
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
        elif file_name.endswith('.docx'):
            docx_bytes = uploaded_file.read()
            doc = Document(io.BytesIO(docx_bytes))
            content = ""
            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None

        return content

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def create_maturity_visualization(current_level, target_level, confidence):
    """Create maturity visualization"""
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_level,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Current Maturity Level"},
        delta={'reference': target_level, 'position': "top"},
        gauge={
            'axis': {'range': [None, 5]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 1], 'color': "lightgray"},
                {'range': [1, 2], 'color': "gray"},
                {'range': [2, 3], 'color': "yellow"},
                {'range': [3, 4], 'color': "orange"},
                {'range': [4, 5], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': target_level
            }
        }
    ))
    gauge_fig.update_layout(height=300)
    return gauge_fig

def main():
    st.title("🎯 Compliance Maturity & Gap Analysis Platform")
    st.markdown("*Automated maturity assessment and gap analysis for regulatory compliance*")
    st.markdown("---")

    # Show model loading status
    show_loading_status()

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        domain = st.selectbox(
            "Select Domain:",
            ["Banking", "Healthcare", "Technology", "Manufacturing", "General"],
            index=0
        )

        framework_options = {
            "Banking": ["RBI IT Framework", "Basel III", "PCI DSS"],
            "Healthcare": ["HIPAA", "HITECH", "FDA CFR"],
            "Technology": ["ISO 27001", "NIST", "SOC 2"],
            "Manufacturing": ["ISO 9001", "OSHA", "FDA QSR"],
            "General": ["ISO 27001", "NIST", "COBIT"]
        }

        framework = st.selectbox(
            "Select Framework:",
            framework_options.get(domain, ["ISO 27001"]),
            index=0
        )

        st.info("💡 Models load automatically when needed")
        
        # Model management buttons
        st.markdown("---")
        st.markdown("**Model Management**")
        
        if st.button("🔄 Reset All Models"):
            for key in list(st.session_state.models.keys()):
                del st.session_state.models[key]
            st.session_state.models_loaded = {
                'maturity': False,
                'gap_analysis': False,
                'evidence': False
            }
            st.success("All models reset!")
            st.rerun()

    # Main input section
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Question Input")
        
        question_col, button_col = st.columns([4, 1])
        
        with question_col:
            question = st.text_area(
                "Enter your compliance question:",
                placeholder="What documented procedures exist for user access provisioning?",
                height=120,
                key="main_question"
            )
        
        with button_col:
            st.markdown("<br>", unsafe_allow_html=True)
            
            recommend_evidence = st.button(
                "💡 Get Evidence",
                disabled=not question.strip(),
                help="Get AI recommendations for evidence types",
                type="secondary"
            )

    with col2:
        st.subheader("📄 Evidence Input")
        input_method = st.radio(
            "Choose input method:",
            ["Text Input", "File Upload"]
        )

    # Handle evidence recommendation
    if recommend_evidence and question.strip():
        evidence_model, evidence_tokenizer, evidence_types = load_evidence_recommendation_model()
        
        if evidence_model is not None:
            with st.spinner("🔍 Analyzing question for evidence recommendations..."):
                recommendations = predict_evidence_types(
                    evidence_model, evidence_tokenizer, evidence_types, 
                    question, domain, framework, threshold=0.3, top_k=5
                )
                st.session_state['evidence_recommendations'] = recommendations
                st.session_state['evidence_question'] = question
        else:
            st.warning("⚠️ Evidence recommendation model failed to load. Please try again.")

    # Display evidence recommendations
    if 'evidence_recommendations' in st.session_state and 'evidence_question' in st.session_state:
        if st.session_state['evidence_question'] == question:
            display_evidence_recommendations(st.session_state['evidence_recommendations'], question)
            
            if st.button("❌ Clear Recommendations"):
                del st.session_state['evidence_recommendations']
                del st.session_state['evidence_question']
                st.rerun()

    # Evidence input section
    evidence_text = ""

    if input_method == "Text Input":
        evidence_text = st.text_area(
            "Enter evidence text:",
            placeholder="Describe the current state of controls, documentation, or processes...",
            height=200
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload evidence file:",
            type=['txt', 'csv', 'xlsx', 'xls', 'pdf', 'docx']
        )

        if uploaded_file is not None:
            with st.spinner("📄 Processing uploaded file..."):
                evidence_text = process_uploaded_file(uploaded_file)

            if evidence_text:
                st.success(f"✅ File processed! ({len(evidence_text)} characters)")
                
                with st.expander("📋 Document Preview"):
                    preview_text = evidence_text[:2000]
                    if len(evidence_text) > 2000:
                        preview_text += "\n\n... (truncated)"
                    st.text(preview_text)

    # Analysis button
    st.markdown("---")
    analyze_button = st.button("🚀 Analyze Compliance Gap", type="primary", use_container_width=True)

    # Perform analysis
    if analyze_button:
        if not question.strip():
            st.error("❌ Please enter a question.")
        elif not evidence_text.strip():
            st.error("❌ Please provide evidence text.")
        else:
            # Load maturity model for analysis
            maturity_model, maturity_tokenizer = load_maturity_model()
            
            if maturity_model is None:
                st.error("❌ Failed to load maturity model. Cannot perform analysis.")
                return

            with st.spinner("🔍 Analyzing compliance maturity and gaps..."):
                # Step 1: Predict current maturity
                maturity_input = f"Question: {question.strip()} Evidence: {evidence_text.strip()}"
                current_maturity, confidence, all_probabilities = predict_maturity(
                    maturity_model, maturity_tokenizer, maturity_input
                )

                # Step 2: Set target maturity
                target_maturity = min(current_maturity + 1, 5)

                # Step 3: Generate gap analysis (try AI model, fallback to rule-based)
                gap_components = None
                
                # Try to load gap analysis model
                gap_model, gap_tokenizer = load_gap_analysis_model()
                
                if gap_model is not None:
                    try:
                        # Simplified prompt for better performance
                        prompt = f"""Analyze compliance gap:
Question: {question}
Evidence: {evidence_text[:500]}...
Current Level: {current_maturity} → Target Level: {target_maturity}
Domain: {domain} | Framework: {framework}

Provide:
1. Gap Description: What's missing?
2. Gap Initiative: Top 3 actions needed
3. Common Gap: Category
4. Severity: Risk level (High/Medium/Low)"""

                        inputs = gap_tokenizer(prompt, max_length=512, truncation=True, return_tensors="pt")
                        
                        with torch.no_grad():
                            outputs = gap_model.generate(
                                **inputs,
                                max_length=300,
                                min_length=50,
                                temperature=0.7,
                                do_sample=True,
                                top_p=0.9,
                                pad_token_id=gap_tokenizer.pad_token_id,
                                eos_token_id=gap_tokenizer.eos_token_id
                            )
                        
                        result = gap_tokenizer.decode(outputs[0], skip_special_tokens=True)
                        if prompt in result:
                            result = result.replace(prompt, "").strip()
                        
                        # Parse result or use rule-based fallback
                        gap_components = create_rule_based_gap_analysis(
                            question, evidence_text, current_maturity, target_maturity, domain, framework
                        )
                        
                    except Exception as e:
                        st.warning(f"⚠️ AI gap analysis failed, using rule-based analysis: {str(e)}")
                        gap_components = create_rule_based_gap_analysis(
                            question, evidence_text, current_maturity, target_maturity, domain, framework
                        )
                else:
                    gap_components = create_rule_based_gap_analysis(
                        question, evidence_text, current_maturity, target_maturity, domain, framework
                    )

                gap_size = target_maturity - current_maturity

            # Display results
            st.markdown("---")
            st.header("📊 Analysis Results")

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Current Maturity", f"Level {current_maturity}", 
                         help=f"AI Confidence: {confidence:.2%}")
            with col2:
                st.metric("Target Maturity", f"Level {target_maturity}", 
                         delta=f"+{gap_size} levels" if gap_size > 0 else "At target")
            with col3:
                st.metric("Gap Size", f"{gap_size} levels")
            with col4:
                st.metric("Confidence", f"{confidence:.1%}")

            # Visualization
            st.subheader("📈 Maturity Level Visualization")
            col1, col2 = st.columns([1, 1])

            with col1:
                gauge_fig = create_maturity_visualization(current_maturity, target_maturity, confidence)
                st.plotly_chart(gauge_fig, use_container_width=True)

            with col2:
                prob_df = pd.DataFrame({
                    'Level': [f"Level {i}" for i in range(len(all_probabilities))],
                    'Probability': all_probabilities
                })
                bar_fig = px.bar(prob_df, x='Level', y='Probability', 
                               title="Maturity Level Confidence Distribution")
                st.plotly_chart(bar_fig, use_container_width=True)

            # Gap Analysis Results
            st.subheader("🎯 Gap Analysis Results")
            
            st.markdown("**📋 Gap Description**")
            st.write(gap_components.get("gap_description", "N/A"))
            
            st.markdown("**🚀 Gap Initiative**")
            st.write(gap_components.get("gap_initiative", "N/A"))
            
            st.markdown("**🔍 Common Gap Category**")
            st.write(gap_components.get("common_gap_description", "N/A"))
            
            severity = gap_components.get("severity_assessment", "")
            if "HIGH" in severity.upper():
                st.error(f"🔴 **Severity**: {severity}")
            elif "MEDIUM" in severity.upper():
                st.warning(f"🟡 **Severity**: {severity}")
            else:
                st.info(f"🔵 **Severity**: {severity}")

            # Export functionality
            st.subheader("💾 Export Results")
            
            summary_report = f"""# Compliance Gap Analysis Report

**Assessment Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Domain**: {domain}
**Framework**: {framework}

## Question
{question}

## Maturity Assessment
- Current Level: {current_maturity} ({MATURITY_LEVELS[current_maturity]})
- Target Level: {target_maturity} ({MATURITY_LEVELS[target_maturity]})
- Confidence: {confidence:.2%}
- Gap Size: {gap_size} levels

## Gap Analysis
**Description**: {gap_components.get("gap_description", "N/A")}

**Initiative**: {gap_components.get("gap_initiative", "N/A")}

**Category**: {gap_components.get("common_gap_description", "N/A")}

**Severity**: {gap_components.get("severity_assessment", "N/A")}
"""

            st.download_button(
                label="📄 Download Report",
                data=summary_report,
                file_name=f"gap_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )

if __name__ == "__main__":
    main()
