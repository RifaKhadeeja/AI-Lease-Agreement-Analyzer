import streamlit as st
import os
from src.document_processor import DocumentProcessor
from src.nlp_analyzer import MultilingualNLPAnalyzer  # Changed import
from src.document_highlighter import DocumentHighlighter
import tempfile


st.set_page_config(
    page_title="Lease Agreement Analyzer",
    page_icon="📄",
    layout="wide"
)

# Initialize components
document_processor = DocumentProcessor()
nlp_analyzer = MultilingualNLPAnalyzer(use_mistral=True)  # Changed class name
document_highlighter = DocumentHighlighter()

def display_clause_details(analysis_results):
    """Display detailed clause information with explanations"""
    
    severity_colors = {
        'high_severity': '🔴',
        'medium_severity': '🟡', 
        'low_severity': '🔵'
    }
    
    severity_labels = {
        'high_severity': 'High Risk Clauses',
        'medium_severity': 'Medium Risk Clauses',
        'low_severity': 'General Information'
    }
    
    for severity in ['high_severity', 'medium_severity', 'low_severity']:
        clauses = analysis_results.get(severity, [])
        if clauses:
            st.markdown(f"### {severity_colors[severity]} {severity_labels[severity]} ({len(clauses)})")
            
            with st.expander(f"View {severity_labels[severity]}", expanded=(severity == 'high_severity')):
                for i, clause in enumerate(clauses, 1):
                    clause_text = clause.get('text', '')
                    reason = clause.get('reason', 'No explanation provided')
                    
                    # Show if this was translated from Kannada
                    if clause.get('kannada_text'):
                        st.markdown(f"**{i}.** {clause_text} 🌐")
                        st.caption("📝 *Originally in Kannada*")
                    else:
                        st.markdown(f"**{i}.** {clause_text}")
                    
                    st.markdown(f"*Reason:* {reason}")
                    st.markdown("---")

def main():
    st.title("📄 AI Lease Agreement Analyzer")
    st.markdown("""
    This tool analyzes lease agreements using AI to identify important clauses, 
    highlight them by severity, and provide a favorability score for renters.
    
    **Supported Languages:** 🇬🇧 English | 🇮🇳 ಕನ್ನಡ Kannada
    
    **Color Legend:**
    - 🔴 **Red**: High risk clauses (penalties, eviction terms, major liabilities)
    - 🟡 **Yellow**: Medium risk clauses (rent terms, maintenance, notice periods)
    - 🔵 **Blue**: General information (definitions, property details, basic terms)
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your lease agreement (English or Kannada)",
        type=['pdf', 'docx', 'txt'],
        help="Supported formats: PDF, DOCX, TXT | Supported languages: English, Kannada"
    )
    
    if uploaded_file is not None:
        # Create temporary file with same extension as uploaded
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
            
        try:
            # Process document
            with st.spinner("Processing document..."):
                text, file_ext = document_processor.process_document(tmp_file_path)
                st.success(f"✅ Document processed successfully ({len(text)} characters extracted)")
                
            # Analyze text with language detection
            with st.spinner("Analyzing content with AI (detecting language)..."):
                analysis_results = nlp_analyzer.analyze_text(text)
                
                # Show language detection result
                detected_language = analysis_results.get('original_language', 'english')
                if detected_language == 'kannada':
                    st.info("🌐 **Kannada document detected** - Translating to English for analysis...")
                    
                    # Show translation info if available
                    if analysis_results.get('translation_info'):
                        with st.expander("📖 View Translation Details", expanded=False):
                            translation_info = analysis_results['translation_info']
                            st.markdown("**Original Text (Kannada):**")
                            st.text(translation_info['original'][:500] + "..." if len(translation_info['original']) > 500 else translation_info['original'])
                            st.markdown("**Translated Text (English):**")
                            st.text(translation_info['translated'][:500] + "..." if len(translation_info['translated']) > 500 else translation_info['translated'])
                else:
                    st.info("🌐 **English document detected**")
                
            # Calculate favorability score
            score = nlp_analyzer.calculate_favorability_score(analysis_results)
            
            # Generate summary
            with st.spinner("Generating detailed summary..."):
                summary = nlp_analyzer.generate_summary(analysis_results, full_text=text)
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["📊 Overview", "📋 Detailed Analysis", "📄 Highlighted Document"])
            
            with tab1:
                st.subheader("Analysis Overview")
                
                # Show original language
                language_emoji = "🇮🇳" if detected_language == 'kannada' else "🇬🇧"
                language_name = "Kannada (ಕನ್ನಡ)" if detected_language == 'kannada' else "English"
                st.markdown(f"**Document Language:** {language_emoji} {language_name}")
                
                # Score display with interpretation
                score_color = "🟢" if score >= 7 else "🟡" if score >= 5 else "🔴"
                score_interpretation = (
                    "Tenant-friendly" if score >= 7 
                    else "Moderate" if score >= 5 
                    else "Tenant should review carefully"
                )
                
                st.markdown(f"""
                ### {score_color} Favorability Score: {score}/10
                **Interpretation:** {score_interpretation}
                """)
                
                # Clause statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "🔴 High Risk",
                        len(analysis_results['high_severity']),
                        help="Clauses with potential significant impact (penalties, eviction, liability)"
                    )
                    
                with col2:
                    st.metric(
                        "🟡 Medium Risk", 
                        len(analysis_results['medium_severity']),
                        help="Important obligations (rent, maintenance, notices)"
                    )
                    
                with col3:
                    st.metric(
                        "🔵 General Info",
                        len(analysis_results['low_severity']),
                        help="Basic information (definitions, property details)"
                    )
                
                # Summary
                st.markdown("### 📝 AI Summary")
                st.write(summary)
                
                # Sentiment analysis
                sentiment = analysis_results['sentiment']
                sentiment_emoji = "😊" if sentiment['label'] == 'POSITIVE' else "😐" if sentiment['label'] == 'NEUTRAL' else "😟"
                st.markdown(f"**Document Tone:** {sentiment_emoji} {sentiment['label']} (Confidence: {sentiment['score']:.2%})")
            
            with tab2:
                st.subheader("Detailed Clause Analysis")
                
                if detected_language == 'kannada':
                    st.info("💡 **Note:** Clauses below are shown in English (translated from Kannada) for analysis. The highlighted PDF will show the original Kannada text.")
                
                display_clause_details(analysis_results)
                
                # Key entities
                if analysis_results.get('entities'):
                    st.markdown("### 🏷️ Key Entities Found")
                    entities_by_type = {}
                    for entity in analysis_results['entities']:
                        label = entity['label']
                        if label not in entities_by_type:
                            entities_by_type[label] = []
                        entities_by_type[label].append(entity['text'])
                    
                    for label, texts in entities_by_type.items():
                        if label in ['PERSON', 'ORG', 'DATE', 'MONEY', 'GPE']:
                            unique_texts = list(set(texts))[:5]  # Show up to 5 unique entities
                            st.write(f"**{label}**: {', '.join(unique_texts)}")
            
            with tab3:
                st.subheader("Highlighted Document")
                
                if detected_language == 'kannada':
                    st.info("📌 **Highlighting Note:** The PDF will be highlighted in the original Kannada text, not the English translation.")
                
                # Highlight document
                with st.spinner("Generating highlighted document..."):
                    output_path = os.path.join(
                        tempfile.gettempdir(),
                        f"highlighted_{os.path.splitext(os.path.basename(uploaded_file.name))[0]}.pdf"
                    )
                    
                    try:
                        # Get highlighting statistics
                        highlight_stats = document_highlighter.highlight_document(
                            tmp_file_path,
                            analysis_results,
                            output_path
                        )
                        
                        # Ensure highlight_stats is a dictionary
                        if highlight_stats is None or not isinstance(highlight_stats, dict):
                            st.warning("⚠️ Highlighting process encountered issues. Creating basic document copy.")
                            highlight_stats = {
                                'high_severity': {'expected': len(analysis_results.get('high_severity', [])), 'found': 0, 'missed': []},
                                'medium_severity': {'expected': len(analysis_results.get('medium_severity', [])), 'found': 0, 'missed': []},
                                'low_severity': {'expected': len(analysis_results.get('low_severity', [])), 'found': 0, 'missed': []}
                            }
                    except Exception as e:
                        st.error(f"Highlighting error: {str(e)}")
                        highlight_stats = {
                            'high_severity': {'expected': len(analysis_results.get('high_severity', [])), 'found': 0, 'missed': []},
                            'medium_severity': {'expected': len(analysis_results.get('medium_severity', [])), 'found': 0, 'missed': []},
                            'low_severity': {'expected': len(analysis_results.get('low_severity', [])), 'found': 0, 'missed': []}
                        }
                
                # Display highlighting statistics
                st.markdown("#### Highlighting Results")
                
                try:
                    total_expected = sum(stats.get('expected', 0) for stats in highlight_stats.values())
                    total_found = sum(stats.get('found', 0) for stats in highlight_stats.values())
                    
                    if total_expected > 0:
                        success_rate = (total_found / total_expected) * 100
                        st.metric("Highlighting Success Rate", f"{success_rate:.1f}%", f"{total_found}/{total_expected} clauses")
                        
                        # Special note for Kannada documents with low success rate
                        if detected_language == 'kannada' and success_rate < 50:
                            st.warning("⚠️ **Translation Note:** Some clauses may not highlight perfectly due to translation differences between the analyzed English text and original Kannada document.")
                    else:
                        st.info("No clauses to highlight")
                    
                    # Show detailed stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        stats = highlight_stats.get('high_severity', {'found': 0, 'expected': 0})
                        st.metric("🔴 High Risk Highlighted", f"{stats.get('found', 0)}/{stats.get('expected', 0)}")
                    with col2:
                        stats = highlight_stats.get('medium_severity', {'found': 0, 'expected': 0})
                        st.metric("🟡 Medium Risk Highlighted", f"{stats.get('found', 0)}/{stats.get('expected', 0)}")
                    with col3:
                        stats = highlight_stats.get('low_severity', {'found': 0, 'expected': 0})
                        st.metric("🔵 General Info Highlighted", f"{stats.get('found', 0)}/{stats.get('expected', 0)}")
                    
                    # Show missed clauses if any
                    missed_clauses = []
                    for severity, stats in highlight_stats.items():
                        if isinstance(stats, dict) and 'missed' in stats:
                            missed_clauses.extend(stats.get('missed', []))
                            
                except Exception as e:
                    st.error(f"Error displaying highlighting statistics: {str(e)}")
                    st.info("Document processing completed, but statistics unavailable.")
                
                if missed_clauses:
                    with st.expander("⚠️ Clauses that couldn't be highlighted", expanded=False):
                        if detected_language == 'kannada':
                            st.warning("Some clauses were identified by AI but couldn't be found in the document for highlighting. This is common with translated documents due to text formatting and translation differences.")
                        else:
                            st.warning("Some clauses were identified by AI but couldn't be found in the document for highlighting. This may be due to text formatting differences.")
                        for i, missed_clause in enumerate(missed_clauses[:5], 1):  # Show up to 5
                            st.write(f"{i}. {missed_clause}")
                
                # Download button for highlighted document
                if os.path.exists(output_path):
                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Highlighted Document (PDF)",
                            data=file,
                            file_name=f"highlighted_{os.path.splitext(uploaded_file.name)[0]}.pdf",
                            mime="application/pdf",
                            help="Download the PDF with color-coded highlights"
                        )
                else:
                    st.error("Could not generate highlighted document")
                
            # Clean up temporary files
            try:
                os.unlink(tmp_file_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)
            except:
                pass  # Ignore cleanup errors
                
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.error("Please check your document format and try again.")
            
            # Clean up on error
            try:
                os.unlink(tmp_file_path)
            except:
                pass

    else:
        # Show example when no file is uploaded
        st.markdown("""
        ### 📖 How it works:
        
        1. **Upload** your lease agreement (PDF, DOCX, or TXT) in English or Kannada
        2. **Language Detection** automatically identifies English or Kannada text
        3. **AI Analysis** extracts and classifies important clauses (translating if needed)
        4. **Color Highlighting** marks different risk levels in the original document
        5. **Favorability Score** gives you a tenant-friendly rating
        6. **Detailed Summary** explains why each clause matters
        
        ### 🌐 Multilingual Support:
        
        - **🇬🇧 English**: Full support with direct analysis
        - **🇮🇳 ಕನ್ನಡ Kannada**: Automatic translation to English for analysis, highlighting in original Kannada
        
        ### 🎯 What to look for:
        
        - **🔴 Red highlights**: Critical terms that could significantly impact you
        - **🟡 Yellow highlights**: Important obligations and responsibilities  
        - **🔵 Blue highlights**: General information and definitions
        
        The AI will explain why each clause was classified at its severity level.
        """)

if __name__ == "__main__":
    main()