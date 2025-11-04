import spacy
from transformers import pipeline
from typing import Dict, List
import re
from mistralai import Mistral
from dotenv import load_dotenv
import os
import json
import difflib

load_dotenv()

class MultilingualNLPAnalyzer:
    def __init__(self, use_mistral=True):
        # Load spaCy model
        self.nlp = spacy.load("en_core_web_lg")

        # Initialize sentiment analyzer
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        self.use_mistral = use_mistral

        if self.use_mistral:
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY is not set in environment variables. Please add it to your .env file.")
            
            # Debug: Check if key is loaded (show first/last 4 chars only)
            print(f"API Key loaded: {api_key[:4]}...{api_key[-4:]}")
            
            self.mistral_client = Mistral(api_key=api_key)
            self.mistral_model = "mistral-large-latest"
        else:
            raise ValueError("Mistral is required for this analyzer")

    def detect_language(self, text: str) -> str:
        """Detect if text is in Kannada or English."""
        # Check for Kannada Unicode range (0C80-0CFF)
        kannada_chars = re.findall(r'[\u0C80-\u0CFF]', text)
        
        # If more than 10% of characters are Kannada, consider it Kannada text
        if len(kannada_chars) > len(text) * 0.1:
            return "kannada"
        return "english"

    def translate_kannada_to_english_with_mistral(self, text: str) -> Dict[str, str]:
        """
        Translate Kannada text to English using Mistral API.
        Creates sentence-level mappings for better highlighting.
        """
        print("Translating Kannada text to English using Mistral...")
        
        try:
            # Split Kannada text into sentences/paragraphs for better mapping
            # Kannada uses । (devanagari danda) or . as sentence terminators
            kannada_sentences = []
            
            # First split by paragraphs
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                    
                # Then split by Kannada sentence terminators
                # Look for: । (devanagari danda), . (period), or newlines
                sentences = re.split(r'[।\.\n]+', para)
                for sent in sentences:
                    sent = sent.strip()
                    if len(sent) > 20:  # Only meaningful sentences
                        kannada_sentences.append(sent)
            
            print(f"Translating {len(kannada_sentences)} Kannada sentences...")
            
            # Translate in small batches for better mapping
            batch_size = 5
            all_mappings = {}
            reverse_mappings = {}
            translated_sentences = []
            
            for i in range(0, len(kannada_sentences), batch_size):
                batch = kannada_sentences[i:i+batch_size]
                batch_text = "\n".join([f"{j+1}. {sent}" for j, sent in enumerate(batch)])
                
                print(f"Translating batch {i//batch_size + 1}/{(len(kannada_sentences)-1)//batch_size + 1}...")
                
                prompt = f"""Translate these Kannada sentences to English. Keep the numbering.
Preserve legal terminology. Provide ONLY the translations, one per line with the same numbers.

{batch_text}

English translations:"""

                response = self.mistral_client.chat.complete(
                    model=self.mistral_model,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                translated_batch = response.choices[0].message.content.strip()
                
                # Parse the numbered translations
                translated_lines = translated_batch.split('\n')
                for j, line in enumerate(translated_lines):
                    # Remove numbering if present
                    line = re.sub(r'^\d+\.\s*', '', line).strip()
                    if line and j < len(batch):
                        original_sent = batch[j]
                        all_mappings[original_sent] = line
                        reverse_mappings[line] = original_sent
                        translated_sentences.append(line)
            
            full_translation = "\n".join(translated_sentences)
            print(f"Translation completed: {len(all_mappings)} sentence pairs created")
            
            return {
                "original": text,
                "translated": full_translation,
                "mappings": all_mappings,
                "reverse_mappings": reverse_mappings,
                "kannada_sentences": kannada_sentences,  # Store original sentences
                "language": "kannada"
            }
            
        except Exception as e:
            print(f"Translation error: {e}")
            raise Exception(f"Failed to translate Kannada text: {str(e)}")

    def _normalize_text(self, text: str) -> str:
        """Normalize text for better matching"""
        text = re.sub(r'\s+', ' ', text.strip())
        text = text.replace('"', '').replace('"', '').replace('"', '')
        return text

    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences using spaCy with better filtering"""
        doc = self.nlp(text)
        sentences = []
        
        spacy_sentences = []
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            if len(sentence_text) > 15:
                spacy_sentences.append(sentence_text)
        
        paragraph_sentences = []
        for paragraph in text.split('\n\n'):
            para = paragraph.strip()
            if len(para) > 20:
                if len(para) > 200:
                    sub_sentences = re.split(r'[.!?]+\s+', para)
                    for sub_sent in sub_sentences:
                        sub_sent = sub_sent.strip()
                        if len(sub_sent) > 15:
                            if not sub_sent.endswith(('.', '!', '?')):
                                sub_sent += '.'
                            paragraph_sentences.append(sub_sent)
                else:
                    if not para.endswith(('.', '!', '?')):
                        para += '.'
                    paragraph_sentences.append(para)
        
        all_sentences = spacy_sentences + paragraph_sentences
        
        seen = set()
        for sentence in all_sentences:
            normalized = self._normalize_text(sentence)
            if normalized not in seen and len(normalized) > 15:
                sentences.append(sentence)
                seen.add(normalized)
        
        print(f"Extracted {len(sentences)} unique sentences")
        return sentences

    def _find_best_match(self, target_sentence: str, available_sentences: List[str]) -> str:
        """Find the best matching sentence in the document"""
        target_normalized = self._normalize_text(target_sentence)
        
        # Exact match (case insensitive)
        for sentence in available_sentences:
            if target_normalized.lower() == sentence.lower():
                return sentence
        
        # Match ignoring punctuation
        target_clean = re.sub(r'[^\w\s]', '', target_normalized.lower())
        for sentence in available_sentences:
            sentence_clean = re.sub(r'[^\w\s]', '', sentence.lower())
            if target_clean == sentence_clean:
                return sentence
        
        # Substring matching
        for sentence in available_sentences:
            if (target_normalized.lower() in sentence.lower() and len(target_normalized) > 30) or \
               (sentence.lower() in target_normalized.lower() and len(sentence) > 30):
                return sentence
        
        # Fuzzy matching
        best_match = None
        best_ratio = 0.6
        
        for sentence in available_sentences:
            ratios = [
                difflib.SequenceMatcher(None, target_normalized.lower(), sentence.lower()).ratio(),
                difflib.SequenceMatcher(None, target_clean, re.sub(r'[^\w\s]', '', sentence.lower())).ratio()
            ]
            
            max_ratio = max(ratios)
            if max_ratio > best_ratio:
                best_ratio = max_ratio
                best_match = sentence
        
        return best_match

    def _find_original_kannada_text(self, english_sentence: str, translation_info: Dict) -> str:
        """
        Find the original Kannada text for a translated English sentence.
        Uses improved mapping from sentence-level translation.
        """
        if not translation_info or 'reverse_mappings' not in translation_info:
            return None
        
        reverse_mappings = translation_info['reverse_mappings']
        
        # Exact match first
        if english_sentence in reverse_mappings:
            return reverse_mappings[english_sentence]
        
        # Normalized match
        english_normalized = self._normalize_text(english_sentence)
        for en_sent, kn_sent in reverse_mappings.items():
            if self._normalize_text(en_sent).lower() == english_normalized.lower():
                return kn_sent
        
        # Check if English sentence is contained in any translated sentence
        for translated_chunk, kannada_chunk in reverse_mappings.items():
            if english_sentence.lower() in translated_chunk.lower():
                return kannada_chunk
        
        # Fuzzy matching with higher threshold for sentence-level
        best_match_kannada = None
        best_ratio = 0.75  # Higher threshold for sentence-level matching
        
        for translated_sent, kannada_sent in reverse_mappings.items():
            ratio = difflib.SequenceMatcher(
                None, 
                english_normalized.lower(), 
                self._normalize_text(translated_sent).lower()
            ).ratio()
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_match_kannada = kannada_sent
        
        if best_match_kannada:
            print(f"  ✓ Found Kannada mapping with {best_ratio:.2%} confidence")
        
        return best_match_kannada

    def analyze_text(self, text: str, language: str = None) -> Dict:
        """
        Analyze lease agreement text in English or Kannada.
        Uses Mistral for Kannada translation with sentence-level mapping.
        """
        original_text = text
        original_language = language if language else self.detect_language(text)
        
        translation_info = None
        
        # If Kannada, translate to English using Mistral
        if original_language == "kannada":
            print("Detected Kannada text - translating with Mistral...")
            translation_result = self.translate_kannada_to_english_with_mistral(text)
            text = translation_result["translated"]
            translation_info = translation_result
            print("Analysis will be performed on translated English text")
        
        # Analyze the (possibly translated) English text
        doc = self.nlp(text)
        document_sentences = self._extract_sentences(text)

        results = {
            "high_severity": [],
            "medium_severity": [],
            "low_severity": [],
            "sentiment": None,
            "entities": [],
            "document_sentences": document_sentences,
            "severity_explanations": {},
            "original_language": original_language,
            "translation_info": translation_info
        }

        # Extract named entities
        for ent in doc.ents:
            results["entities"].append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })

        # Sentiment analysis
        sentiment_result = self.sentiment_analyzer(text[:512])[0]
        results["sentiment"] = sentiment_result

        try:
            language_note = ""
            if original_language == "kannada":
                language_note = "\n\nNOTE: This document was originally in Kannada and has been translated to English for analysis."
            
            prompt = f"""
You are a legal AI assistant. Analyze this lease agreement and classify ONLY the most important sentences by severity level. For each sentence you classify, explain WHY it belongs to that category.{language_note}

Categories:
- high_severity: Critical risks like eviction threats, large penalties, liability issues, termination clauses, breach consequences
- medium_severity: Important obligations like rent payment terms, maintenance duties, notice requirements, access rights
- low_severity: General information like parties involved, property description, basic definitions

Return your analysis in this EXACT JSON format:
{{
  "high_severity": [
    {{
      "text": "exact sentence from document",
      "reason": "explanation why this is high severity"
    }}
  ],
  "medium_severity": [
    {{
      "text": "exact sentence from document", 
      "reason": "explanation why this is medium severity"
    }}
  ],
  "low_severity": [
    {{
      "text": "exact sentence from document",
      "reason": "explanation why this is low severity"  
    }}
  ]
}}

IMPORTANT: Use the exact sentences from the document. Do not paraphrase or modify them.

Document text:
\"\"\"{text}\"\"\"
"""

            chat_response = self.mistral_client.chat.complete(
                model=self.mistral_model,
                messages=[{"role": "user", "content": prompt}]
            )

            content = chat_response.choices[0].message.content.strip()
            print("Raw Mistral response received")

            # Clean the response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1]

            content = content.strip()

            parsed = json.loads(content)
            
            # Process each severity level
            for severity_level in ['high_severity', 'medium_severity', 'low_severity']:
                if severity_level in parsed:
                    for item in parsed[severity_level]:
                        sentence_text = item.get('text', '').strip()
                        reason = item.get('reason', 'No explanation provided')
                        
                        if sentence_text:
                            matched_sentence = self._find_best_match(sentence_text, document_sentences)
                            
                            if matched_sentence:
                                clause_data = {
                                    "text": matched_sentence,
                                    "original_text": sentence_text,
                                    "reason": reason
                                }
                                
                                # If Kannada, find and store the original Kannada text
                                if translation_info:
                                    original_kannada = self._find_original_kannada_text(
                                        matched_sentence, 
                                        translation_info
                                    )
                                    if original_kannada:
                                        clause_data["kannada_text"] = original_kannada
                                        print(f"  ✓ Mapped to Kannada: {original_kannada[:50]}...")
                                    else:
                                        print(f"  ⚠ Could not find Kannada mapping for: {matched_sentence[:50]}...")
                                
                                results[severity_level].append(clause_data)
                                results["severity_explanations"][matched_sentence] = {
                                    "severity": severity_level,
                                    "reason": reason
                                }
                            else:
                                clause_data = {
                                    "text": sentence_text,
                                    "original_text": sentence_text,
                                    "reason": reason,
                                    "match_failed": True
                                }
                                
                                results[severity_level].append(clause_data)
                                results["severity_explanations"][sentence_text] = {
                                    "severity": severity_level,
                                    "reason": reason
                                }

        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing error: {e}")
            self._fallback_classification(document_sentences, results)
        except Exception as e:
            print(f"Mistral API failed: {e}")
            self._fallback_classification(document_sentences, results)

        return results

    def _fallback_classification(self, sentences: List[str], results: Dict):
        """Fallback classification when Mistral fails"""
        high_keywords = ['eviction', 'penalty', 'breach', 'terminate', 'default', 'forfeit', 'liable']
        medium_keywords = ['rent', 'payment', 'maintenance', 'repair', 'notice', 'access', 'inspect']
        
        for sentence in sentences[:10]:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in high_keywords):
                results['high_severity'].append({
                    "text": sentence,
                    "reason": "Contains high-risk keywords"
                })
            elif any(keyword in sentence_lower for keyword in medium_keywords):
                results['medium_severity'].append({
                    "text": sentence,
                    "reason": "Contains obligation-related keywords"
                })
            else:
                results['low_severity'].append({
                    "text": sentence,
                    "reason": "General information"
                })

    def calculate_favorability_score(self, analysis_results: Dict) -> float:
        """Calculate a balanced favorability score (0-10)."""
        base_score = 7.0
        
        high_count = len(analysis_results['high_severity'])
        medium_count = len(analysis_results['medium_severity'])
        low_count = len(analysis_results['low_severity'])
        
        score = base_score
        score -= high_count * 0.3
        score -= medium_count * 0.1
        score += low_count * 0.05
        
        sentiment = analysis_results['sentiment']
        if sentiment['label'] == 'POSITIVE':
            score += 0.5
        elif sentiment['label'] == 'NEGATIVE':
            score -= 0.5
        
        total_clauses = high_count + medium_count + low_count
        if total_clauses > 5:
            score += 0.3
        
        return max(1.0, min(10.0, round(score, 1)))

    def generate_summary(self, analysis_results: Dict, full_text: str) -> str:
        """Generate an enhanced summary with language awareness."""
        try:
            severity_details = []
            
            for severity in ['high_severity', 'medium_severity', 'low_severity']:
                items = analysis_results.get(severity, [])
                if items:
                    severity_details.append(f"\n{severity.replace('_', ' ').title()} ({len(items)} clauses):")
                    for item in items[:3]:
                        text = item.get('text', '')[:100] + "..." if len(item.get('text', '')) > 100 else item.get('text', '')
                        reason = item.get('reason', 'No explanation')
                        severity_details.append(f"• {text}")
                        severity_details.append(f"  Reason: {reason}")

            severity_summary = "\n".join(severity_details)
            
            language_note = ""
            if analysis_results.get('original_language') == 'kannada':
                language_note = "\n\nNOTE: This document was originally in Kannada and analyzed after translation to English."

            prompt = f"""
Create a comprehensive summary of this lease agreement analysis. Include:

1. Overall Assessment: Brief overview of the document's tenant-friendliness
2. Key Findings: Major risks and important obligations
3. Severity Breakdown: Explain the classification of clauses
4. Recommendations: Advice for the tenant{language_note}

Analysis Results:
- High Severity: {len(analysis_results['high_severity'])} clauses
- Medium Severity: {len(analysis_results['medium_severity'])} clauses  
- Low Severity: {len(analysis_results['low_severity'])} clauses
- Overall Sentiment: {analysis_results['sentiment']['label']}

Severity Details:{severity_summary}

Document Excerpt:
\"\"\"{full_text[:1000]}...\"\"\"
"""

            chat_response = self.mistral_client.chat.complete(
                model=self.mistral_model,
                messages=[{"role": "user", "content": prompt}]
            )
            return chat_response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Mistral API failed for summary: {e}")
            return self._generate_fallback_summary(analysis_results)

    def _generate_fallback_summary(self, analysis_results: Dict) -> str:
        """Generate fallback summary without Mistral"""
        sentiment = analysis_results['sentiment']
        high_count = len(analysis_results['high_severity'])
        medium_count = len(analysis_results['medium_severity']) 
        low_count = len(analysis_results['low_severity'])
        
        summary_parts = [
            f"**Overall Assessment**: This lease agreement has a {sentiment['label'].lower()} tone overall.",
            f"**Clause Analysis**: Found {high_count} high-severity, {medium_count} medium-severity, and {low_count} low-severity clauses.",
        ]
        
        if analysis_results.get('original_language') == 'kannada':
            summary_parts.insert(0, "**Language**: This document was originally in Kannada and has been analyzed after translation.")
        
        if high_count > 0:
            summary_parts.append(f"**Key Concerns**: The {high_count} high-severity clauses may pose significant risks and should be carefully reviewed.")
            
        if medium_count > 0:
            summary_parts.append(f"**Important Obligations**: The {medium_count} medium-severity clauses outline key responsibilities and terms.")
            
        return " ".join(summary_parts)