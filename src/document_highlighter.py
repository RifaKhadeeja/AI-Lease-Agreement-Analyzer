import fitz  # PyMuPDF
from typing import Dict, List
import os
from docx2pdf import convert as docx2pdf_convert
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import re
import difflib

class DocumentHighlighter:
    def __init__(self):
        self.color_map = {
            'high_severity': (1, 0.2, 0.2),   # Red
            'medium_severity': (1, 1, 0.2),   # Yellow  
            'low_severity': (0.2, 0.2, 1)     # Blue
        }

    def highlight_document(self, file_path: str, analysis_results: Dict, output_path: str) -> Dict:
        """
        Convert any document to PDF first, then highlight important sentences.
        Supports multilingual documents (English, Kannada).
        Returns highlighting statistics.
        """
        # Initialize default stats in case of error
        default_stats = {
            'high_severity': {'expected': 0, 'found': 0, 'missed': []},
            'medium_severity': {'expected': 0, 'found': 0, 'missed': []},
            'low_severity': {'expected': 0, 'found': 0, 'missed': []}
        }
        
        try:
            file_ext = file_path.split('.')[-1].lower()
            temp_pdf = "temp_converted.pdf"

            if file_ext == "pdf":
                pdf_path = file_path
            elif file_ext == "docx":
                print("Converting DOCX to PDF...")
                docx2pdf_convert(file_path, temp_pdf)
                pdf_path = temp_pdf
            elif file_ext == "txt":
                print("Converting TXT to PDF...")
                self._txt_to_pdf(file_path, temp_pdf)
                pdf_path = temp_pdf
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

            print(f"Processing PDF: {pdf_path}")
            
            # Check if document is Kannada
            is_kannada = analysis_results.get('original_language') == 'kannada'
            if is_kannada:
                print("📌 Kannada document detected - will use original Kannada text for highlighting")
            
            stats = self._highlight_pdf(pdf_path, analysis_results, output_path)

            # Clean up temp file
            if os.path.exists(temp_pdf) and pdf_path == temp_pdf:
                try:
                    os.remove(temp_pdf)
                    print("Cleaned up temporary PDF")
                except Exception as e:
                    print(f"Warning: Could not remove temp file: {e}")
            
            return stats if stats is not None else default_stats
            
        except Exception as e:
            print(f"Error in highlight_document: {e}")
            # Try to create a basic PDF copy at least
            try:
                if os.path.exists(file_path) and file_path.endswith('.pdf'):
                    import shutil
                    shutil.copy2(file_path, output_path)
                    print("Created copy of original PDF")
            except Exception as copy_error:
                print(f"Could not create PDF copy: {copy_error}")
            
            return default_stats

    def _normalize_text_for_search(self, text: str) -> str:
        """Normalize text for better PDF searching"""
        # Handle common PDF text issues
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove quotes that might interfere
        text = text.replace('"', '').replace('"', '').replace('"', '')
        return text

    def _fuzzy_search_in_pdf(self, page, target_text: str, threshold: float = 0.8) -> List:
        """
        Search for text in PDF using fuzzy matching.
        Useful for Kannada text where exact matching may fail.
        """
        try:
            # Get all text blocks from the page
            page_dict = page.get_text("dict")
            page_text = page.get_text()
            
            # Try exact search first
            results = page.search_for(target_text)
            if results and len(results) > 0:
                return results
            
            # If no exact match, try fuzzy matching on text blocks
            target_normalized = self._normalize_text_for_search(target_text).lower()
            target_words = set(target_normalized.split())
            
            for block in page_dict.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        for span in line.get("spans", []):
                            line_text += span.get("text", "") + " "
                        
                        line_normalized = self._normalize_text_for_search(line_text).lower()
                        
                        # Word overlap matching
                        line_words = set(line_normalized.split())
                        if len(line_words) > 0 and len(target_words) > 0:
                            overlap = len(target_words.intersection(line_words))
                            overlap_ratio = overlap / min(len(target_words), len(line_words))
                            
                            if overlap_ratio >= threshold:
                                # Found a match, get the bounding box
                                bbox = line.get("bbox", None)
                                if bbox:
                                    return [fitz.Rect(bbox)]
                        
                        # Fuzzy string matching
                        ratio = difflib.SequenceMatcher(None, target_normalized, line_normalized).ratio()
                        if ratio >= threshold:
                            bbox = line.get("bbox", None)
                            if bbox:
                                return [fitz.Rect(bbox)]
            
            return []
            
        except Exception as e:
            print(f"    Fuzzy search error: {e}")
            return []

    def _search_text_variations(self, page, text: str, is_kannada: bool = False) -> List:
        """
        Try multiple variations of text search with better error handling.
        Improved for Kannada text.
        """
        search_results = []
        
        # For Kannada, use more aggressive fuzzy matching
        if is_kannada:
            try:
                # Try fuzzy search with lower threshold for Kannada
                results = self._fuzzy_search_in_pdf(page, text, threshold=0.7)
                if results and len(results) > 0:
                    print(f"    ✓ Found via fuzzy matching")
                    return results
            except Exception as e:
                print(f"    Fuzzy search failed: {e}")
        
        try:
            # Original text
            results = page.search_for(text)
            if results and len(results) > 0:
                return results
        except Exception as e:
            print(f"    Error in original text search: {e}")
        
        try:
            # Try without extra punctuation
            clean_text = re.sub(r'[^\w\s\u0C80-\u0CFF]', ' ', text)  # Preserve Kannada characters
            clean_text = re.sub(r'\s+', ' ', clean_text.strip())
            results = page.search_for(clean_text)
            if results and len(results) > 0:
                return results
        except Exception as e:
            print(f"    Error in clean text search: {e}")
        
        # Try searching for significant portions of the text
        try:
            words = text.split()
            if len(words) >= 3:
                # Try first 70% of words
                partial_length = int(len(words) * 0.7)
                if partial_length >= 3:
                    partial_text = ' '.join(words[:partial_length])
                    results = page.search_for(partial_text)
                    if results and len(results) > 0:
                        print(f"    ✓ Found via partial match (70%)")
                        return results
        except Exception as e:
            print(f"    Error in partial search: {e}")
        
        # Try searching for key phrases
        try:
            if len(text) > 50:
                # Try middle portion
                mid_start = len(text) // 4
                mid_end = 3 * len(text) // 4
                middle_text = text[mid_start:mid_end].strip()
                if len(middle_text) > 20:
                    results = page.search_for(middle_text)
                    if results and len(results) > 0:
                        print(f"    ✓ Found via middle portion")
                        return results
        except Exception as e:
            print(f"    Error in middle portion search: {e}")
        
        return []

    def _highlight_pdf(self, file_path: str, analysis_results: Dict, output_path: str) -> Dict:
        """
        Add highlights to sentences flagged by NLPAnalyzer with multilingual support.
        Improved for Kannada document highlighting.
        """
        # Initialize stats
        highlighting_stats = {
            'high_severity': {'expected': 0, 'found': 0, 'missed': []},
            'medium_severity': {'expected': 0, 'found': 0, 'missed': []},
            'low_severity': {'expected': 0, 'found': 0, 'missed': []}
        }
        
        try:
            doc = fitz.open(file_path)
            
            # Check if this is a Kannada document
            is_kannada = analysis_results.get('original_language') == 'kannada'
            
            # Extract all text from PDF for reference
            full_pdf_text = ""
            try:
                for page_num, page in enumerate(doc):
                    page_text = page.get_text()
                    if page_text is not None:
                        full_pdf_text += page_text
                    else:
                        print(f"Warning: Page {page_num + 1} returned None for text extraction")
                print(f"PDF contains {len(full_pdf_text)} characters of text")
                
                # Detect Kannada in PDF
                kannada_chars_in_pdf = len(re.findall(r'[\u0C80-\u0CFF]', full_pdf_text))
                if kannada_chars_in_pdf > 0:
                    print(f"PDF contains {kannada_chars_in_pdf} Kannada characters")
            except Exception as e:
                print(f"Warning: Could not extract full PDF text: {e}")

            for severity, color in self.color_map.items():
                clauses = analysis_results.get(severity, [])
                highlighting_stats[severity]['expected'] = len(clauses)
                
                print(f"\nProcessing {severity}: {len(clauses)} clauses")
                
                for i, clause in enumerate(clauses):
                    # CRITICAL: Use Kannada text if available, otherwise use English
                    if is_kannada and clause.get("kannada_text"):
                        clause_text = clause.get("kannada_text", "").strip()
                        print(f"  [{i+1}/{len(clauses)}] Using Kannada text: {clause_text[:40]}...")
                    else:
                        clause_text = clause.get("text", "").strip()
                        print(f"  [{i+1}/{len(clauses)}] Using English text: {clause_text[:40]}...")
                    
                    if not clause_text:
                        continue
                    
                    # Check if this clause failed to match during NLP analysis
                    if clause.get("match_failed", False):
                        print(f"    ⚠ Clause flagged as match-failed, skipping")
                        highlighting_stats[severity]['missed'].append(clause_text[:100])
                        continue
                    
                    found_match = False
                    
                    # Search through all pages
                    try:
                        for page_num, page in enumerate(doc):
                            try:
                                # Skip pages that might be problematic
                                if page is None:
                                    continue
                                
                                # Try different search variations
                                text_instances = self._search_text_variations(page, clause_text, is_kannada)
                                
                                if text_instances and len(text_instances) > 0:
                                    print(f"    ✅ Found {len(text_instances)} match(es) on page {page_num + 1}")
                                    found_match = True
                                    
                                    for inst in text_instances:
                                        try:
                                            highlight = page.add_highlight_annot(inst)
                                            highlight.set_colors(stroke=color)
                                            highlight.update()
                                        except Exception as e:
                                            print(f"    Error highlighting: {e}")
                                    
                                    highlighting_stats[severity]['found'] += 1
                                    break  # Found on this page, move to next clause
                            except Exception as e:
                                print(f"    Error searching page {page_num + 1}: {e}")
                                continue
                    except Exception as e:
                        print(f"    Error iterating through pages: {e}")
                    
                    if not found_match:
                        highlighting_stats[severity]['missed'].append(clause_text[:100])
                        print(f"    ❌ Could not find match in PDF")

            # Print highlighting statistics
            print("\n=== Highlighting Statistics ===")
            total_expected = 0
            total_found = 0
            for severity in highlighting_stats:
                stats = highlighting_stats[severity]
                total_expected += stats['expected']
                total_found += stats['found']
                print(f"{severity}: {stats['found']}/{stats['expected']} highlighted")
                if stats['missed']:
                    print(f"  Missed: {len(stats['missed'])} clauses")
            
            if total_expected > 0:
                success_rate = (total_found / total_expected) * 100
                print(f"\nOverall success rate: {success_rate:.1f}% ({total_found}/{total_expected})")

            # Save the document
            doc.save(output_path)
            doc.close()
            print(f"✅ Saved highlighted PDF to: {output_path}")
            
        except Exception as e:
            print(f"Error in _highlight_pdf: {e}")
            # Try to create a basic copy
            try:
                if os.path.exists(file_path):
                    import shutil
                    shutil.copy2(file_path, output_path)
                    print("Created basic copy of PDF")
            except Exception as copy_error:
                print(f"Could not create PDF copy: {copy_error}")
        
        return highlighting_stats

    def _txt_to_pdf(self, txt_file: str, pdf_file: str) -> None:
        """
        Convert a TXT file into a PDF for highlighting with better formatting.
        Supports Kannada Unicode characters.
        """
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()

        c = canvas.Canvas(pdf_file, pagesize=letter)
        width, height = letter
        y = height - 40
        margin = 40
        line_height = 15
        max_chars_per_line = 80

        # Register Kannada font if needed (optional - PyMuPDF handles Unicode well)
        # For basic Unicode support, the default font should work
        
        # Split text into lines that fit the page width
        lines = []
        for paragraph in text.splitlines():
            if len(paragraph) <= max_chars_per_line:
                lines.append(paragraph)
            else:
                # Word wrap long lines
                words = paragraph.split()
                current_line = ""
                for word in words:
                    if len(current_line + " " + word) <= max_chars_per_line:
                        current_line += " " + word if current_line else word
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)

        for line in lines:
            try:
                c.drawString(margin, y, line)
            except Exception as e:
                # If drawing fails (e.g., unsupported characters), skip
                print(f"Warning: Could not draw line: {e}")
            y -= line_height
            
            if y < 40:  # Start new page
                c.showPage()
                y = height - 40

        c.save()