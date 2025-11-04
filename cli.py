import argparse
import os
from src.document_processor import DocumentProcessor
from src.nlp_analyzer import NLPAnalyzer
from src.document_highlighter import DocumentHighlighter

def main():
    parser = argparse.ArgumentParser(description="AI Lease Agreement Analyzer")
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the lease agreement file (PDF, DOCX, or TXT)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the highlighted document (default: highlighted_[original_filename])"
    )
    
    args = parser.parse_args()
    
    # Initialize components
    document_processor = DocumentProcessor()
    nlp_analyzer = NLPAnalyzer()
    document_highlighter = DocumentHighlighter()
    
    try:
        # Process document
        print("Processing document...")
        text, file_ext = document_processor.process_document(args.file)
        
        # Analyze text
        print("Analyzing content...")
        analysis_results = nlp_analyzer.analyze_text(text)
        
        # Calculate favorability score
        score = nlp_analyzer.calculate_favorability_score(analysis_results)
        
        # Generate summary
        summary = nlp_analyzer.generate_summary(analysis_results)
        
        # Display results
        print("\nAnalysis Results:")
        print(f"Favorability Score: {score}/10")
        print("\nDocument Summary:")
        print(summary)
        print("\nClause Statistics:")
        print(f"High Severity Clauses: {len(analysis_results['high_severity'])}")
        print(f"Medium Severity Clauses: {len(analysis_results['medium_severity'])}")
        print(f"Low Severity Clauses: {len(analysis_results['low_severity'])}")
        
        # Generate output path
        if args.output:
            output_path = args.output
        else:
            filename = os.path.basename(args.file)
            output_path = f"highlighted_{filename}"
        
        # Highlight document
        print("\nGenerating highlighted document...")
        document_highlighter.highlight_document(
            args.file,
            analysis_results,
            output_path
        )
        
        print(f"\nHighlighted document saved to: {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main()) 