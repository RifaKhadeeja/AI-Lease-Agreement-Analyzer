# AI Lease Agreement Analyzer

An intelligent system that analyzes lease agreements using Natural Language Processing (NLP) to identify and highlight important clauses, generate summaries, and provide a favorability score for renters.

## Features

- Document upload support (.pdf, .docx, .txt)
- NLP-based clause detection and severity classification
- Visual highlighting of clauses by severity:
  - 🔴 High severity (red)
  - 🟡 Medium severity (yellow)
  - 🔵 Low severity (blue)
- Document summary generation
- Renter favorability scoring (0-10)

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Download the spaCy model:
```bash
python -m spacy download en_core_web_lg
```

## Usage

### Streamlit UI
```bash
streamlit run app.py
```

### CLI Mode
```bash
python cli.py --file path/to/lease_agreement.pdf
```

## Project Structure

```
lease_agreement_analyser/
├── app.py                 # Streamlit web application
├── cli.py                 # Command-line interface
├── src/
│   ├── document_processor.py  # Document handling and processing
│   ├── nlp_analyzer.py       # NLP-based analysis
│   ├── clause_classifier.py  # Clause classification
│   ├── document_highlighter.py # Document highlighting
│   └── summary_generator.py   # Summary generation
├── models/               # Trained models (if any)
└── tests/               # Test files
```

## License

MIT License 