from transformers import MarianMTModel, MarianTokenizer

# ✅ Use a multilingual model that includes Kannada among supported languages
model_name = "Helsinki-NLP/opus-mt-mul-en"

# Load tokenizer and model
print("Loading model...")
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_kn_to_en(text):
    """Translate Kannada text to English."""
    # Prepare input
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    
    # Generate translation
    translated = model.generate(**inputs, max_length=256)
    
    # Decode output
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Example Kannada text
kannada_text = "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಾ?"
english_translation = translate_kn_to_en(kannada_text)

print("Kannada:", kannada_text)
print("English:", english_translation)
