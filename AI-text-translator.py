from transformers import MarianMTModel, MarianTokenizer

# Function to translate
def translate(text, src_lang="it", tgt_lang="en"):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    # Tokenization
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    
    # Decoding
    translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return translated_text[0]

# Input from terminal
text = input("Scrivi una frase da tradurre in inglese: ")

# Output
translated_text = translate(text)
print(f"\nTraduzione: {translated_text}")
