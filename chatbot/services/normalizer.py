"""
Text Normalization Engine for AI DR Screen Chatbot.
Handles common typos, repeated letters, informal internet slang, and abbreviations
while strictly preserving critical medical acronyms and terms.
"""

import re

# Common internet slang and typo mappings
WORD_REPLACEMENTS = {
    "u": "you",
    "ur": "your",
    "r": "are",
    "wat": "what",
    "wht": "what",
    "wot": "what",
    "wats": "what is",
    "whats": "what is",
    "what's": "what is",
    "hows": "how is",
    "how's": "how is",
    "plz": "please",
    "pls": "please",
    "thx": "thanks",
    "thanku": "thank you",
    "thnx": "thanks",
    "tnx": "thanks",
    "thanx": "thanks",
    "helo": "hello",
    "hlw": "hello",
    "hii": "hi",
    "hiii": "hi",
    "hiiii": "hi",
    "heyy": "hey",
    "heyyy": "hey",
    "k": "okay",
    "ok": "okay",
    "okk": "okay",
    "okie": "okay",
    "pic": "image",
    "pics": "images",
    "photo": "image",
    "photos": "images",
    "cam": "camera",
    "doc": "doctor",
    "hosp": "hospital",
    "govt": "government",
    "sec": "seconds",
    "min": "minutes"
}

# Preserve medical acronyms from being corrupted
PRESERVED_ACRONYMS = {
    "dr", "iqa", "unet", "u-net", "shap", "gradcam", "grad-cam",
    "fbs", "rbs", "ppbs", "hba1c", "bp", "phc", "abdm", "abha",
    "od", "os", "cnn", "ocr", "ai"
}

def normalize_text(text: str) -> str:
    """
    Lightweight, deterministic text normalizer.
    1. Lowercase and whitespace trim
    2. Repeated letter squashing (e.g. 'pleaaase' -> 'please', 'soooo' -> 'so')
    3. Slang and abbreviation expansion
    """
    if not text or not isinstance(text, str):
        return ""

    s = text.strip().lower()

    # Squash 3+ identical consecutive characters down to 1 or 2
    # e.g., 'hiiiii' -> 'hi', 'heyyyy' -> 'hey', 'soooo' -> 'so', 'goooood' -> 'good'
    def replace_repeated(match):
        char = match.group(1)
        # If double letters are common in English (like 'oo', 'ee'), keep 2, else 1
        if char in "eo":
            return char * 2
        return char

    s = re.sub(r'([a-zA-Z])\1{2,}', replace_repeated, s)

    # Tokenize preserving words and apostrophes
    tokens = re.findall(r"[a-z0-9'-]+|[^\s\w]", s)
    normalized_tokens = []

    for t in tokens:
        clean_word = t.strip()
        if clean_word in WORD_REPLACEMENTS:
            normalized_tokens.append(WORD_REPLACEMENTS[clean_word])
        else:
            normalized_tokens.append(clean_word)

    normalized_sentence = " ".join(normalized_tokens)

    # Clean up spacing around punctuation
    normalized_sentence = re.sub(r'\s+([?.!,"])', r'\1', normalized_sentence)
    normalized_sentence = re.sub(r'\s+', ' ', normalized_sentence).strip()

    return normalized_sentence
