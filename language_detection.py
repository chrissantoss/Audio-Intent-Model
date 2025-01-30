from langdetect import detect, LangDetectException

def detect_language(text):
    """
    Detect the language of input text.
    Returns ISO 639-1 language code (e.g., 'en', 'es', 'fr', etc.)
    """
    try:
        return detect(text)
    except LangDetectException:
        return 'en'  # Default to English if detection fails 