from deep_translator import GoogleTranslator

def translate_to_english(text: str) -> tuple[str, str]:
    """
    Translates the given text to English using Google Translate.

    Args:
        text (str): The text to translate.

    Returns:
        tuple[str, str]: The translated text and the source language.
    """
    if not text:  # If the text is None or empty, return it as-is
        return text, "unknown"

    try:
        # Translate to English (auto-detect source language)
        translated_text = GoogleTranslator(source='auto', target='en').translate(text)
        return translated_text, "auto"  # Return translated text and "auto" as the detected language
    except Exception as e:
        print(f"Translation failed: {e}. Using original text.")
        return text, "unknown"  # Return the original text and "unknown" language if translation fails