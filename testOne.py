from laser_encoders.language_list import LASER2_LANGUAGE, LASER3_LANGUAGE
from laser_encoders.download_models import LaserModelDownloader, initialize_encoder, initialize_tokenizer

def validate_language_models_and_tokenize():
    downloader = LaserModelDownloader()
    failed_languages = []

    lang  = LASER2_LANGUAGE[0]
    try:
        # Use the downloader to download the model
        downloader.download_laser3(lang)
        encoder = initialize_encoder(lang)
        tokenizer = initialize_tokenizer(lang)
        # Test tokenization with a sample sentence
        tokenized = tokenizer.tokenize("This is a sample sentence.")
        
    except Exception as e:
        failed_languages.append((lang, e))


if __name__ == "__main__":
    validate_language_models_and_tokenize()