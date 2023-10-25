from laser_encoders.language_list import LASER2_LANGUAGE, LASER3_LANGUAGE
from laser_encoders.download_models import LaserModelDownloader, initialize_encoder, initialize_tokenizer

def validate_language_models_and_tokenize():
    downloader = LaserModelDownloader()
    failed_languages = []

    for lang in LASER3_LANGUAGE:
        try:
            # Use the downloader to download the model
            downloader.download_laser3(lang)
            encoder = initialize_encoder(lang)
            tokenizer = initialize_tokenizer(lang)
            # Test tokenization with a sample sentence
            tokenized = tokenizer.tokenize("This is a sample sentence.")
        except Exception as e:
            failed_languages.append((lang, e))

    for lang in LASER2_LANGUAGE:
        try:
            # Use the downloader to download the model
            downloader.download_laser2()
            encoder = initialize_encoder(lang, laser="laser2")
            tokenizer = initialize_tokenizer(lang)
            # Test tokenization with a sample sentence
            tokenized = tokenizer.tokenize("This is a sample sentence.")
        except Exception as e:
            failed_languages.append((lang, e))

    if not failed_languages:
        print("All language models validated successfully.")
    else:
        print("Failed to validate the following language models:")
        for lang, error in failed_languages:
            print(f"Language: {lang}, Error: {error}")

if __name__ == "__main__":
    validate_language_models_and_tokenize()
