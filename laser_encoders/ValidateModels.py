import os
from language_list import LASER2_LANGUAGE, LASER3_LANGUAGE
from download_models import LaserModelDownloader, initialize_encoder, initialize_tokenizer

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
        finally:
            # Delete the downloaded models if they exist
            model_files = [f"laser3-{lang}.v1.pt", f"laser3-{lang}.v1.spm", f"laser3-{lang}.v1.cvocab"]
            for file in model_files:
                file_path = os.path.join(downloader.model_dir, file)
                if os.path.exists(file_path):
                    os.remove(file_path)

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
        finally:
            # Delete the downloaded models if they exist
            model_files = ["laser2.pt", "laser2.spm", "laser2.cvocab"]
            for file in model_files:
                file_path = os.path.join(downloader.model_dir, file)
                if os.path.exists(file_path):
                    os.remove(file_path)

    if not failed_languages:
        print("All language models validated and deleted successfully.")
    else:
        print("Failed to validate the following language models:")
        for lang, error in failed_languages:
            print(f"Language: {lang}, Error: {error}")

if __name__ == "__main__":
    validate_language_models_and_tokenize()
