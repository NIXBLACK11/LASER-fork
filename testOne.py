import os
from laser_encoders.language_list import LASER2_LANGUAGE, LASER3_LANGUAGE
from laser_encoders.download_models import LaserModelDownloader, initialize_encoder, initialize_tokenizer

def validate_language_models_and_tokenize():
    downloader = LaserModelDownloader()
    failed_languages = []
    i=0

    for lang in LASER3_LANGUAGE:
        i=i+1
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
            # Delete the downloaded models
            model_files = [f"laser3-{lang}.v1.pt", f"laser3-{lang}.v1.spm", f"laser3-{lang}.v1.cvocab"]
            for file in model_files:
                os.remove(os.path.join(downloader.model_dir, file))
        
        if i==1:
            break


if __name__ == "__main__":
    validate_language_models_and_tokenize()