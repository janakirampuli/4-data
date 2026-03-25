import fasttext
import os

_nsfw_model = None
_toxic_model = None

def _get_model(model_type: str):
    global _nsfw_model, _toxic_model
    
    if model_type == "nsfw":
        if _nsfw_model is None:
            model_path = "./data/jigsaw_fasttext_bigrams_nsfw_final.bin"
            if os.path.exists(model_path):
                _nsfw_model = fasttext.load_model(model_path)
            else:
                raise FileNotFoundError("model not found")
        return _nsfw_model
        
    elif model_type == "toxic":
        if _toxic_model is None:
            model_path = "./data/jigsaw_fasttext_bigrams_hatespeech_final.bin"
            if os.path.exists(model_path):
                _toxic_model = fasttext.load_model(model_path)
            else:
                raise FileNotFoundError("model not found")
        return _toxic_model
    
def classify_nsfw(text: str) -> tuple[str, float]:
    if not text or not text.strip():
        return ("unknown", 0.0)
    
    model = _get_model("nsfw")
    clean_text = text.replace('\n', ' ').replace('\r', ' ')

    pred, prob = model.predict(clean_text, k=1)
    label = pred[0].replace('__label__', '')

    return label, float(prob[0])

def classify_toxic_speech(text: str) -> tuple[str, float]:
    if not text or not text.strip():
        return ("unknown", 0.0)
    
    model = _get_model("toxic")
    clean_text = text.replace('\n', ' ').replace('\r', ' ')

    pred, prob = model.predict(clean_text, k=1)
    label = pred[0].replace('__label__', '')

    return label, float(prob[0])

if __name__ == "__main__":
    prediction, score = classify_nsfw(
        "SUCK MY C*CK WIKIPEDIA EDITORS...F*CKING *SSH*LE DORKS."
        "JUST TRYING TO MAKE THE SITE BETTER YOU UPTIGHT C*NTS"
    )

    print(prediction, score)

    prediction, score = classify_toxic_speech(
        "JUST TRYING TO MAKE THE SITE BETTER YOU UPTIGHT C*NTS"
    )

    print(prediction, score)