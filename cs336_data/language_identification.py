import fasttext
import os

_fasttext_model = None

def _get_fasttext_model():
    global _fasttext_model
    if _fasttext_model is None:
        model_path = "./data/lid.176.bin"

        if os.path.exists(model_path):
            _fasttext_model = fasttext.load_model(model_path)
        else:
            raise FileNotFoundError("model not found")
        
    return _fasttext_model

def identify_language(text: str) -> tuple[str, float]:
    if not text or not text.strip():
        return ("unknown", 0.0)
    
    model = _get_fasttext_model()

    clean_text = text.replace('\n', ' ').replace('\r', ' ')

    pred, prob = model.predict(clean_text, k=1)

    raw_label = pred[0]
    lang_id = raw_label.replace('__label__', '')
    score = float(prob[0])

    return lang_id, score

if __name__ == "__main__":
    l, s = identify_language("bonjour")
    print(l, s)