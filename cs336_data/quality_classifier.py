import fasttext
import os

_quality_model = None

def _get_quality_model():
    global _quality_model
    if _quality_model is None:
        model_path = "quality_classifier.bin"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'model file not found')
        _quality_model = fasttext.load_model(model_path)
    return _quality_model

def classify_quality(text: str) -> tuple[str, float]:
    if not text or not text.strip():
        return ("low", 0.0)
        
    model = _get_quality_model()
    
    clean_text = text.replace('\n', ' ').replace('\r', ' ')
    
    predictions, probabilities = model.predict(clean_text, k=1)
    
    raw_label = predictions[0]
    label = raw_label.replace('__label__', '')
    score = float(probabilities[0])
    
    return label, score