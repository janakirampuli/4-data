import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

def gopher_quality_filter(text: str) -> bool:
    if not text or not text.strip():
        return False

    words = nltk.word_tokenize(text)
    num_words = len(words)

    if num_words < 50 or num_words > 100000:
        return False
    
    total_length = sum(len(w) for w in words)
    mean_word_length = total_length / num_words
    if mean_word_length < 3 or mean_word_length > 10:
        return False
    
    alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
    if (alpha_words / num_words) < 0.80:
        return False
    
    lines = text.splitlines()
    if lines:
        num_lines = len(lines)
        ellipsis_lines = sum(1 for line in lines if line.strip().endswith("..."))
        if (ellipsis_lines / num_lines) > 0.30:
            return False

    return True


if __name__ == "__main__":
    text = "The string you are reading is a short snippet of text."
    print(gopher_quality_filter(text))