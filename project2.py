import re
import numpy as np
import os
import spacy
import string

from collections import defaultdict, Counter
from itertools import islice
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def preprocess_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)        # remove pontuação
    text = re.sub(r'\s+', ' ', text)                   # normaliza espaços
    return text.lower().strip()

def scp_score(ngram, ngram_counts, unigram_counts, alpha=0.01):
    total_words = sum(unigram_counts.values())
    vocab_size = len(unigram_counts)
    joint_prob = (ngram_counts.get(ngram, 0) + alpha) / (total_words + alpha * vocab_size)
    product = 1
    for word in ngram:
        p_word = (unigram_counts.get((word,), 0) + alpha) / (total_words + alpha * vocab_size)
        product *= p_word
    return (joint_prob ** 2) / product if product != 0 else 0

def dice_score(ngram, ngram_counts, unigram_counts):
    f_ngram = ngram_counts[ngram]
    f_sum = sum(unigram_counts.get((w,), 0) for w in ngram)
    return (len(ngram) * f_ngram) / f_sum if f_sum else 0

def phi2_score(ngram, ngram_counts, unigram_counts):
    O = ngram_counts[ngram]
    N = sum(unigram_counts.values())
    E = 1
    for w in ngram:
        E *= unigram_counts.get((w,), 1) / N
    E *= N
    return ((O - E) ** 2) / E if E else 0

def extract_ngrams(tokens, max_n=7):
    ngram_counts = Counter()
    for n in range(1, max_n+1):
        for i in range(len(tokens)-n+1):
            ngram = tuple(tokens[i:i+n])
            ngram_counts[ngram] += 1
    return ngram_counts

def filter_by_frequency(ngrams, min_freq=2):
    return {k: v for k, v in ngrams.items() if v >= min_freq and len(k) > 1}

def print_top(title, scores):
    print(f"\n{title}")
    for ngram, score in sorted(scores.items(), key=lambda x: -x[1])[:10]:
        print(f"{' '.join(ngram):<40} → {score:.3f}")

nlp = spacy.load("en_core_web_sm")

def tokenize(text):
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]

directory = 'corpus2mw'

files = sorted([f for f in os.listdir(directory) if f.startswith('fil_')],
               key=lambda x: int(x.split('_')[1]))

for filename in files[:5]:
    with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
        original = f.read()

    preprocessed = preprocess_text(original)
    tokens = tokenize(preprocessed)

    ngram_counts = extract_ngrams(tokens, max_n=7)
    filtered_ngrams = filter_by_frequency(ngram_counts, min_freq=2)
    unigram_counts = extract_ngrams(tokens, max_n=1)

    scp_scores = {
        ngram: np.log1p(scp_score(ngram, filtered_ngrams, unigram_counts))
        for ngram in filtered_ngrams
    }

    dice_scores = {
        ngram: np.log1p(dice_score(ngram, filtered_ngrams, unigram_counts))
        for ngram in filtered_ngrams
    }

    phi2_scores = {
        ngram: np.log1p(phi2_score(ngram, filtered_ngrams, unigram_counts))
        for ngram in filtered_ngrams
    }

    sorted_scp_scores = sorted(scp_scores.items(), key=lambda x: -x[1])
    sorted_dice_scores = sorted(scp_scores.items(), key=lambda x: -x[1])
    sorted_phi2_scores = sorted(scp_scores.items(), key=lambda x: -x[1])

    print(f"\nTop 10 n-grams in {filename}:")
    print_top("🔹 SCP", scp_scores)
    print_top("🔹 Dice", dice_scores)
    print_top("🔹 Phi²", phi2_scores)
    