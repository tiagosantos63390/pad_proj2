import re
import numpy as np
import os
import spacy
import string
import unicodedata

from collections import defaultdict, Counter
from itertools import islice
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def preprocess_text(text):
    text = unicodedata.normalize('NFKD', text)

    text = text.replace("’", "'").replace("‘", "'").replace("−", "-").replace("–", "-").replace("—", "-")

    text = ''.join(c for c in text if not unicodedata.combining(c))

    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r"(?<=\w)-(?=\w)", "_", text)
    text = re.sub(r"[^a-zA-Z0-9\s_']", ' ', text)  
    text = re.sub(r'\s+', ' ', text)

    return text.lower().strip()


nlp = spacy.load("en_core_web_sm")


def tokenize(text):
    doc = nlp(text)
    tokens = []

    for token in doc:
        if token.is_punct or token.is_space:
            continue

        txt = token.text.lower()

        if "_" in txt:
            if re.match(r'^[a-z0-9_]+$', txt):
                tokens.append(txt)
                
        else:
            lemma = token.lemma_.lower()
            if token.pos_ == "VERB" or token.like_num or token.is_alpha:
                if re.match(r'^[a-z0-9]+$', lemma): 
                    tokens.append(lemma)

    return tokens


def extract_ngrams(tokens, max_n):
    ngram_counts = Counter()

    for n in range(1, max_n + 1):
        ngrams = zip(*(islice(tokens, i, None) for i in range(n)))
        ngram_counts.update(ngrams)

    return ngram_counts


def filter_by_frequency(ngrams, min_freq=2):
    return {k: v for k, v in ngrams.items() if v >= min_freq and len(k) > 1}


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


directory = 'corpus2mw'


files = sorted([f for f in os.listdir(directory) if f.startswith('fil_')],
               key=lambda x: int(x.split('_')[1]))


for filename in files[:4]:
    with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
        original = f.read()

    preprocessed = preprocess_text(original)
    tokens = tokenize(preprocessed)
    ngrams = extract_ngrams(tokens, 7)
    unigrams = extract_ngrams(tokens, 1)
    filtered = filter_by_frequency(ngrams)

    scp_scores = {ng: scp_score(ng, filtered, unigrams) for ng in filtered}
    dice_scores = {ng: dice_score(ng, filtered, unigrams) for ng in filtered}
    phi2_scores = {ng: phi2_score(ng, filtered, unigrams) for ng in filtered}

    print(f"Original text from {filename}:\n{original[:500]}...\n")
    print(f"Preprocessed text from {filename}:\n{preprocessed[:500]}...\n")

    print(f"Tokens from {filename}:\n{tokens}...\n")

    print(f"Most common n-grams from {filename}:\n{ngrams.most_common(5)}\n")
    print(f"Most common uni-grams from {filename}:\n{unigrams.most_common(5)}\n")

    print(f"Filtered n-grams from {filename}:\n{filtered}")

    print("\nTop 10 by SCP:")
    for ng, score in sorted(scp_scores.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{' '.join(ng)}: {score:.4f}")
    
    print("\nTop 10 by Dice:")
    for ng, score in sorted(dice_scores.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{' '.join(ng)}: {score:.4f}")
    
    print("\nTop 10 by Phi²:")
    for ng, score in sorted(phi2_scores.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{' '.join(ng)}: {score:.4f}")
        
    print(f"\n==========================================================================\n")
