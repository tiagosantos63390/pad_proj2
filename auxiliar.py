import re
import numpy as np
import os
import spacy
import string
import unicodedata
import random

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
    result = Counter()

    for n in range(1, max_n + 1):
        ngrams = zip(*(islice(tokens, i, None) for i in range(n)))
        result.update(ngrams)

    return result


def filter_by_frequency(ngrams, min_freq):
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


def is_local_max(ngram, scores, ngram_counts):
    n = len(ngram)
    score = scores.get(ngram, 0)

    for i in range(n):
        sub_ngram = ngram[:i] + ngram[i+1:]
        if sub_ngram in scores and scores[sub_ngram] > score:
            return False

    for other_ngram in scores.keys():
        if len(other_ngram) == n + 1:

            for j in range(len(other_ngram) - n + 1):
                if other_ngram[j:j+n] == ngram and scores[other_ngram] > score:
                    return False
    return True


def extract_local_max(scores, ngram_counts, min_len=2):
    local_max_ngrams = {}

    for ngram in scores:
        if len(ngram) >= min_len and is_local_max(ngram, scores, ngram_counts):
            local_max_ngrams[ngram] = scores[ngram]

    return local_max_ngrams


def select_most_informative(localmax_scores, top_n=15):
    sorted_res = sorted(
        localmax_scores.items(),
        key=lambda x: (x[1], len(x[0])),
        reverse=True
    )

    return sorted_res[:top_n]


def calculate_similarity_matrix(ngram_sets):
    all_res = list(set(' '.join(ng) for ngrams in ngram_sets for ng in ngrams))
    
    doc_texts = [' '.join(' '.join(ng) for ng in ngrams) for ngrams in ngram_sets]

    vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split())
    tfidf_matrix = vectorizer.fit_transform(doc_texts + all_res)

    re_matrix = tfidf_matrix[-len(all_res):]
    similarity = cosine_similarity(re_matrix)

    return {
        all_res[i]: {
            all_res[j]: similarity[i][j] for j in range(len(all_res)) if i != j
        } for i in range(len(all_res))
    }


def find_implicit_keywords(explicit_keywords, similarity_matrix, threshold=0.3, top_n=5):
    implicit_keywords = set()
    
    for ek in explicit_keywords:
        ek = ek.lower()
        if ek not in similarity_matrix:
            continue
        
        similar = similarity_matrix.get(ek, {})
        candidates = [(re, sim) for re, sim in similar.items() if sim >= threshold and re != ek]
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        for re, _ in candidates[:top_n]:
            implicit_keywords.add(re)
    
    return list(implicit_keywords)


def process_documents_for_keywords(directory, num_files=None):
    files = sorted([f for f in os.listdir(directory) if f.startswith('fil_')],
                  key=lambda x: int(x.split('_')[1]))
    
    if num_files:
        files = files[:num_files]
    
    all_documents = []
    all_explicit_keywords = []
    all_ngrams = []
    
    for filename in tqdm(files, desc="Extracting REs"):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
            original = f.read()
        
        preprocessed = preprocess_text(original)
        tokens = tokenize(preprocessed)
        unigrams = extract_ngrams(tokens, 1)
        ngrams = extract_ngrams(tokens, 7)
        filtered = filter_by_frequency(ngrams, 2)
        
        scp_scores = {ng: scp_score(ng, filtered, unigrams) for ng in filtered}
        dice_scores = {ng: dice_score(ng, filtered, unigrams) for ng in filtered}
        phi2_scores = {ng: phi2_score(ng, filtered, unigrams) for ng in filtered}

        sorted_scp_scores = dict(sorted(scp_scores.items(), key=lambda x: x[1], reverse=True))
        sorted_dice_scores = dict(sorted(dice_scores.items(), key=lambda x: x[1], reverse=True))
        sorted_phi2_scores = dict(sorted(phi2_scores.items(), key=lambda x: x[1], reverse=True))
        
        localmax_scp = extract_local_max(sorted_scp_scores, filtered)
        localmax_dice = extract_local_max(sorted_dice_scores, filtered)
        localmax_phi2 = extract_local_max(sorted_phi2_scores, filtered)
        
        combined_localmax = {}
        for method in [localmax_scp, localmax_dice, localmax_phi2]:
            for ng, score in method.items():
                key = ' '.join(ng)
                if key not in combined_localmax or score > combined_localmax[key]:
                    combined_localmax[key] = score
        
        informative_res = select_most_informative(combined_localmax)
        
        all_documents.append({
            'filename': filename,
            'original': original,
            'preprocessed': preprocessed,
            'tokens': tokens,
            'localmax': combined_localmax,
            'informative_res': informative_res
        })
        
        all_ngrams.append(set(tuple(kw.split()) for kw in combined_localmax.keys()))
        all_explicit_keywords.append([(kw, score) for kw, score in informative_res])
    
    similarity_matrix = calculate_similarity_matrix(all_ngrams)
    
    for i, doc in enumerate(all_documents):
        explicit_keywords = [ek[0] for ek in all_explicit_keywords[i]]
        implicit_keywords = find_implicit_keywords(
            explicit_keywords, 
            similarity_matrix
        )
        
        doc['explicit_keywords'] = explicit_keywords
        doc['implicit_keywords'] = implicit_keywords
    
    return all_documents


if __name__ == "__main__":
    directory = 'corpus2mw'
    documents_with_keywords = process_documents_for_keywords(directory, num_files=2)
    
    for doc in documents_with_keywords:
        print(f"\nDocument: {doc['filename']}")

        print("\nAll Local Maxs (REs):")
        for i, (kw, score) in enumerate(sorted(doc['localmax'].items(), key=lambda x: -x[1]), 1):
            print(f"{i}. {kw} (score: {score:.4f})")

        print("\nExplicit Keywords (Top Informative REs):")
        for i, (kw, score) in enumerate(doc['informative_res'], 1):
            print(f"{i}. {kw} (score: {score:.4f})")
        
        print("\nImplicit Keywords (Similarity-Based):")
        for i, kw in enumerate(doc['implicit_keywords'], 1):
            print(f"{i}. {kw}")
        
        print("\n" + "="*100 + "\n")
