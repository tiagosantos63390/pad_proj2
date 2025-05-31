import re
import os
import math

from tqdm import tqdm
from collections import Counter
from itertools import islice


def preprocess_text(text):
    # normalize punctuation
    text = text.replace("’", "'").replace("‘", "'").replace("−", "-").replace("–", "-").replace("—", "-")

    # agregate numbers with commas like 4,000–7,000 → 4000_7000
    text = re.sub(r'(\d{1,3}(?:,\d{3})*)\s*[-–—]\s*(\d{1,3}(?:,\d{3})*)',
                  lambda m: m.group(1).replace(',', '') + '_' + m.group(2).replace(',', ''),
                  text)

    # remove commas in numbers like 9,500 → 9500
    text = re.sub(r'(?<=\d),(?=\d)', '', text)

    # remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # substitute hyphens between words with underscores like "word-word" → "word_word"
    text = re.sub(r"(?<=\w)-(?=\w)", "_", text)

    # remove characters that are not letters, numbers, spaces, or underscores
    text = re.sub(r"[^a-zA-Z0-9\s_áéíóúâêîôûàèìòùãõçÁÉÍÓÚÂÊÎÔÛÀÈÌÒÙÃÕÇ']", ' ', text)

    # normalize spaces
    text = re.sub(r'\s+', ' ', text)

    return text.lower().strip()


def tokenize(text):
    return re.findall(r'\b\w+\b', text)


def extract_ngrams(tokens, max_n):
    result = Counter()

    for n in range(1, max_n + 1):
        ngrams = zip(*(islice(tokens, i, None) for i in range(n)))
        result.update(ngrams)

    return result


""" 
    the scores must be contained in the interval [0, 1],
    where 0 means no association and 1 means perfect association.
"""
def dice_score(ngram, freq, ngrams):
    if len(ngram) < 2:
        return 0

    total = 0
    for k in range(1, len(ngram)):
        part1 = ngram[:k]
        part2 = ngram[k:]
        freq1 = ngrams.get(part1, 0)
        freq2 = ngrams.get(part2, 0)
        total += freq1 + freq2

    if total == 0:
        return 0

    return (freq * 2) / (total / (len(ngram) - 1))


"""
    ?
"""
def scp_score(ngram, freq, ngrams):
    if len(ngram) < 2:
        return 0

    total = 0
    for k in range(1, len(ngram)):
        part1 = ngram[:k]
        part2 = ngram[k:]
        freq1 = ngrams.get(part1, 0)
        freq2 = ngrams.get(part2, 0)
        total += freq1 * freq2

    if total == 0:
        return 0

    return (freq ** 2) / (total / (len(ngram) - 1))


"""
    ?
"""
def mi_score(ngram, freq, ngrams, corpus_size):
    if len(ngram) < 2:
        return 0

    total = 0
    for k in range(1, len(ngram)):
        part1 = ngram[:k]
        part2 = ngram[k:]
        freq1 = ngrams.get(part1, 0)
        freq2 = ngrams.get(part2, 0)
        total += (freq1 / corpus_size) * (freq2 / corpus_size)

    if total == 0:
        return 0

    return math.log((freq / corpus_size) / (total / (len(ngram) - 1)))


def process_documents_for_keywords(directory, num_files=None, specific_file=None, metric="scp"):
    files = sorted([f for f in os.listdir(directory) if f.startswith('fil_')],
                  key=lambda x: int(x.split('_')[1]))
    
    if specific_file:
        files = [specific_file] if specific_file in files else exit(f"File {specific_file} not found in directory {directory}.")
    elif num_files:
        files = files[:num_files]
    
    all_documents = []
    
    for filename in tqdm(files, desc="Extracting REs"):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
            original = f.read()
        
        preprocessed = preprocess_text(original)
        tokens = tokenize(preprocessed)
        ngrams = extract_ngrams(tokens, max_n=7) # up to 7 words per n-gram
        
        dice_scores = {}
        for ngram, freq in ngrams.items():
            score = dice_score(ngram, freq, ngrams)
            dice_scores[ngram] = score
        sorted_dice_scores = sorted(dice_scores.items(), key=lambda x: x[1], reverse=True)

        scp_scores = {}
        for ngram, freq in ngrams.items():
            score = scp_score(ngram, freq, ngrams)
            scp_scores[ngram] = score
        sorted_scp_scores = sorted(scp_scores.items(), key=lambda x: x[1], reverse=True)

        mi_scores = {}
        for ngram, freq in ngrams.items():
            score = mi_score(ngram, freq, ngrams, len(tokens))
            mi_scores[ngram] = score
        sorted_mi_scores = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)

        all_documents.append({
            'filename': filename,
            'original': original,
            'preprocessed': preprocessed,
            'tokens': tokens,
            'dice_scores': sorted_dice_scores,
            'scp_scores': sorted_scp_scores,
            'mi_scores': sorted_mi_scores
        })
    
    return all_documents


if __name__ == "__main__":
    # directory containing the corpus files
    directory = 'corpus2mw'

    # process a specific file
    #documents_with_keywords = process_documents_for_keywords(directory, specific_file='fil_24')
    # or process a limited number of files
    documents_with_keywords = process_documents_for_keywords(directory, num_files=5)
    # or process all files
    #documents_with_keywords = process_documents_for_keywords(directory)
    
    for doc in documents_with_keywords:
        print(f"\nDocument: {doc['filename']}")

        #print("\nOriginal Text:")
        #print(doc['original'])

        #print("\nPreprocessed Text:")
        #print(doc['preprocessed'])

        print("\nDice Scores:")
        for ngram, score in doc['dice_scores'][:10]:
            print(f"{' '.join(ngram)}: {score:.4f}")

        print("\nSCP Scores:")
        for ngram, score in doc['scp_scores'][:10]:
            print(f"{' '.join(ngram)}: {score:.4f}")

        print("\nMI Scores:")
        for ngram, score in doc['mi_scores'][:10]:
            print(f"{' '.join(ngram)}: {score:.4f}")

        print("\n" + "=" * 50)
