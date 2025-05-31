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


def tfdi_score(term, doc_tokens, all_documents):
    tf = doc_tokens.count(term) / len(doc_tokens)
    df = sum(1 for doc in all_documents if term in doc)
    idf = np.log(len(all_documents) / (df + 1))

    return tf * idf


def calculate_covariance(term1, term2, all_docs):
    mean_t1 = np.mean([doc.count(term1) for doc in all_docs])
    mean_t2 = np.mean([doc.count(term2) for doc in all_docs])

    cov = np.mean([
        (doc.count(term1) - mean_t1) * (doc.count(term2) - mean_t2)
        for doc in all_docs
    ])
    return cov


def correlation_score(term, keywords, all_docs):
    numerator = sum(calculate_covariance(term, kw, all_docs) for kw in keywords)
    denominator = sum(
        np.sqrt(calculate_covariance(term, term) * calculate_covariance(kw, kw)) 
        for kw in keywords
    )
    return numerator / denominator if denominator else 0


# filtrar só depois de obter REs
# tfdi
# implicitas: Score(T, di) = (SUMATÓRIO corr(T, Ki)) / i (Ki ∈ KeyWordsExplicitas(di))
# coV(T, Ki) = (1 / len(DOCS)) * SUMATÓRIO ((P(T,d)-P(T, Docs)) *(P(Ki, d) - P(Ki, Docs))) / i (d ∈ Docs)
# corr(T, Ki) = coV(T, Ki) / sqrt(coV(T, T)) * sqrt(coV(Ki, Ki))
# uma métrica de cada vez
