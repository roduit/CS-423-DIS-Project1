#All the functions in this file are used to compute the BM25 ranking.
import os
import math
from collections import defaultdict

def BM25_preprocessing(corpus_tokenized):
    """
    Precompute idf, tf, avg_doc_len, and doc_len for the BM25 ranking
    
    :param corpus_tokenized: list of lists of str, with tokenized documents.
    :return: idf(dict), tf(dict), avg_doc_len(float), doc_len(list)
    """
    
    # Compute the document frequency for each term in the corpus.
    df = {}
    for document in corpus_tokenized:
        for term in set(document):
            if term in df:
                df[term] += 1
            else:
                df[term] = 1
    
    # Compute the inverse document frequency for each term in the corpus.
    idf = {}
    num_documents = len(corpus_tokenized)
    for term in df:
        idf[term] = math.log(1 + (num_documents - df[term] + 0.5) / (df[term] + 0.5))

    #Compute the term frequency for each in term in each document
    tf = {}
    for i, document in enumerate(corpus_tokenized):
        tf[i] = {}
        for term in document:
            if term in tf[i]:
                tf[i][term] += 1
            else:
                tf[i][term] = 1

    #Compute doc_len
    doc_len = [len(document) for document in corpus_tokenized]
    # Compute the average document length. 
    avg_doc_len = sum(doc_len) / len(doc_len)
    
  
    return idf, tf, avg_doc_len, doc_len

def BM25_score(query, document_id, idf, tf, avg_doc_len, doc_len, k1=1.5, b=0.75):
    """
    Compute the BM25 score for a given query and the document position in the corpus

    :param query: list of str (tokenized query).
    :param document_id: int (document position in every list; it's not docid).
    :param idf: dict, with the inverse document frequency for each term.
    :param tf: dict, with the term frequency for each term in each document.
    :param avg_doc_len: float, with the average document length.
    :param doc_len: list of int, with the length of each document.
    :param k1: float (parameter).
    :param b: float (parameter).
    :return: float, with the BM25 score.
    """
    score = 0
    doc_length = doc_len[document_id]
    length_norm = k1 * (1 - b + b * doc_length / avg_doc_len)  # Precompute normalization factor
    for term in set(query):  # Use set to avoid redundant term checks
        if document_id in tf and term in tf[document_id]:
            idf_term = idf.get(term, 0)  # Use .get() to handle missing terms
            tf_term = tf[document_id][term]
            score += idf_term * (tf_term * (k1 + 1) / (tf_term + length_norm))
    return score

def BM25_search(query, idf, tf, avg_doc_len, doc_len, docid, lang, target_lang, k=10):
    """
    Compute BM25 score for all documents in the corpus for a given query and language and return the top-k documents
    """
    # Filter docid and doc_len by language once
    relevant_docs = [i for i in range(len(docid)) if lang[i] == target_lang]
    # Calculate scores only for relevant documents
    scores = []
    for i in relevant_docs:
        score = BM25_score(query, i, idf, tf, avg_doc_len, doc_len)
        scores.append((score, docid[i]))

    # Sort and get top-k documents by score
    scores.sort(key=lambda x: -x[0])
    top_doc_ids = [doc_id for _, doc_id in scores[:k]]
    
    return top_doc_ids