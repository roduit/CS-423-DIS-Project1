# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-09-30 -*-
# -*- Last revision: 2024-10-17 by Vincent Roduit -*-
# -*- python version : 3.9.19 -*-
# -*- Description: Functions to calculate scores *-

#import files
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def rank_results(queries, df):
    """Rank the results of the queries based on the cosine similarity

    Args:
        * queries(pd.DataFrame): The queries with their vectors.

        * df(pd.DataFrame): The corpus with the vectors.
    
    Returns:
        * list: The list of the top 10 results for each query.
    """
    # Extract vectors from queries and corpus
    query_vectors = np.stack(queries['vectors'].values)
    doc_vectors = np.stack(df['vectors'].values)

    # Compute cosine similarities in one step for all query-document pairs
    similarities = cosine_similarity(query_vectors, doc_vectors)

    results = []

    for i, row in tqdm(enumerate(queries.iterrows()), total=len(queries)):
        id = row[0]
        # Get similarities for the current query and sort them
        similarity_scores = similarities[i]
        top10_indices = np.argsort(similarity_scores)[::-1][:10]
        top10_docids = df.iloc[top10_indices]['docid'].tolist()

        top10_results = {
            "id": id,
            "docids": top10_docids
        }
        results.append(top10_results)
    
    return results, similarities

def bm25_score(query, document_id, idf, tf, avg_doc_len, doc_len, k1=1.5, b=0.75):
    """Compute the BM25 score for a given query and the document position in the corpus

    Args:
        * query(list): The list of query terms.

        * document_id(int): The document position in the corpus.

        * idf(dict): The inverse document frequency of the terms.

        * tf(dict): The term frequency of the terms in the documents.

        * avg_doc_len(float): The average document length.

        * doc_len(dict): The length of the documents.

        * k1(float): The BM25 parameter k1. Defaults to 1.5.

        * b(float): The BM25 parameter b. Defaults to 0.75.
    
    Returns:
        * float: The BM25 score.
    """
    score = 0
    doc_length = doc_len[document_id]
    length_norm = k1 * (1 - b + b * doc_length / avg_doc_len)  # Precompute normalization factor

    query_terms = set(query)  # Use set to avoid redundant term checks
    for term in query_terms:
        if document_id in tf and term in tf[document_id]:
            idf_term = idf.get(term, 0)  # Use .get() to handle missing terms
            tf_term = tf[document_id][term]
            score += idf_term * (tf_term * (k1 + 1) / (tf_term + length_norm))
    return score