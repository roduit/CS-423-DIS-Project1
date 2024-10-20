# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-09-30 -*-
# -*- Last revision: 2024-10-20 by Vincent Roduit -*-
# -*- python version : 3.9.19 -*-
# -*- Description: Functions to calculate scores *-

#import librairies
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pandas as pd

#import files
from constants import *

def rank_results(queries, df):
    """Rank the results of the queries based on the cosine similarity

    Args:
        * queries(pd.DataFrame): The queries with their vectors.

        * df(pd.DataFrame): The corpus with the vectors.
    
    Returns:
        * list: The list of the top 10 results for each query.

        * np.array: The cosine similarities.
    """
    # Extract vectors from queries and corpus
    query_vectors = np.stack(queries['vectors'].values)
    doc_vectors = np.stack(df['vectors'].values)

    # Compute cosine similarities in one step for all query-document pairs
    similarities = cosine_similarity(query_vectors, doc_vectors)

    results = []

    for i, row in enumerate(queries.iterrows()):
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

def bm25_score(query, document_id, idf, tf,length_norm, k1=1.5, b=0.75):
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
    if document_id not in tf:
        return 0
    
    score = 0

    query_terms = set(query)  # Use set to avoid redundant term checks
    for term in query_terms:
        if term in tf[document_id]:
            idf_term = idf[term]
            tf_term = tf[document_id][term]
            score += idf_term * (tf_term * (k1 + 1) / (tf_term + length_norm))
    return score

def recall_at_k(results, relevant_docs, k=10):
    """Compute the recall at k for the results

    Args:
        * results(list): The list of the top 10 results for each query.

        * relevant_docs(dict): The relevant documents for each query.

        * k(int): The number of results to consider. Defaults to 10.
    
    Returns:
        * float: The recall at k.
    """
    if relevant_docs in results:
        return 1.0
    else:
        return 0.0

def evaluate_recall_at_k(submission_name, queries_path):
    df_results = pd.read_csv(os.path.join(SUBMISSIONS_FOLDER, submission_name))
    df_queries = pd.read_csv(queries_path)
    df_results = df_results.merge(df_queries, left_index=True, right_index=True)
    predictions = df_results['docids'].tolist()
    ground_truth = df_results['positive_docs'].tolist()
    recalls = []
    for i in range(len(predictions)):
        pred = predictions[i]
        gt = ground_truth[i]
        rec = recall_at_k(pred, gt, 10)
        recalls.append(rec)

    print(f"Recall@10: {np.mean(recalls):.2f}")

def evaluate_recall_at_k_per_lang(submission_name, queries_path):
    df_queries = pd.read_csv(queries_path)
    df_results = pd.read_csv(os.path.join(SUBMISSIONS_FOLDER, submission_name))
    df_results = df_results.merge(df_queries, left_index=True, right_index=True)
    langs = df_results['lang'].unique()
    recalls = {}
    for lang in langs:
        df_lang = df_results[df_results['lang'] == lang]
        predictions = df_lang['docids'].tolist()
        ground_truth = df_lang['positive_docs'].tolist()
        recalls[lang] = []
        for i in range(len(predictions)):
            pred = predictions[i]
            gt = ground_truth[i]
            rec = recall_at_k(pred, gt, 10)
            recalls[lang].append(rec)
        
        print(f"Recall@10 for {lang}: {np.mean(recalls[lang]):.2f}")