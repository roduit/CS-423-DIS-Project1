# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-09-30 -*-
# -*- Last revision: 2024-10-21 by Vincent Roduit -*-
# -*- python version : 3.9.19 -*-
# -*- Description: Functions to calculate scores *-

#import librairies
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pandas as pd
from typing import Tuple

#import files
from constants import *

def rank_results(queries:pd.DataFrame, df:pd.DataFrame) -> Tuple[list, np.array]:
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

def bm25_score(query:list, document_id:int, idf:dict, tf:dict,length_norm:float, k1:float=1.5) -> float:
    """Compute the BM25 score for a given query and the document position in the corpus

    Args:
        * query(list): The list of query terms.

        * document_id(int): The document position in the corpus.

        * idf(dict): The inverse document frequency of the terms.

        * tf(dict): The term frequency of the terms in the documents.

        * length_norm(float): The length normalization of the document.

        * k1(float): The BM25 parameter k1. Defaults to 1.5.
    
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

def recall_at_k(results:list, relevant_docs:str, k:int=10) -> float:
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

def evaluate_recall_at_k(submission_name:str, queries_path:str, verbose:bool=True) -> float:
    """Evaluate the recall at k for the submission

    Args:
        * submission_name(str): The name of the submission file.

        * queries_path(str): The path to the queries file.

        * verbose(bool): Whether to print the results or not. Defaults to False.

    Returns:
        * float: The average recall at k.
    """
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
    if verbose:
        print(f"Recall@10: {np.mean(recalls):.2f}")
    return np.mean(recalls)

def evaluate_recall_at_k_per_lang(submission_name:str, queries_path:str, verbose:bool=True) -> dict:
    """Evaluate the recall at k for the submission per language

    Args:
        * submission_name(str): The name of the submission file.

        * queries_path(str): The path to the queries file.

        * verbose(bool): Whether to print the results or not. Defaults to False.
    
    Returns:
        * dict: The average recall at k per language.
    """
    
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
        recalls[lang] = np.mean(recalls[lang])
        if verbose:
            print(f"Recall@10 for {lang}: {np.mean(recalls[lang]):.2f}")

    return recalls