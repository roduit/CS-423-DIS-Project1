# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-09-30 -*-
# -*- Last revision: 2024-10-15 by Vincent Roduit -*-
# -*- python version : 3.9.19 -*-
# -*- Description: Functions to calculate scores *-

#import files
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def rank_results(queries, df):
    """
    Rank the results of the queries
    
    :param queries: pd.DataFrame, the queries
    :param df: pd.DataFrame, the corpus
    :param model: Word2Vec, the Word2Vec model

    :return: pd.DataFrame, the ranked results
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