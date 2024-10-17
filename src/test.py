import multiprocessing as mp
from tqdm import tqdm

class BM25Search:
    def __init__(self, idf, tf, length_norm, k1=1.5, b=0.75):
        self.idf = idf
        self.tf = tf
        self.length_norm = length_norm
        self.k1 = k1
        self.b = b
        self.results = []

    def run_parallel_BM25(self, list_test_queries, list_lang_test_queries, dict_relevant_docs, docid, num_processes=4):
        """Run BM25 search in parallel using multiprocessing."""
        
        # Prepare data for parallel processing
        query_doc_tuples = [
            (list_test_queries[idx], dict_relevant_docs[list_lang_test_queries[idx]], docid)
            for idx in range(len(list_test_queries))
        ]
        
        # Use multiprocessing to distribute the workload
        with mp.Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(self.bm25_search_parallel, query_doc_tuples), 
                                total=len(list_test_queries), 
                                desc="Calculating BM25 scores"))

        # Store results
        for idx, result in enumerate(results):
            self.results.append({
                'id': idx,  # You may replace idx with actual query ID if available
                'docids': result['docids']
            })
        return self.results

    def bm25_search_parallel(self, query_doc_tuple):
        """Helper function for parallel BM25 search."""
        query, relevant_docs, docid = query_doc_tuple
        top_docs = self._BM25_search(query, docid, relevant_docs, k=10)
        return {'docids': top_docs}

    def _BM25_search(self, query, docid, relevant_docs, k=10):
        """Compute BM25 score for all documents in the corpus for a given query and language."""
        scores = []
        for i in relevant_docs:
            length_norm = self.length_norm[i]
            score = bm25_score(query, i, self.idf, self.tf, length_norm, self.k1, self.b)
            scores.append((score, docid[i]))

        # Sort and get top-k documents by score
        scores.sort(key=lambda x: -x[0])
        top_doc_ids = [doc_id for _, doc_id in scores[:k]]
        
        return top_doc_ids