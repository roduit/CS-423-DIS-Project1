# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-09-30 -*-
# -*- Last revision: 2024-10-21 by Vincent Roduit -*-
# -*- python version : 3.9.19 -*-
# -*- Description: Class for the processing of the corpus *-

#import libraries
import os
import pandas as pd
from tqdm import tqdm
from collections import Counter
import math
from time import time
import multiprocessing as mp
from collections import defaultdict

#import files
from utils import *
from constants import *
from processing import *
from scores import bm25_score
from corpus_base import CorpusBase

class CorpusBm25(CorpusBase):
    def __init__(
            self, corpus_path: str, 
            query_path: str, 
            k1:float=1.5, 
            b:float=0.75, 
            filter:bool=False,
            filt_docs:int=10000,
            verbose:bool=True):
        """
        Initialize the CorpusBM25 object.

        Args:
            * corpus_path: str, the path to the corpus file.

            * query_path: str, the path to the query file.

            * k1: float, the BM25 parameter k1. Defaults to 1.5.

            * b: float, the BM25 parameter b. Defaults to 0.75.

            * filter: bool, whether to filter the results or not. Defaults to False.

            * filt_docs: int, the number of documents to filter. Defaults to 10000.

            * verbose: bool, whether to print the progress or not. Defaults to False.

        Class attributes:
            * tf (dict): the term frequency for each term in each document.

            * idf (dict): the inverse document frequency for each term.

            * df (dict): the document frequency for each term.

            * avg_doc_len (float): the average document length.

            * doc_len (list): the length of each document in the corpus.

            * results (list): the results of the queries.

            * k1 (float): the BM25 parameter k1.

            * b (float): the BM25 parameter b.

            * filter (bool): whether to filter the results or not.

            * filt_docs (int): the number of documents to filter.

            * inverted_index (dict): the inverted index of the corpus.

            * term_to_id (dict): the mapping of terms to IDs.

            * time (float): the time taken to process the queries and calculate the BM25 scores.

        """
        super().__init__(corpus_path, query_path)
        self.tf = None
        self.idf = None
        self.df = None
        self.avg_doc_len = None
        self.doc_len = None
        self.results = None
        self.inverted_index = None
        self.term_to_id = None
        self.k1 = k1
        self.b = b
        self.filter = filter
        self.filt_docs = int(filt_docs)
        self.time = None
        self.verbose = verbose

    def _compute_df(self):
        """Compute the document frequency for each term in the corpus (i.e., the number of documents in which the term appears).
        """
        if self.df is None:
            if os.path.exists(os.path.join(PICKLES_FOLDER, self.corpus_file_name + "_df.pkl")) \
                and os.path.exists(os.path.join(PICKLES_FOLDER, self.corpus_file_name + "_term_to_id.pkl")):
                if self.verbose:
                    print("Loading df from pickle")
                self.df = load_data(self.corpus_file_name + "_df.pkl", PICKLES_FOLDER)
                self.term_to_id = load_data(self.corpus_file_name + "_term_to_id.pkl", PICKLES_FOLDER)
            
            else:
                if self.corpus is None:
                    self.load_corpus()
                if 'tokens' not in self.corpus.columns:
                    self.tokenize_corpus()
                if self.term_to_id is None:
                    self.term_to_id = create_term_to_id(self.corpus['tokens'].tolist())
                    save_data(self.term_to_id, self.corpus_file_name + "_term_to_id.pkl", PICKLES_FOLDER)
                self.corpus['tokens'] = self.corpus['tokens'].apply(lambda x: transform_query_to_int(x, self.term_to_id))
                if self.verbose:
                    print("Computing df")
                corpus_tokenized = self.corpus['tokens'].tolist()
                self.df = Counter(term for document in corpus_tokenized for term in set(document))
                save_data(self.df, self.corpus_file_name + "_df.pkl", PICKLES_FOLDER)   
    
    def _compute_idf(self):
        """Compute the inverse document frequency for each term in the corpus.
        """
        if self.idf is None:
            if os.path.exists(os.path.join(PICKLES_FOLDER, self.corpus_file_name + "_idf.pkl")):
                if self.verbose:
                    print("Loading idf from pickle")
                self.idf = load_data(self.corpus_file_name + "_idf.pkl", PICKLES_FOLDER)    
            else:
                if self.df is None:
                    self._compute_df()
                if self.corpus is None:
                    self.load_corpus()
                if self.verbose:
                    print("Computing idf")
                num_documents = len(self.corpus)
                self.idf = {term: math.log(1 + (num_documents - self.df[term] + 0.5) / (self.df[term] + 0.5)) for term in self.df}
                save_data(self.idf, self.corpus_file_name + "_idf.pkl", PICKLES_FOLDER)
    
    def _compute_tf(self):
        """Compute the term frequency for each term in each document.
        """
        if self.tf is None:
            if os.path.exists(os.path.join(PICKLES_FOLDER, self.corpus_file_name + "_tf.pkl")):
                if self.verbose:
                    print("Loading tf from pickle")
                self.tf = load_data(self.corpus_file_name + "_tf.pkl", PICKLES_FOLDER)
            else:
                if self.corpus is None:
                    self.load_corpus()
                if 'tokens' not in self.corpus.columns:
                    self.tokenize_corpus()
                if self.verbose:
                    print("Computing tf")
                corpus_tokenized = self.corpus['tokens'].tolist()
                self.tf = {i: dict(Counter(document)) for i, document in enumerate(corpus_tokenized)}
                save_data(self.tf, self.corpus_file_name + "_tf.pkl", PICKLES_FOLDER)
    
    def _compute_doc_len(self):
        """Compute the length of each document in the corpus.
        """
        if self.doc_len is None or self.avg_doc_len is None:
            if os.path.exists(os.path.join(PICKLES_FOLDER, self.corpus_file_name + "_doc_len.pkl")) \
                and os.path.exists(os.path.join(PICKLES_FOLDER, self.corpus_file_name + "_avg_doc_len.pkl")):
                if self.verbose:
                    print("Loading doc_len from pickle")
                self.doc_len = load_data(self.corpus_file_name + "_doc_len.pkl", PICKLES_FOLDER)
                self.avg_doc_len = load_data(self.corpus_file_name + "_avg_doc_len.pkl", PICKLES_FOLDER)
            else:
                if self.corpus is None:
                    self.load_corpus()
                if 'tokens' not in self.corpus.columns:
                    self.tokenize_corpus()
                if self.verbose:
                    print("Computing doc_len")
                self.doc_len = [len(document) for document in self.corpus['tokens'].tolist()]
                self.avg_doc_len = sum(self.doc_len) / len(self.doc_len)
                save_data(self.doc_len, self.corpus_file_name + "_doc_len.pkl", PICKLES_FOLDER)
                save_data(self.avg_doc_len, self.corpus_file_name + "_avg_doc_len.pkl", PICKLES_FOLDER)

    def _compute_length_norm(self):
        """Compute the length normalization factor for a given document length.

        Args:
            * doc_len (int): the length of the document.

            * avg_doc_len (float): the average document length.

            * k1 (float): the BM25 parameter k1. Defaults to 1.5.

            * b (float): the BM25 parameter b. Defaults to 0.75.
        """
        if self.doc_len is None:
            self._compute_doc_len()
        if self.avg_doc_len is None:
            self._compute_doc_len()
        if self.verbose:
            print("Computing length_norm")
        self.length_norm = [self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_len) for doc_length in self.doc_len]
    
    def _compute_inverted_index(self):
        if os.path.exists(os.path.join(PICKLES_FOLDER, self.corpus_file_name + "_inverted_index.pkl")):
            if self.verbose:
                print("Loading inverted index from pickle")
            self.inverted_index = load_data(self.corpus_file_name + "_inverted_index.pkl", PICKLES_FOLDER)
        else:
            if self.tf is None:
                self._compute_tf()
            self.inverted_index = {}
            if self.verbose:
                for doc_id, doc in tqdm(self.tf.items(), total=len(self.tf), desc="Computing inverted index"):
                    for word, _ in doc.items():
                        if word not in self.inverted_index:
                            self.inverted_index[word] = []
                        self.inverted_index[word].append(doc_id)
            else:
                for doc_id, doc in self.tf.items():
                    for word, _ in doc.items():
                        if word not in self.inverted_index:
                            self.inverted_index[word] = []
                        self.inverted_index[word].append(doc_id)
            save_data(self.inverted_index, self.corpus_file_name + "_inverted_index.pkl", PICKLES_FOLDER)
    
    def _BM25_search(self,query, docid,relevant_docs, k=10):
        """Compute BM25 score for all documents in the corpus for a given query and language and return the top-k documents

        Args:
            * query (list): the tokenized query.

            * docid (list): the list of document IDs.

            * relevant_docs (list): the list of relevant document IDs.

            * k (int): the number of documents to return.

        Returns:
            * top_doc_ids (list): the list of document IDs with the highest BM25 scores.
        """

        # Calculate scores only for relevant documents
        scores = []
        for i in relevant_docs:
            length_norm = self.length_norm[i]
            score = bm25_score(query, i, self.idf, self.tf, length_norm, self.k1)
            scores.append((score, docid[i]))

        # Sort and get top-k documents by score
        scores.sort(key=lambda x: -x[0])
        top_doc_ids = [doc_id for _, doc_id in scores[:k]]
        
        return top_doc_ids
    
    def _get_relevant_docs(self, query, relevant_docs, k=10000):
        query_test = set(query)
        rel_docs = defaultdict(int)
        for word in query_test:
            if word in self.inverted_index:
                for doc_id in self.inverted_index[word]:
                    rel_docs[doc_id] += 1
        sorted_rel_docs = np.array(sorted(rel_docs.items(), key=lambda x: x[1], reverse=True))
        rel_docs_ids = np.intersect1d(sorted_rel_docs, relevant_docs)
        rel_docs_ids = rel_docs_ids[:k] if len(rel_docs_ids) > k else rel_docs_ids
        return rel_docs_ids

    
    def get_results(self):
        """Get the results of the queries
        """
        self.results = []

        #initialize the idf, tf, avg_doc_len, doc_len
        if self.verbose:
            print("Computing idf, tf, avg_doc_len, doc_len")
        self._compute_df()
        self._compute_idf()
        self._compute_tf()
        self._compute_doc_len()
        self._compute_length_norm()
        if self.filter:
            self._compute_inverted_index()
        
        if os.path.exists(os.path.join(PICKLES_FOLDER, self.corpus_file_name + "_docid.pkl")):
            if self.verbose:
                print("Loading docid from pickle")
            docid = load_data(self.corpus_file_name + "_docid.pkl", PICKLES_FOLDER)
        if os.path.exists(os.path.join(PICKLES_FOLDER, self.corpus_file_name + "_lang.pkl")):
            if self.verbose:
                print("Loading lang from pickle")
            lang = load_data(self.corpus_file_name + "_lang.pkl", PICKLES_FOLDER)
        else:
            #load the corpus
            if self.corpus is None:
                self.load_corpus()

            #extract list of docid, lang and tokenized text from the corpus
            docid = self.corpus['docid'].tolist()
            save_data(docid, self.corpus_file_name + "_docid.pkl", PICKLES_FOLDER)
            lang = self.corpus['lang'].tolist()
            save_data(lang, self.corpus_file_name + "_lang.pkl", PICKLES_FOLDER)

        # Create a dictionary with the relevant documents for each language
        langs = set(lang)
        dict_relevant_docs = {l: [i for i in range(len(docid)) if lang[i] == l] for l in langs}

        start = time()

        # Load the queries
        if self.query is None:
            self.load_query()
        if 'tokens' not in self.query.columns:
            self.tokenize_query()
            self.query['tokens'] = self.query['tokens'].apply(lambda x: transform_query_to_int(x, self.term_to_id))

        #extract list of tokenized text and lang from the test queries
        list_test_queries = self.query["tokens"].tolist()
        list_lang_test_queries = self.query["lang"].tolist()

        # Loop over each query and calculate the BM25 scores
        if self.verbose:
            for idx, query in tqdm(enumerate(list_test_queries), total=len(list_test_queries), desc="Calculating BM25 scores"):
                query_lang = list_lang_test_queries[idx]  # Get the language for the current query
                
                # Get the top 10 documents for the current query
                relevant_docs = np.array(dict_relevant_docs[query_lang])
                if self.filter:
                    relevant_docs = self._get_relevant_docs(query, relevant_docs, self.filt_docs)
                top_docs = self._BM25_search(query, docid,relevant_docs, k=10)
                
                # Append the result as a dictionary
                self.results.append({
                    'id': idx,  # You may replace idx with actual query ID if available
                    'docids': top_docs
                })  
        else:
            for idx, query in enumerate(list_test_queries):
                query_lang = list_lang_test_queries[idx]  # Get the language for the current query
                
                # Get the top 10 documents for the current query
                relevant_docs = np.array(dict_relevant_docs[query_lang])
                if self.filter:
                    relevant_docs = self._get_relevant_docs(query, relevant_docs, self.filt_docs)
                top_docs = self._BM25_search(query, docid,relevant_docs, k=10)
                
                # Append the result as a dictionary
                self.results.append({
                    'id': idx,  # You may replace idx with actual query ID if available
                    'docids': top_docs
                }) 
        end = time()
        if self.verbose:
            #print minutes and seconds
            print(f"Time taken to process queries and compute BM25 scores: {int((end - start) / 60)} min {int((end - start) % 60)} sec")
        self.time = end - start

    def create_submission(self, output_path: str):
        """ Create a submission file for the BM25 model.

        Args:
            * output_path (str): the path to the output file.
        """
        if self.results is None:
            self.get_results()
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(output_path, index=False)