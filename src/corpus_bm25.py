# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-09-30 -*-
# -*- Last revision: 2024-10-15 by Vincent Roduit -*-
# -*- python version : 3.9.19 -*-
# -*- Description: Class for the processing of the corpus *-

#import libraries
import os
import pandas as pd
from tqdm import tqdm
from collections import Counter
import math


#import files
from utils import *
from constants import *
from processing import *
from scores import bm25_score
from corpus_base import CorpusBase

class CorpusBm25(CorpusBase):
    def __init__(self, corpus_path: str, query_path: str):
        super().__init__(corpus_path, query_path)
        self.tf = None
        self.idf = None
        self.df = None
        self.avg_doc_len = None
        self.doc_len = None
        self.results = None

    def _compute_df(self):
        """
        Compute the document frequency for each term in the corpus (i.e., the number of documents in which the term appears).
        """
        if self.df is None:
            if os.path.exists(os.path.join(PICKLES_FOLDER, self.corpus_file_name + "_df.pkl")):
                print("Loading df from pickle")
                self.df = load_data(self.corpus_file_name + "_df.pkl", PICKLES_FOLDER)
            else:
                if self.corpus is None:
                    self.load_corpus()
                if 'tokens' not in self.corpus.columns:
                    self.tokenize_corpus()
                corpus_tokenized = self.corpus['tokens'].tolist()
                self.df = Counter(term for document in corpus_tokenized for term in set(document))
                save_data(self.df, self.corpus_file_name + "_df.pkl", PICKLES_FOLDER)   
    
    def _compute_idf(self):
        """
        Compute the inverse document frequency for each term in the corpus.
        """
        if self.idf is None:
            if os.path.exists(os.path.join(PICKLES_FOLDER, self.corpus_file_name + "_idf.pkl")):
                print("Loading idf from pickle")
                self.idf = load_data(self.corpus_file_name + "_idf.pkl", PICKLES_FOLDER)    
            else:
                if self.df is None:
                    self._compute_df()
                num_documents = len(self.corpus)
                self.idf = {term: math.log(1 + (num_documents - self.df[term] + 0.5) / (self.df[term] + 0.5)) for term in self.df}
                save_data(self.idf, self.corpus_file_name + "_idf.pkl", PICKLES_FOLDER)
    
    def _compute_tf(self):
        """
        Compute the term frequency for each term in each document.
        """
        if self.tf is None:
            if os.path.exists(os.path.join(PICKLES_FOLDER, self.corpus_file_name + "_tf.pkl")):
                print("Loading tf from pickle")
                self.tf = load_data(self.corpus_file_name + "_tf.pkl", PICKLES_FOLDER)
            else:
                if self.corpus is None:
                    self.load_corpus()
                if 'tokens' not in self.corpus.columns:
                    self.tokenize_corpus()
                    corpus_tokenized = self.corpus['tokens'].tolist()
                    self.tf = {i: dict(Counter(document)) for i, document in enumerate(corpus_tokenized)}
                    save_data(self.tf, self.corpus_file_name + "_tf.pkl", PICKLES_FOLDER)
    
    def _compute_doc_len(self):
        """
        Compute the length of each document in the corpus.
        """
        if self.doc_len is None or self.avg_doc_len is None:
            if os.path.exists(os.path.join(PICKLES_FOLDER, self.corpus_file_name + "_doc_len.pkl")) \
                and os.path.exists(os.path.join(PICKLES_FOLDER, self.corpus_file_name + "_avg_doc_len.pkl")):
                print("Loading doc_len from pickle")
                self.doc_len = load_data(self.corpus_file_name + "_doc_len.pkl", PICKLES_FOLDER)
                self.avg_doc_len = load_data(self.corpus_file_name + "_avg_doc_len.pkl", PICKLES_FOLDER)
            else:
                if self.corpus is None:
                    self.load_corpus()
                if 'tokens' not in self.corpus.columns:
                    self.tokenize_corpus()
                self.doc_len = [len(document) for document in self.corpus['tokens'].tolist()]
                self.avg_doc_len = sum(self.doc_len) / len(self.doc_len)
                save_data(self.doc_len, self.corpus_file_name + "_doc_len.pkl", PICKLES_FOLDER)
                save_data(self.avg_doc_len, self.corpus_file_name + "_avg_doc_len.pkl", PICKLES_FOLDER)
    
    def _BM25_search(self,query, docid, lang, target_lang, k=10):
        """
        Compute BM25 score for all documents in the corpus for a given query and language and return the top-k documents
        """
        # Filter docid and doc_len by language once
        relevant_docs = [i for i in range(len(docid)) if lang[i] == target_lang]
        # Calculate scores only for relevant documents
        scores = []
        for i in relevant_docs:
            score = bm25_score(query, i, self.idf, self.tf, self.avg_doc_len, self.doc_len)
            scores.append((score, docid[i]))

        # Sort and get top-k documents by score
        scores.sort(key=lambda x: -x[0])
        top_doc_ids = [doc_id for _, doc_id in scores[:k]]
        
        return top_doc_ids
    
    def get_results(self):
        """
        Get the results of the queries
        """
        self.results = []

        #initialize the idf, tf, avg_doc_len, doc_len
        print("Computing idf, tf, avg_doc_len, doc_len")
        self._compute_df()
        self._compute_idf()
        self._compute_tf()
        self._compute_doc_len()

        # Load the queries
        if self.query is None:
            self.load_query()
        if 'tokens' not in self.query.columns:
            self.tokenize_query()
        #load the corpus
        if self.corpus is None:
            self.load_corpus()

        #extract list of docid, lang and tokenized text from the corpus
        docid = self.corpus['docid'].tolist()
        lang = self.corpus['lang'].tolist()
        #extract list of tokenized text and lang from the test queries
        list_test_queries = self.query["tokens"].tolist()
        list_lang_test_queries = self.query["lang"].tolist()

        # Loop over each query
        for idx, query in tqdm(enumerate(list_test_queries), total=len(list_test_queries), desc="Processing queries"):
            query_lang = list_lang_test_queries[idx]  # Get the language for the current query
            
            # Get the top 10 documents for the current query
            top_docs = self._BM25_search(query, docid, lang, target_lang=query_lang, k=10)
            
            # Append the result as a dictionary
            self.results.append({
                'id': idx,  # You may replace idx with actual query ID if available
                'docids': top_docs
            })   

    def create_submission(self, output_path: str):
        """
        Create a submission file for the BM25 model.
        
        :param output_path: str, the path to the output file.
        """
        if self.results is None:
            self.get_results()
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(output_path, index=False)