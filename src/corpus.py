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

#import files
from utils import *
from constants import *
from processing import *
from scores import *

class Corpus:
    def __init__(self, corpus_path: str, query_path: str):
        self.corpus = None
        self.corpus_path = corpus_path
        self.file_name = corpus_path.split('/')[-1].split('.')[0]
        self.query_file_name = query_path.split('/')[-1].split('.')[0]
        self.tokens_list = None 
        self.w2v_model = None
        self.results = None
        self.query = None
        self.query_tokens_list = None
        self.similarities = None
        self.query_path = query_path

    def load_corpus(self):
        """
        Load the corpus
        """
        if os.path.exists(os.path.join(PICKLES_FOLDER, self.file_name + ".pkl")):
            print("Loading corpus from pickle")
            self.corpus = load_data(self.file_name + ".pkl", PICKLES_FOLDER)
        else:
            print(f"Loading corpus from {self.corpus_path}")
            if '.csv' in self.corpus_path:
                self.corpus = pd.read_csv(self.corpus_path)
            elif '.json' in self.corpus_path:
                self.corpus = pd.read_json(self.corpus_path)
            else:
                raise ValueError("The file format is not supported")
            save_data(self.corpus, self.file_name + ".pkl", PICKLES_FOLDER)
    
    def load_query(self):
        """
        Load the query
        """
        if os.path.exists(os.path.join(PICKLES_FOLDER, self.query_file_name + ".pkl")):
            print("Loading query from pickle")
            self.query = load_data(self.query_file_name + ".pkl", PICKLES_FOLDER)
        else:
            print(f"Loading query from {self.query_path}")
            if '.csv' in self.query_path:
                self.query = pd.read_csv(self.query_path)
            elif '.json' in self.query_path:
                self.query = pd.read_json(self.query_path)
            else:
                raise ValueError("The file format is not supported")
            save_data(self.query, self.query_file_name + ".pkl", PICKLES_FOLDER)

    def tokenize_corpus(self):
        """Tokenize the corpus
        """
        if self.corpus is None:
            self.load_corpus()
        self.corpus, self.tokens_list = tokenize_documents(self.file_name, self.corpus)
    
    def tokenize_query(self):
        """Tokenize the query
        """
        if self.query is None:
            self.load_query()
        if 'query' in self.query.columns:
            self.query.rename(columns={'query': 'text'}, inplace=True)
        self.query, self.query_tokens_list = tokenize_documents(self.query_file_name, self.query)  
        
    
    def create_word2vec(self, model_path:str=BASE_MODEL_PATH, model_name:str=BASE_MODEL_NAME):
        """Create the Word2Vec model

        Args:
            * model_path(str): The path to save the model

            * model_name(str): The name of the model

        Returns:
            * Word2Vec: The Word2Vec model
        """
        if not os.path.exists(os.path.join(model_path, model_name)):
            if self.tokens_list is None:
                self.tokenize_corpus()
            self.model_name = model_name
            self.w2v_model = create_word2vec_model(model_path, model_name, self.tokens_list)
        else:
            print("Word2Vec model already exists, loading it...")
            self.w2v_model = Word2Vec.load(os.path.join(model_path, model_name))

    def vectorize_corpus(self):
        """Vectorize the corpus
        """
        if os.path.exists(os.path.join(PICKLES_FOLDER, self.file_name + "_vectors.pkl")):
            print("Loading vectors from pickle")
            self.corpus =  load_data(self.file_name + "_vectors.pkl", PICKLES_FOLDER)
        else:
            print("Vectorizing corpus")
            if self.tokens_list is None:
                self.tokenize_corpus()
            if self.w2v_model is None:
                self.create_word2vec()
            self.corpus = vectorize_documents(self.file_name, self.w2v_model, self.corpus)
    
    def vectorize_query(self):
        """Vectorize the query
        """
        if os.path.exists(os.path.join(PICKLES_FOLDER, self.query_file_name + "_vectors.pkl")):
            print("Loading vectors from pickle")
            self.query =  load_data(self.query_file_name + "_vectors.pkl", PICKLES_FOLDER)
        else:
            print("Vectorizing query")
            if self.query_tokens_list is None:
                self.tokenize_query()
            if self.w2v_model is None:
                self.create_word2vec()
            self.query = vectorize_documents(self.query_file_name, self.w2v_model, self.query)
        
    def rank_results(self):
        """
        Rank the results of the queries
        """
        if self.corpus is None:
            self.vectorize_corpus()
        if self.query is None:
            self.vectorize_query()
        print("Ranking results")
        self.results, self.similarities = rank_results(self.query, self.corpus)

    def create_submission(self, output_path:str):
        """
        Create the submission file

        Args:
            * output_path(str): The path to save the submission file
        """
        if self.results is None:
            self.rank_results()
        submission = pd.DataFrame(self.results)
        submission.to_csv(output_path, index=False)
        print(f"Submission file saved at {output_path}")
