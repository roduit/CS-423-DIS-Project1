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
from corpus_base import CorpusBase

#import files
from utils import *
from constants import *
from processing import *
from scores import *

class CorpusWord2Vec(CorpusBase):
    def __init__(self, corpus_path: str, query_path: str):
        super().__init__(corpus_path, query_path)
        self.w2v_model = None
        self.model_name = None
        self.similarities = None 

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
        if os.path.exists(os.path.join(PICKLES_FOLDER, self.corpus_file_name + "_vectors.pkl")):
            print("Loading corpus vectors from pickle")
            self.corpus =  load_data(self.corpus_file_name + "_vectors.pkl", PICKLES_FOLDER)
        else:
            print("Vectorizing corpus")
            if self.tokens_list is None:
                self.tokenize_corpus()
            if self.w2v_model is None:
                self.create_word2vec()
            self.corpus = vectorize_documents(self.corpus_file_name, self.w2v_model, self.corpus)
    
    def vectorize_query(self):
        """Vectorize the query
        """
        if os.path.exists(os.path.join(PICKLES_FOLDER, self.query_file_name + "_vectors.pkl")):
            print("Loading query vectors from pickle")
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