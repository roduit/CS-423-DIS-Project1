# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-09-30 -*-
# -*- Last revision: 2024-10-17 by Vincent Roduit -*-
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

class CorpusBase:
    def __init__(self, corpus_path: str, query_path: str):
        """ Initialize the CorpusBase object.
        Args:
            * corpus_path (str): the path to the corpus file.

            * query_path (str): the path to the query file.

        Class attributes:
            * corpus (pd.DataFrame): the corpus.

            * corpus_path (str): the path to the corpus file.

            * corpus_file_name (str): the name of the corpus file.

            * query_file_name (str): the name of the query file.

            * tokens_list (list): the tokens list.

            * results (list): the results of the queries.

            * query (pd.DataFrame): the query.

            * query_tokens_list (list): the query tokens list.

            * query_path (str): the path to the query file.
        """
        self.corpus = None
        self.corpus_path = corpus_path
        self.corpus_file_name = corpus_path.split('/')[-1].split('.')[0]
        self.query_file_name = query_path.split('/')[-1].split('.')[0]
        self.tokens_list = None 
        self.results = None
        self.query = None
        self.query_tokens_list = None
        self.query_path = query_path

    def load_corpus(self):
        """Load the corpus
        """
        if os.path.exists(os.path.join(PICKLES_FOLDER, self.corpus_file_name + ".pkl")):
            print("Loading corpus from pickle")
            self.corpus = load_data(self.corpus_file_name + ".pkl", PICKLES_FOLDER)
        else:
            print(f"Loading corpus from {self.corpus_path}")
            if '.csv' in self.corpus_path:
                self.corpus = pd.read_csv(self.corpus_path)
            elif '.json' in self.corpus_path:
                self.corpus = pd.read_json(self.corpus_path)
            else:
                raise ValueError("The file format is not supported")
            save_data(self.corpus, self.corpus_file_name + ".pkl", PICKLES_FOLDER)
    
    def load_query(self):
        """Load the query
        """
        if os.path.exists(os.path.join(PICKLES_FOLDER, self.query_file_name + ".pkl")):
            print("Loading query from pickle")
            self.query = load_data(self.query_file_name + ".pkl", PICKLES_FOLDER)
        else:
            print(f"Loading query from {self.query_path}")
            if '.csv' in self.query_path:
                self.query = pd.read_csv(self.query_path)

    def tokenize_corpus(self, drop_text=True):
        """Tokenize the corpus

        Args:
            * drop_text (bool): whether to drop the text column or not.
        """
        if self.corpus is None:
            self.load_corpus()
        self.corpus, self.tokens_list = tokenize_documents(self.corpus_file_name, self.corpus, drop_text)
    
    def tokenize_query(self, drop_text=True):
        """Tokenize the query

        Args:
            * drop_text (bool): whether to drop the text column or not.
        """
        if self.query is None:
            self.load_query()
        if 'query' in self.query.columns:
            self.query.rename(columns={'query': 'text'}, inplace=True)
        self.query, self.query_tokens_list = tokenize_documents(self.query_file_name, self.query, drop_text) 