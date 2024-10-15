# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-09-30 -*-
# -*- Last revision: 2024-10-15 by Vincent Roduit -*-
# -*- python version : 3.9.19 -*-
# -*- Description: Class for the processing of the corpus *-

#import libraries
import os
import pandas as pd

#import files
from utils import *
from constants import *
from processing import *

class Corpus:
    def __init__(self, corpus_path: str):
        self.corpus = None
        self.corpus_path = corpus_path
        self.file_name = corpus_path.split('/')[-1].split('.')[0]
        self.corpus_reduced = None

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

    def vectorize