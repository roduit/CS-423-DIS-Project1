# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-09-30 -*-
# -*- Last revision: 2024-10-15 by Vincent Roduit -*-
# -*- python version : 3.9.19 -*-
# -*- Description: Constants used in the code *-

#import files
import os

# Path to the data folder
DATA_FOLDER = "../data"
if not os.path.exists("../data"):
    os.mkdir("../data")

#Path for the models
MODEL_FOLDER = os.path.join(DATA_FOLDER, "models")

#Path for pickles
PICKLES_FOLDER = os.path.join(DATA_FOLDER, "pickles")

if not os.path.exists("../data/models"):
    os.mkdir("../data/models")

#Path for the stopwords
STOPWORDS_FOLDER = os.path.join(DATA_FOLDER, "stopwords")

def load_stopwords(path):
    with open(path, 'r') as f:
        arabic_stopwords = f.read().splitlines()
    return arabic_stopwords

STOP_WORDS = {
    "en": set(),
    "fr": set(load_stopwords(os.path.join(STOPWORDS_FOLDER, "french"))),
    "de": set(load_stopwords(os.path.join(STOPWORDS_FOLDER, "german"))),
    "es": set(load_stopwords(os.path.join(STOPWORDS_FOLDER, "spanish"))),
    "it": set(load_stopwords(os.path.join(STOPWORDS_FOLDER, "italian"))),
    "ko": set(load_stopwords(os.path.join(STOPWORDS_FOLDER, "korean"))),
    "ar": set(load_stopwords(os.path.join(STOPWORDS_FOLDER, "arabic"))),
}

#Path for the corpus
CORPUS = os.path.join(DATA_FOLDER, "corpus", "corpus.json")
CORPUS_PKL = os.path.join(DATA_FOLDER, "pickles", "corpus.pkl")
CORPUS_REDUCED_PKL = os.path.join(DATA_FOLDER, "pickles", "corpus_reduced.pkl")