# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-09-30 -*-
# -*- Last revision: 2024-10-16 by Vincent Roduit -*-
# -*- python version : 3.9.19 -*-
# -*- Description: Constants used in the code *-

#import files
import os
import multiprocessing

CORES = multiprocessing.cpu_count()

# Path to the data folder
DATA_FOLDER = "../data"

#Path for the models
MODEL_FOLDER = os.path.join(DATA_FOLDER, "models")
BASE_MODEL_PATH = os.path.join(MODEL_FOLDER, "word2vec_stem")
BASE_MODEL_NAME = "word2vec.model"

#Path for pickles
PICKLES_FOLDER = os.path.join(DATA_FOLDER, "pickles")

#Path for the stopwords
STOPWORDS_FOLDER = os.path.join(DATA_FOLDER, "stopwords")

#Path for the submissions
SUBMISSIONS_FOLDER = os.path.join(DATA_FOLDER, "submissions")


if not os.path.exists("../data/models"):
    os.mkdir("../data/models")

if not os.path.exists("../data"):
    os.mkdir("../data")

if not os.path.exists("../data/pickles"):
    os.mkdir("../data/pickles")

if not os.path.exists("../data/submissions"):
    os.mkdir("../data/submissions")



def load_stopwords(path):
    with open(path, 'r') as f:
        arabic_stopwords = f.read().splitlines()
    return arabic_stopwords

STOP_WORDS = {
    "en": set(load_stopwords(os.path.join(STOPWORDS_FOLDER, "english"))),
    "fr": set(load_stopwords(os.path.join(STOPWORDS_FOLDER, "french"))),
    "de": set(load_stopwords(os.path.join(STOPWORDS_FOLDER, "german"))),
    "es": set(load_stopwords(os.path.join(STOPWORDS_FOLDER, "spanish"))),
    "it": set(load_stopwords(os.path.join(STOPWORDS_FOLDER, "italian"))),
    "ko": set(load_stopwords(os.path.join(STOPWORDS_FOLDER, "korean"))),
    "ar": set(load_stopwords(os.path.join(STOPWORDS_FOLDER, "arabic"))),
}

#Path for the corpus
CORPUS = os.path.join(DATA_FOLDER, "corpus", "corpus.json")
CORPUS_REDUCED = os.path.join(DATA_FOLDER, "corpus", "corpus_reduced.json")
QUERIES = os.path.join(DATA_FOLDER, "test.csv")