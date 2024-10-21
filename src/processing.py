# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-09-30 -*-
# -*- Last revision: 2024-10-21 by Vincent Roduit -*-
# -*- python version : 3.9.19 -*-
# -*- Description: Functions for processing *-

# import libraries
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from time import time
from tqdm import tqdm
import pandas as pd 

#import libraries
from constants import*
from utils import*

tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()

def tokenize(text:str, lang:str="en") -> list:
    """Tokenizes and stems the input text efficiently.

    Args:
        * text(str): The text to tokenize.

        * lang(str): The language of the text. Defaults to "en".

    Returns:
        * list: The list of stemmed tokens.
    """
    
    tokens = tokenizer.tokenize(text)
    
    # Combine stemming and stopword filtering into one pass for efficiency
    return [stemmer.stem(word.lower()) for word in tokens if word.lower() not in STOP_WORDS[lang] or stemmer.stem(word.lower()) not in STOP_WORDS[lang]]

def tokenize_documents(file_name:str, corpus:pd.DataFrame, drop_text:bool=True, verbose:bool=False) -> pd.DataFrame:
    """Tokenize the corpus

    Args:
        * file_name(str): The name of the file to save the tokenized corpus.

        * corpus(pd.DataFrame): The corpus to tokenize.
    
    Returns:
        * pd.DataFrame: The tokenized corpus.
    """
    tqdm.pandas() 
    if os.path.exists(os.path.join(PICKLES_FOLDER, file_name + "_tokenized.pkl")) and os.path.exists(os.path.join(PICKLES_FOLDER, file_name + "_tokens_list.pkl")):
        if verbose:
            print("Loading tokenized corpus from pickle")
        corpus = load_data(file_name + "_tokenized.pkl", PICKLES_FOLDER)
        tokens_list = load_data(file_name + "_tokens_list.pkl", PICKLES_FOLDER)
    else: 
        if verbose:
            print("Tokenizing corpus")
        corpus["tokens"] = corpus.progress_apply(lambda row: tokenize(row['text'], lang=row['lang']), axis=1)
        if drop_text:
            corpus.drop(columns=["text"], inplace=True)
        tokens_list = corpus["tokens"].tolist()
        save_data(corpus, file_name + "_tokenized.pkl", PICKLES_FOLDER)
        save_data(tokens_list, file_name + "_tokens_list.pkl", PICKLES_FOLDER)
    return corpus, tokens_list

def get_vectors(words:list, model:Word2Vec) -> np.array:
    """Get the vectors of the words from the model

    Args:
        * words(list): The list of words to get the vectors of.

        * model(Word2Vec): The Word2Vec model.
    
    Returns:
        * np.array: The vectors of the words.
    """
    vectors = []
    for word in words:
        try:
            vectors.append(model.wv[word])
        except KeyError:
            pass
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def vectorize_documents(file_name, w2v_model, corpus, verbose) -> np.array:
    """Get the vectors of the words from the model

    Args:
        * words(list): The list of words to get the vectors of.

    Returns:
        * np.array: The vectors of the words.
    """
    if verbose:
        tqdm.pandas()
        print("Getting vectors")
        corpus["vectors"] = corpus.progress_apply(lambda row: get_vectors(row['tokens'], w2v_model), axis=1)
    else:
        corpus["vectors"] = corpus.apply(lambda row: get_vectors(row['tokens'], w2v_model), axis=1)
    save_data(corpus, file_name + "_vectors.pkl", PICKLES_FOLDER)

    return corpus

def create_word2vec_model(
        model_path:str='word2vec_stem', 
        model_name:str='wor2vec.model', 
        tokens_list:list=None,
        min_count:int=1, 
        window:int=5, 
        vector_size:int=100, 
        workers:int=CORES-1, 
        epochs:int=100,
        verbose:bool=False):
    """Create a Word2Vec model from the tokens list and save it to a file.

    Args:
        * model_path(str): The path to save the model file.

        * model_name(str): The name of the model file.

        * tokens_list(list): The list of tokens to create the model from. Defaults to None.

        * min_count(int): Ignores all words with total frequency lower than this. Defaults to 1.

        * window(int): The maximum distance between the current and predicted word within a sentence. Defaults to 5.

        * vector_size(int): Dimensionality of the word vectors. Defaults to 100.

        * workers(int): The number of worker threads to train the model. Defaults to cores-1.

        * epochs(int): Number of iterations (epochs) over the corpus. Defaults to 100.

        * verbose(bool): Whether to print the progress or not. Defaults to False.
    
    Returns:
        * Word2Vec: The Word2Vec model.
    """
    if tokens_list is None:
        raise ValueError("tokens_list must be provided to create the model")
    if verbose:
        print(f"Creating Word2Vec model with min_count={min_count}, window={window}, vector_size={vector_size}, workers={workers}, epochs={epochs}")
    w2v_model = Word2Vec(min_count=min_count,
                        window=window,
                        vector_size=vector_size,
                        workers=workers)
    
    t = time()
    w2v_model.build_vocab(tokens_list, progress_per=10000)
    if verbose:
        print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
    
    t = time()
    w2v_model.train(tokens_list, total_examples=w2v_model.corpus_count, epochs=epochs, report_delay=1)
    if verbose:
        print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
    
    # Save the model
    w2v_model.save(os.path.join(model_path, model_name))
    if verbose:
        print(f"Word2Vec model saved as {model_name} at {model_path}")
        
    return w2v_model