# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-09-30 -*-
# -*- Last revision: 2024-10-15 by Vincent Roduit -*-
# -*- python version : 3.9.19 -*-
# -*- Description: Functions for processing *-

# import libraries
import numpy as np
from gensim.models import Word2Vec

def vectorize(doc:str, stopwords:set, w2v_model:Word2Vec) -> np.ndarray:
    """
    Identify the vector values for each word in the given document
    Args:
        * doc (str): the document

        * stopwords (set): the stopwords

        * w2v_model (Word2Vec): the Word2Vec model

    Returns:
        * np.ndarray: the vector values
    """
    doc = doc.lower()
    words = [w for w in doc.split(" ") if w not in stopwords]
    word_vecs = []
    for word in words:
        try:
            vec = w2v_model[word]
            word_vecs.append(vec)
        except KeyError:
            # Ignore, if the word doesn't exist in the vocabulary
            pass
    if not word_vecs:
        # If empty - return zeros
        return np.zeros(w2v_model.vector_size)
    vector = np.mean(word_vecs, axis=0)
    return vector