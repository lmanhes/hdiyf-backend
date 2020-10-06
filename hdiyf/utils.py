import numpy as np
import torch
import torch.nn as nn
from gensim.models import KeyedVectors


def calculate_maxlen(docs, percentile=90):
    return int(np.percentile(np.array([len(doc) for doc in docs]), percentile))


def init_embedding(input_embedding):
    """
    Initialize embedding tensor with values from the uniform distribution.
    :param input_embedding: embedding tensor
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)


def load_word2vec_embeddings(word2vec_file, word_map, from_object=False):
    """
    Load pre-trained embeddings for words in the word map.
    :param word2vec_file: location of the trained word2vec model
    :param word_map: word map
    :return: embeddings for words in the word map, embedding size
    """
    if from_object:
        w2v = word2vec_file
    else:
        # Load word2vec model into memory
        w2v = KeyedVectors.load_word2vec_format(word2vec_file,
                                                binary=True,
                                                unicode_errors='ignore')

    # Create tensor to hold embeddings for words that are in-corpus
    bias = np.sqrt(3.0 / w2v.vector_size)
    embeddings = np.random.uniform(-bias, bias, size=(len(word_map), w2v.vector_size))

    # Read embedding file
    for word in word_map:
        if word in w2v.vocab:
            embeddings[word_map[word]] = w2v[word]

    return embeddings, w2v.vector_size