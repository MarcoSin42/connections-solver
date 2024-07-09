"""
A validation script.  Validates our model to see if it works.  Outputs a confusion matrix and an accuracy value
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from k_means_constrained import KMeansConstrained

from preprocess import parse_json
from utils import get_vectors, permute_to_minimize_cmatrix

import gensim.downloader as api
import gensim

if __name__ == '__main__':
    # Load model
    path = api.load("word2vec-google-news-300", return_path=True)
    model = gensim.models.keyedvectors.load_word2vec_format(path, binary=True)
    
    # Parse json
    df = parse_json()
    
    # Actual labels, all groups are labeled sequentially in order in the dataset
    actual = sum([[i]*4 for i in range(4)],[])
    df['actual'] = [actual for i in df.index] 
    
    kmeans_const = KMeansConstrained(
        n_clusters=4,
        size_min=4,
        size_max=4,
        random_state=0
    )
    
    # Prediction labels
    df['predicted_vec'] =  df['answers'].apply(lambda x: get_vectors(x, model))
    
    
    
    
    print(df)