"""
A validation script.  Validates our model to see if it works.  Outputs a confusion matrix and an accuracy value
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

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
    
    # Filter out Nones
    df = df[df['predicted_vec'].notna()]
    
    # Apply prediction
    df['predicted'] = df['predicted_vec'].apply(kmeans_const.fit_predict)
    
    # Apply permutation algorithm
    df['predicted'] = df['predicted'].apply(permute_to_minimize_cmatrix)
    df['predicted'] = df['predicted'].apply(lambda x: x[1])
    
    # Get our confusion matrix
    df['cmatrix'] = df['predicted'].apply(lambda x: confusion_matrix(actual, x))
    
    # Get our accuracy
    df['accuracy'] = df['predicted'].apply(lambda x: accuracy_score(actual, x))
    
    # Get our accuracies for each individual difficulty level
    df['lvl0'] = df['predicted'].apply(lambda x: accuracy_score(actual[0:4], x[0:4]))
    df['lvl1'] = df['predicted'].apply(lambda x: accuracy_score(actual[4:8], x[4:8]))
    df['lvl2'] = df['predicted'].apply(lambda x: accuracy_score(actual[8:12], x[8:12]))
    df['lvl3'] = df['predicted'].apply(lambda x: accuracy_score(actual[12:], x[12:]))
    
    cmatrix_sum = np.sum(df['cmatrix'])
    lvl0_mean = df['lvl0'].mean()
    lvl1_mean = df['lvl1'].mean()
    lvl2_mean = df['lvl2'].mean()
    lvl3_mean = df['lvl3'].mean()
    
    """
    print(df)
    print(cmatrix_sum)
    print(lvl0_mean)
    print(lvl1_mean)
    print(lvl2_mean)
    print(lvl3_mean)
    """
    
    # Plot our stuff and save it!
    lvl_avgs = [lvl0_mean, lvl1_mean, lvl2_mean, lvl3_mean]
    lvls = [0, 1, 2, 3]
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    
    ConfusionMatrixDisplay(cmatrix_sum).plot(ax=axs[1])
    
    axs[0].set_ylim([0, 1])
    axs[0].set_xticks(lvls, labels=['lvl0', 'lvl1', 'lvl2', 'lvl3'])
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Difficulty level")
    axs[0].bar(lvls, lvl_avgs)
    
    axs[0].title.set_text("Average scores")
    axs[1].title.set_text("Confusion Matrix")
    
    fig.savefig("Validation_plot.png")