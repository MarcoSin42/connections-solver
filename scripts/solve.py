import gensim.downloader as api
import gensim
import sys
import numpy as np
from numpy.typing import ArrayLike

from k_means_constrained import KMeansConstrained
from utils import get_vectors


if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print("Usage: python solve.py <filename>")
        print("Where filename contains line separated words")
    
    # Parse our file
    fname:str = sys.argv[1]
    words: list[str] = open(fname,'r').read().split()
    print(f"Words inputted: {words}")
    # Load model
    
    print("Loading model...  This can take a minute!")
    path = api.load("word2vec-google-news-300", return_path=True)
    model = gensim.models.keyedvectors.load_word2vec_format(path, binary=True)
    
    kmeans_const = KMeansConstrained(
        n_clusters=4,
        size_min=4,
        size_max=4,
        random_state=0
    )
    
    # Predict
    print("Doing the prediction...   This can take a while!")
    word_vecs: ArrayLike = get_vectors(words, model)
    categorized_words = kmeans_const.fit_predict(word_vecs)
    
    
    out = {0: [], 1: [], 2: [], 3:[]}
    print(categorized_words)
    for word,cat in zip(words, categorized_words):
        out[cat].append(word)
    
    print("==== Potential categorization of connections words ====")
    for line in out:
        print(out[line])
    
    
    
    