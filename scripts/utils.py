import numpy as np
from numpy.typing import ArrayLike
from itertools import permutations
from sklearn.metrics import accuracy_score
import warnings

# An actual situation where currying is warranted. 
def apply_perm(perm: ArrayLike):
    """Applys a permutation to some element

    Args:
        perm (ArrayLike): The permutation map.  Permute indexes perm to apply the permutation for each element
    """
    def permute(i:int):
        try:
            return perm[i]
        except:
            warnstr = f'Attempted to permute {i}, but {i} was not found in the permutation map.  The default behaviour is to return the unpermuted element.'
            warnings.warn(warnstr)
            return i
        
    return permute


def permute_to_minimize_cmatrix(predicted: ArrayLike,
                                actual: ArrayLike = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3],
                                actual_label_sequential: bool = True
                                ) -> ArrayLike:
    """Permutes the labels to minimize the confusion matrix
    Since KMeans clustering isn't necessarily consistent with the labeling scheme of the true values there can 
    arise situations where KMeans swaps the labels of two categories, for example, 
    it can mislabel 2ss as 1s and 1s as 2s giving a false 

    Args:
        predicted (ArrayLike): The predicted values
        actual (ArrayLike): The actual values
    
    returns: The best possible confusion matrix
    """
    best_perm:tuple = (0,1,2,3)
    best_score:int = accuracy_score(actual, predicted)
    
    # For KMeans, 0 is a fixed point ¯\_(ツ)_/¯ 
    permutations = np.array([(0, 1, 2, 3),
        (0, 1, 3, 2),
        (0, 2, 1, 3),
        (0, 2, 3, 1),
        (0, 3, 1, 2),
        (0, 3, 2, 1),
        (1, 0, 2, 3),
        (1, 0, 3, 2),
        (1, 2, 0, 3),
        (1, 2, 3, 0),
        (1, 3, 0, 2),
        (1, 3, 2, 0),
        (2, 0, 1, 3),
        (2, 0, 3, 1),
        (2, 1, 0, 3),
        (2, 1, 3, 0),
        (2, 3, 0, 1),
        (2, 3, 1, 0),
        (3, 0, 1, 2),
        (3, 0, 2, 1),
        (3, 1, 0, 2),
        (3, 1, 2, 0),
        (3, 2, 0, 1),
        (3, 2, 1, 0)])
    
    

    
    if actual_label_sequential:
        for perm in permutations:
            
            permute = np.vectorize(apply_perm(perm))
            permuted_predicted = permute(predicted)
            
            candidate_score = accuracy_score(actual, permuted_predicted)
            
            if best_score < candidate_score:
                best_score = candidate_score
                best_perm = perm
    
    permute = np.vectorize(apply_perm(best_perm))
    permuted_predicted = permute(predicted)
    
    return best_perm, permuted_predicted

def get_vectors(words: list[str], model):
    """Given a Word embedding model and a list of words, return a numpy array consisting of the word vectors

    Args:
        words (list[str]): List of words
        model (_type_): Our vector embedding model, typically Google's Word2Vec

    Returns:
        np.array: A list of vector embeddings if the word is contained within the model, else, return None.
    """
    try:
        lst_of_vecs = []
        for word in words:
            lst_of_vecs.append(model.get_vector(word, norm=True))
        return np.array(lst_of_vecs)
    except:
        return None


if __name__ == '__main__':
    
    test = [0,1,2,3,4]
    perm = [3,2,1,0]
    testperm = apply_perm(perm)
    vperm = np.vectorize(testperm)
    
    
    
    print(vperm(test))
    
    
    actual = sum([[i]*4 for i in range(4)], [])
    predicted = sum([[i]*4 for i in [0,3,1,2]], [])
    
    
    score, permuted_predicted = permute_to_minimize_cmatrix(predicted, actual)
    assert(np.array(actual).all() == permuted_predicted.all())
    
    print(predicted)
    print(actual)    
    print("test")
    
    