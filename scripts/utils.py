import numpy as np
from numpy.typing import ArrayLike
from itertools import permutations
from sklearn.metrics import accuracy_score

# Helper function to apply the permutation
def apply_perm(perm: ArrayLike):
    """Applys a permutation to some element

    Args:
        perm (ArrayLike): _description_
    """
    def permute(i:int):
        try:
            return perm[i]
        except:
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
    best_perm:tuple = (1,2,3,4)
    best_score:int = accuracy_score(actual, predicted)
    
    # For KMeans, 0 is a fixed point ¯\_(ツ)_/¯ 
    permutations = [
        (0, 1, 3, 2),
        (0, 2, 1, 3),
        (0, 2, 3, 1),
        (0, 3, 1, 2),
        (0, 3, 2, 1),
    ]
    

    
    if actual_label_sequential:
        for perm in permutations:
            return


if __name__ == '__main__':
    
    test = [0,1,2,3]
    perm = [3,2,1,0]
    testperm = apply_perm(perm)
    vperm = np.vectorize(testperm)
    
    
    
    print(vperm(test))
    
    print("test")
    
    