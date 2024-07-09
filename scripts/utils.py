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
    best_perm:tuple = (1,2,3,4)
    best_score:int = accuracy_score(actual, predicted)
    
    # For KMeans, 0 is a fixed point ¯\_(ツ)_/¯ 
    permutations = np.array([
        [0, 1, 3, 2],
        [0, 2, 1, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2],
        [0, 3, 2, 1],
    ])
    
    

    
    if actual_label_sequential:
        for perm in permutations:
            
            permute = np.vectorize(apply_perm(perm))
            permuted_predicted = permute(predicted)
            
            candidate_score = accuracy_score(actual, permuted_predicted)
            
            if best_score < candidate_score:
                best_score = candidate_score
                best_perm = perm
    
    return best_perm 


if __name__ == '__main__':
    
    test = [0,1,2,3,4]
    perm = [3,2,1,0]
    testperm = apply_perm(perm)
    vperm = np.vectorize(testperm)
    
    
    
    print(vperm(test))
    
    
    predicted = sum([[i]*4 for i in range(4)], [])
    actual = sum([[i]*4 for i in [0,3,1,2]], [])
    
    print(permute_to_minimize_cmatrix(predicted, actual))
    
    print(predicted)
    print(actual)    
    print("test")
    
    