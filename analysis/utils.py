import numpy as np  


"""
Takes two identically sized matrices and finds the similarity between vectors.
Matrices should be organised (Number of Vectors x Vector Size).
Returns a vector containing the similarity between the appropriate vector from each matrix.
"""
def vector_similarity(matrix1, matrix2, compareFunction):
    if (matrix1.shape != matrix2.shape): return

    output = []

    for i in range(matrix1.shape[0]):
        output.append(compareFunction(matrix1[i], matrix2[i]))

    return output


"""
Returns cosine similarity of two vectors.
"""
def cos_sim(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))