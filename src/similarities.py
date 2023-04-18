import torch


def cos_sim(a, b):
    """
    Returns the cosine similarity of two vectors.
    """
    return torch.dot(a, b)/(torch.linalg.norm(a)*torch.linalg.norm(b))


def get_similarities(m1, m2, compareFunction=cos_sim):
    """
    Take two matrices filled with a set of n vectors of length m and calculate
    the similarities between corresponding vectors. Returns a list of n similarities.
    """
    if m1==None or m2==None: return [0]
    if m1.shape != m2.shape: return [0]

    output = []

    for i in range(m1.shape[0]):
        output.append(compareFunction(m1[i], m2[i]))

    return output