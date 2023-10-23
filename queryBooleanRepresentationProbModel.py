from util import *

def queryBooleanRepresentationProbModel(query: list, vocabulary: list):
    query_terms = set(query)
    query_vector = {}
    for term in vocabulary:
        if term in query_terms:
            query_vector[term] = 1
        else:
            query_vector[term] = 0
    return pd.Series(query_vector)