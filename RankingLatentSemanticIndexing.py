from util import *

def RankingLatentSemanticIndexing(query: list, model: pd.DataFrame, display=False, N=0):
    ranked_docs = retrieval_lsi(query, model)
    if N > 0:
        ranked_docs = ranked_docs[:N]
    if display:
        display_result_lsi(query, ranked_docs)
    result = {}
    i = 1
    for doc in ranked_docs.index.values.tolist():
        result[doc] = i
        i += 1
    return result, ranked_docs