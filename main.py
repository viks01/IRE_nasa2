from util import *

from queryBooleanRepresentationProbModel import queryBooleanRepresentationProbModel
from RankingLatentSemanticIndexing import RankingLatentSemanticIndexing
from TermDocumentMatrixLatentSemanticIndexing import TermDocumentMatrixLatentSemanticIndexing
from TermDocumentMatrixProbModel import TermDocumentMatrixProbModel


directory = "./nasa"
stop_words_path = "./english.stop"

boolean_matrix = TermDocumentMatrixProbModel(directory, stop_words_path, stemming=True)
lsi_model = TermDocumentMatrixLatentSemanticIndexing(directory, stop_words_path, stemming=True, num_eigen=20)


# LSI query ranking
print("LSI Query Ranking:")
N = 10
query = get_query_lsi(directory, 1)
result, ranked_docs = RankingLatentSemanticIndexing(query, lsi_model, display=True, N=N)


# Probabilistic model query ranking
print("\nProbabilistic Model Query Ranking:")
vocabulary = boolean_matrix.index.values.tolist()
query, relevant_docs = get_query(directory, N=1, stemming=True)
query_boolean = queryBooleanRepresentationProbModel(query, vocabulary)

text_files = get_text_files(directory, shuffle=False)
text = get_text(text_files)
tokenized_text = tokenization(text)
tokenized_text = remove_stopwords(stop_words_path, tokenized_text)
_, tokenized_text = stemmer(tokenized_text)
text_files = [text_file.split('/')[-1] for text_file in text_files]

documents = [' '.join(doc) for doc in tokenized_text]

model = ProbabilisticInformationRetrievalModel(documents, query, boolean_matrix)
model.initialize_probabilities()

# Initial ranking
model.calculate_weights()
initial_ranking = model.rank_documents()
print("\nInitial Ranking:")
for rank, (doc_idx, score) in enumerate(initial_ranking, start=1):
    print(f"Rank {rank}: Document {doc_idx} - Score: {score:.2f}")

# Update probabilities and re-rank
top_r = 10  # Choose a threshold for top-ranked documents
top_ranked_docs = initial_ranking[:top_r]
model.update_probabilities(top_ranked_docs)

# Re-rank documents after the update
model.calculate_weights()
updated_ranking = model.rank_documents()
print("\nUpdated Ranking:")
for rank, (doc_idx, score) in enumerate(updated_ranking, start=1):
    print(f"Rank {rank}: Document {doc_idx} - Score: {score:.2f}")

rel_docs = {}
for i, doc in enumerate(text_files):
    if doc in relevant_docs:
        rel_docs[doc] = f"Document {i}"
res = json.dumps(rel_docs, indent=4)
print(f"\nRelevant documents:\n{res}\n")
