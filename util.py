import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import glob

import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from wordcloud import WordCloud
import json

############################################### Preprocessing #########################################################
def remove_characters(text: str) -> str:
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_text_files(path: str, N=0, shuffle=False) -> list:
    text_files = glob.glob(f"{path}/*.txt")
    if shuffle:
        random.shuffle(text_files)
    if N > 0:
        text_files = text_files[:N]
    return text_files

def get_text(text_files: list) -> list:
    text = []
    for text_file in text_files:
        with open(text_file, 'r', errors='ignore') as f:
            content = remove_characters(f.read())
            content = content.lower()
            text.append(content)
    return text

def tokenization(text):
    if type(text) == list:
        return [word_tokenize(t) for t in text]
    elif type(text) == str:
        return word_tokenize(text)
    return None

############################################### Stemming #########################################################
def stemmer(tokenized_text: list):
    ps = PorterStemmer()
    stemmed_text = []
    for doc in tokenized_text:
        stemmed_text.append([ps.stem(token) for token in doc])

    stemmed_dict = {}
    for doc in stemmed_text:
        for token in doc:
            if token in stemmed_dict:
                stemmed_dict[token] += 1
            else:
                stemmed_dict[token] = 1
    
    return stemmed_dict, stemmed_text

def get_top_stems(stemmed_dict: dict, n: int) -> list:
    sorted_items = sorted(stemmed_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:n]

############################################################ WordCloud and graphing ############################################################
def plot_wordcloud(items: list):
    word_freq_dict = {word: freq for word, freq in items}
    font_path = "./US101.TTF"
    wordcloud = WordCloud(width=800, height=800, font_path=font_path).generate_from_frequencies(word_freq_dict)

    plt.figure(figsize=(8, 8))
    plt.imshow(wordcloud, interpolation='none')
    plt.axis('off')
    plt.show()

def plot_tf_dist(items: list):
    word_freq_dict = {word: freq for word, freq in items}
    words = list(word_freq_dict.keys())
    freq = list(word_freq_dict.values())

    fig, ax = plt.subplots(figsize =(20, 20))
    ax.barh(words, freq)

    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_tick_params(pad = 5)
    ax.yaxis.set_tick_params(pad = 10)

    ax.grid(b = True, color ='grey', linestyle ='-.', linewidth = 0.5, alpha = 1)

    ax.invert_yaxis()

    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5, str(round((i.get_width()), 2)), fontsize = 20, fontweight ='bold', color ='grey')
    ax.set_title('Corpus word frequency')

    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.rcParams.update({'font.size': 20})
    plt.show()

############################################################ tf and tf-idf ############################################################
def get_terms_per_doc(tokenized_text: list):
    terms_per_doc = [set(doc) for doc in tokenized_text]
    return terms_per_doc

def get_terms(tokenized_text: list):
    terms = set()
    for doc in tokenized_text:
        for token in doc:
            terms.add(token)
    return list(terms)

# Term Frequency
def get_tf_dict(tokenized_text: list, text_file_names: list, stemming=False):
    tf = {}
    if stemming:
        ps = PorterStemmer()
        for i, doc in enumerate(tokenized_text):
            freq_dict = {}
            for token in doc:
                root = ps.stem(token)
                if root in freq_dict:
                    freq_dict[root] += 1
                else:
                    freq_dict[root] = 1
            file = text_file_names[i]
            tf[file] = freq_dict
    else:
        for i, doc in enumerate(tokenized_text):
            freq_dict = {}
            for token in doc:
                if token in freq_dict:
                    freq_dict[token] += 1
                else:
                    freq_dict[token] = 1
            file = text_file_names[i]
            tf[file] = freq_dict
    return tf

def get_tf_matrix(tf_dict: dict):    
    tf_matrix = pd.DataFrame.from_dict(tf_dict)
    tf_matrix = tf_matrix.fillna(0)
    tf_matrix = tf_matrix / tf_matrix.max()
    return tf_matrix

# Inverse Document Frequency
def get_idf_dict(tokenized_text: list, text_file_names: list, stemming=False):
    if stemming:
        _, tokenized_text = stemmer(tokenized_text)
    terms_per_doc = get_terms_per_doc(tokenized_text)
    terms = get_terms(tokenized_text)
    idf = {}
    N = len(text_file_names)
    for term in terms:
        count = 0
        for doc in terms_per_doc:
            if term in doc:
                count += 1
        idf[term] = np.log(N / count)
    return idf

# Term Frequency - Inverse Document Frequency
def get_tf_idf_matrix(tf_matrix: pd.DataFrame, idf_dict: dict):
    tfidf_matrix = tf_matrix.copy()
    for term in tfidf_matrix.index:
        tfidf_matrix.loc[term] = tfidf_matrix.loc[term] * idf_dict[term]
    return tfidf_matrix

############################################################ Stop Words ############################################################
def get_stopwords(path: str):
    with open(path, 'r') as f:
        stopwords = f.read().splitlines()
    return stopwords

def remove_stopwords(path: str, tokenized_text: list):
    stopwords = get_stopwords(path)
    tokenized_text = [[token for token in doc if token not in stopwords] for doc in tokenized_text]
    return tokenized_text

############################################################ tfidf Model ############################################################
# Stem Analysis
def get_top_stems_per_doc(df: pd.DataFrame, n: int):
    docs = df.columns.values.tolist()
    terms = df.index.values.tolist()
    top_stems = []
    for i in range(len(docs)):
        doc_values = zip(terms, df.iloc[:, i])
        doc_values = sorted(doc_values, key = lambda x : x[1], reverse=True)[:n]
        series = pd.Series([x[1] for x in doc_values], index=[x[0] for x in doc_values])
        top_stems.append(series)
        
    top_stems_by_doc = pd.DataFrame(top_stems)
    top_stems_by_doc = top_stems_by_doc.T
    top_stems_by_doc = top_stems_by_doc.fillna(0)
    top_stems_by_doc.columns = docs
    return top_stems, top_stems_by_doc

# Boolean and vector models based on top p stems
def complete_vocabulary(df: pd.DataFrame, top_stems_by_doc: pd.DataFrame):
    top_stems_doc = top_stems_by_doc.copy()
    vocab = df.index.values.tolist()
    current_vocab = set(top_stems_doc.index.values.tolist())
    for term in vocab:
        if term not in current_vocab:
            top_stems_doc.loc[term] = 0.0
    return top_stems_doc

def boolean_model(df: pd.DataFrame, top_stems_by_doc: pd.DataFrame):
    top_stems_doc = complete_vocabulary(df, top_stems_by_doc)
    docs = top_stems_doc.columns.values.tolist()
    terms = top_stems_doc.index.values.tolist()
    boolean_matrix = pd.DataFrame(0, index=terms, columns=docs)
    for doc in docs:
        for term in terms:
            if top_stems_doc.loc[term, doc] > 0:
                boolean_matrix.loc[term, doc] = 1
    return boolean_matrix

def vector_model(df: pd.DataFrame, top_stems_by_doc: pd.DataFrame):
    vector_matrix = complete_vocabulary(df, top_stems_by_doc)
    return vector_matrix

############################################################ Probabilistic Model ############################################################
# Query functions for probabilistic model
def get_query(directory: str, N=1, stemming=False):
    text_files = get_text_files(directory, N=N, shuffle=True)
    text = get_text(text_files)
    text_files = [Path(file).stem + '.txt' for file in text_files]
    query = ""
    for t in text:
        start = random.randint(0, len(t) // 2)
        end = random.randint(start, len(t))
        query += t[start:end]
    query = tokenization(query)
    if stemming:
        _, query = stemmer([query])
        query = query[0]
    return query, text_files

def get_query_vector(query: list, df: pd.DataFrame):
    query_vector = pd.Series(0, index=df.index.values.tolist())
    for term in query:
        if term in query_vector.index:
            query_vector.loc[term] += 1
    query_vector = query_vector / query_vector.max()
    return query_vector

def queryVectorRepresentation(query: list, idf_dict: dict):
    query_dict = {}
    for term in query:
        if term not in query_dict:
            query_dict[term] = 1
        else:
            query_dict[term] += 1
    
    max_freq = max(query_dict.values())
    for term in query_dict:
        query_dict[term] = query_dict[term] / max_freq
            
    query_vector = {}
    for term in idf_dict:
        if term in query_dict:
            query_vector[term] = query_dict[term] * idf_dict[term]
        else:
            query_vector[term] = 0
    return pd.Series(query_vector)

def get_query_prob(text_files: list, stemming=False):
    text = get_text(text_files)
    text_files = [Path(file).stem + '.txt' for file in text_files]
    query = ""
    for t in text:
        start = random.randint(0, len(t) // 2)
        end = random.randint(start, len(t))
        query += t[start:end]
    query = tokenization(query)
    if stemming:
        _, query = stemmer([query])
        query = query[0]
    return query

class ProbabilisticInformationRetrievalModel:
    def __init__(self, documents, index_terms, df=None):
        self.documents = documents
        self.index_terms = index_terms
        self.num_documents = len(documents)
        self.num_index_terms = len(index_terms)
        self.P_R = None  # Probability that a document is relevant
        self.P_not_R = None  # Probability that a document is non-relevant
        self.weights = None  # Index term weights
        self.df = df

    # Initialize P(t_i|R) to 0.5 and P(t_i|¬R) based on term distribution
    def initialize_probabilities(self):
        self.P_R = np.full(self.num_index_terms, 0.5)
        term_counts = np.zeros(self.num_index_terms)

        for doc in self.documents:
            for i, term in enumerate(self.index_terms):
                if term in doc:
                    term_counts[i] += 1

        self.P_not_R = term_counts / self.num_documents

    # Calculate index term weights based on the probabilistic model
    def calculate_weights(self):
        self.weights = np.zeros(self.num_documents)
        for j, doc in enumerate(self.documents):
            for i, term in enumerate(self.index_terms):
                if not self.df.empty:
                    col = self.df.iloc[:, j]
                    if term in col.index:
                        tfidf = col[term] * self.index_terms.count(term)
                    else:
                        tfidf = 0
                    weight = (
                        np.log((self.P_R[i] / (1 - self.P_R[i])) + 1e-10)
                        + np.log(((1 - self.P_not_R[i]) / self.P_not_R[i]) + 1e-10)
                    )
                    self.weights[j] += tfidf * weight
                else:
                    tf = doc.count(term)
                    weight = (
                        np.log((self.P_R[i] / (1 - self.P_R[i])) + 1e-10)
                        + np.log(((1 - self.P_not_R[i]) / self.P_not_R[i]) + 1e-10)
                    )
                    self.weights[j] += tf * weight

    # Rank documents based on the calculated weights
    def rank_documents(self):
        ranked_docs = list(enumerate(self.weights))
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        return ranked_docs

    # Update P(t_i|R) and P(t_i|¬R) based on the top-ranked documents
    def update_probabilities(self, top_ranked_documents):
        term_counts_in_R = np.zeros(self.num_index_terms)
        term_counts_not_in_R = np.zeros(self.num_index_terms)

        for doc_idx, _ in top_ranked_documents:
            doc = self.documents[doc_idx]
            for i, term in enumerate(self.index_terms):
                if term in doc:
                    term_counts_in_R[i] += 1
                else:
                    term_counts_not_in_R[i] += 1

        self.P_R = (term_counts_in_R + 0.5) / (len(top_ranked_documents) + 1)
        self.P_not_R = (term_counts_not_in_R + 0.5) / (self.num_documents - len(top_ranked_documents) + 1)

def test_model_prob(documents, query, tfidf_matrix, doc_names):
    model = ProbabilisticInformationRetrievalModel(documents, query, tfidf_matrix)
    model.initialize_probabilities()

    # Initial ranking
    model.calculate_weights()
    initial_ranking = model.rank_documents()

    # Update probabilities and re-rank
    top_r = 2  # Choose a threshold for top-ranked documents
    top_ranked_docs = initial_ranking[:top_r]
    model.update_probabilities(top_ranked_docs)

    # Re-rank documents after the update
    model.calculate_weights()
    updated_ranking = model.rank_documents()
    
    result = {}
    for rank, (doc_idx, score) in enumerate(updated_ranking, start=1):
        result[doc_names[doc_idx]] = rank
    ranked_docs = pd.Series(result)
    return result, ranked_docs

############################################################ LSI Model ############################################################
def LSI_model(df: pd.DataFrame, num_eigen=0):
    text_files = df.columns.values.tolist()
    tdm = df.to_numpy()
    if num_eigen == 0:
        num_eigen = min(tdm.shape)
    U, s, Vh = np.linalg.svd(tdm)
    U, s, Vh = U[:, :num_eigen], s[:num_eigen], Vh[:num_eigen, :]
    S = np.diag(s)
    M = U @ S @ Vh
    model = M.T @ M
    model = pd.DataFrame(model, index=text_files, columns=text_files)
    return model

# Query function for LSI model
def get_query_lsi(directory: str, N=1):
    text_files = get_text_files(directory, N=N, shuffle=True)
    query = [Path(file).stem + '.txt' for file in text_files]
    return query

# Retrieval
def retrieval_lsi(query: list, model: pd.DataFrame):
    docs = model.index.values.tolist()
    ranked = pd.Series(0, index = docs)
    for doc in docs:
        total = 0
        for rel in query:
            total += model[rel][doc]
        total /= len(query)
        ranked[doc] = total
    return ranked.sort_values(ascending=False)

# Run model
def test_model_lsi(query: list, model: pd.DataFrame, display=False):
    ranked_docs = retrieval_lsi(query, model)
    if display:
        display_result_lsi(query, ranked_docs)
    result = {}
    i = 1
    for doc in ranked_docs.index.values.tolist():
        result[doc] = i
        i += 1
    return result, ranked_docs

def display_result_lsi(query: list, ranked_docs: pd.Series):
    print(f"Query\n{query}\n")
    # print(f"Ranked List\n{ranked_docs}\n")
    query_pos = {}
    i = 1
    for doc in ranked_docs.index.values.tolist():
        if doc in query:
            query_pos[doc] = i
        print(f"Document {doc} with score {ranked_docs[doc]} is at rank {i}")
        i += 1
    print()
    for doc in query:
        if doc in query_pos:
            print(f"Query document {doc} is at rank {query_pos[doc]}")

############################################################ Visualization ############################################################
def visualize_rankings(data: dict, ranked_docs_dict: dict, type=None):
    # Create a separate heatmap for each query
    for query, query_data in data.items():
        df = ranked_docs_dict[query].to_frame()

        # Create a heatmap for the current query with dynamic figure size
        num_documents = len(df)
        fig_height = max(4, num_documents * 0.4)  # Minimum height of 4 inches
        plt.figure(figsize=(8, fig_height))
        sns.heatmap(df, cmap="YlGnBu", annot=True, fmt=".0f", linewidths=0.5, cbar=False)

        # Customize the y-labels with highlighting
        for i, label in enumerate(plt.gca().get_yticklabels()):
            if label.get_text() in query_data["query"]:
                label.set_weight('bold')
                label.set_color('red')
            else:
                label.set_weight('normal')
                label.set_color('black')

        if type:
            plt.title(f"{type} Model Document Rankings for {query}")
        else:
            plt.title(f"Model Document Rankings for {query}")
        plt.xlabel("Document")
        plt.ylabel("Ranking")
        plt.show()
