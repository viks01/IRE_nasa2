from util import *

def complete_boolean_model(df: pd.DataFrame):
    docs = df.columns.values.tolist()
    terms = df.index.values.tolist()
    boolean_matrix = pd.DataFrame(0, index=terms, columns=docs)
    for doc in docs:
        for term in terms:
            if df.loc[term, doc] > 0:
                boolean_matrix.loc[term, doc] = 1
    return boolean_matrix

def TermDocumentMatrixProbModel(directory: str, stop_words_path: str, stemming=False):
    text_files = get_text_files(directory, shuffle=False)
    text = get_text(text_files)
    tokenized_text = tokenization(text)
    tokenized_text = remove_stopwords(stop_words_path, tokenized_text)
    if stemming:
        _, tokenized_text = stemmer(tokenized_text)
    text_files = [text_file.split('/')[-1] for text_file in text_files]
    tf_dict = get_tf_dict(tokenized_text, text_files, stemming=stemming)
    tf_matrix = get_tf_matrix(tf_dict)
    boolean_matrix = complete_boolean_model(tf_matrix)
    return boolean_matrix