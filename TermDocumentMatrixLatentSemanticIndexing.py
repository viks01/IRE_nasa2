from util import *

def TermDocumentMatrixLatentSemanticIndexing(directory: str, stop_words_path: str, stemming=False, num_eigen=0):
    text_files = get_text_files(directory, shuffle=False)
    text = get_text(text_files)
    tokenized_text = tokenization(text)
    tokenized_text = remove_stopwords(stop_words_path, tokenized_text)
    if stemming:
        _, tokenized_text = stemmer(tokenized_text)
    text_files = [text_file.split('/')[-1] for text_file in text_files]
    tf_dict = get_tf_dict(tokenized_text, text_files, stemming=stemming)
    tf_matrix = get_tf_matrix(tf_dict)
    idf_dict = get_idf_dict(tokenized_text, text_files, stemming=stemming)
    tfidf_matrix = get_tf_idf_matrix(tf_matrix, idf_dict)
    tdm = tfidf_matrix.to_numpy()
    if num_eigen == 0:
        num_eigen = min(tdm.shape)
    U, s, Vh = np.linalg.svd(tdm)
    U, s, Vh = U[:, :num_eigen], s[:num_eigen], Vh[:num_eigen, :]
    S = np.diag(s)
    M = U @ S @ Vh
    model = M.T @ M
    model = pd.DataFrame(model, index=text_files, columns=text_files)
    return model