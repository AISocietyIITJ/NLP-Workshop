from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from yellowbrick.text import TSNEVisualizer
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import plotly.express as px
# from tiktoken import 
import scipy as sp
import tiktoken
import json
from tqdm import tqdm
import pandas as pd
import pickle


dataset = load_dataset('gjyotin305/NLPWorkshop', split='train')
dataset.to_json('dataset.json', lines=True, force_ascii=False)


with open("dataset.json", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

print(len(data))

# enc = tiktoken.get_encoding("o200k_base")
# assert enc.decode(enc.encode("hello world")) == "hello world"

# To get the tokeniser corresponding to a specific model in the OpenAI API:
# enc = tiktoken.encoding_for_model("gpt-4o")
# test_sample = "Yolo"

# check = enc.encode(test_sample)
# print(enc.decode([check[1]]))
# tests = enc.encode_batch([test_sample, test_sample])


# print(enc.special_tokens_set)

def encode_tokens(dataset):
    for x in tqdm(dataset):
        encoded_tokens = enc.encode(x['text'])
        x['tokens'] = encoded_tokens

    with open('encoded_tokens_w_data.json', 'w') as f:
        json.dump(dataset, f, indent=4)

    print('Done')


def check_max_len_encodings(dataset):
    max_tokens = 0
    filter_json = []
    min_tokens = 1000
    num_of_tokens = 0
    for x in tqdm(dataset):
        max_tokens = max(len(x['tokens']), max_tokens)
        min_tokens = min(len(x['tokens']), min_tokens)
        if len(x['tokens']) > 20 and len(x['tokens']) < 40:
            num_of_tokens += 1
            filter_json.append(x)   

    with open('filter_encodes.json', 'w') as f:
        json.dump(filter_json, f, indent=4)

    check_freq = {
        'English': {
            'Positive': 0,
            'Negative': 0
        },
        'Kannada': {
            'Positive': 0,
            'Negative': 0
        },
        'Japanese': {
            'Positive': 0,
            'Negative': 0
        }
    }

    filter_max_tokens = 0
    for x in tqdm(filter_json):
        check_freq[(x['language']).strip()][x['sentiment']] += 1
        filter_max_tokens = max(len(x['tokens']), filter_max_tokens)

    print(f"Filter Max Tokens: {filter_max_tokens}")
    print(check_freq)

    print(f'MAX TOKENS: {max_tokens} | MIN TOKENS: {min_tokens} | NUM_SENTENCES: {num_of_tokens}')


def pad_encodings():
    with open('filter_encodes.json', 'r') as f:
        data_filter = json.load(f)
 
    max_tokens = 39
    token = enc.encode('<|endoftext|>', allowed_special='all')[0]

    for x in tqdm(data_filter):
        pad_length = [token] * (max_tokens - len(x['tokens']))
        x['tokens'].extend(pad_length)
        assert len(x['tokens']) == max_tokens

    with open('filter_encodes_padded.json', 'w') as f:
        json.dump(data_filter, f, indent=4)


def convert_token_to_text():
    with open('filter_encodes_padded.json', 'r') as f:
        data = json.load(f)

    for x in tqdm(data):
        x['text_tokens'] = [enc.decode([tok]) for tok in x['tokens']]

    with open('filter_encoded_padded_text.json', 'w') as f:
        json.dump(data, f, indent=4)


def dummy(text):
    return text


def make_embeddings():
    # with open('filter_encoded_padded_text.json', 'r') as f:
    #     check = json.load(f)
    check = data

    sentiment_to_idx = {
        'Positive': 1,
        'Negative': 0
    }
    
    list_text = []
    list_labels = []
    for x in tqdm(check):
        list_text.append(x['text_tokens'])
        list_labels.append(sentiment_to_idx[x['sentiment']])

    vectorizer = TfidfVectorizer(
        ngram_range=(3, 5),
        lowercase=False,
        sublinear_tf=True,
        analyzer='word',
        tokenizer=dummy,
        preprocessor=dummy,
        token_pattern=None,
        strip_accents='unicode')

    tf_train = vectorizer.fit_transform(list_text)
    labels = np.array(list_labels)
    print(tf_train.shape)
    print(labels.shape)

    sp.sparse.save_npz('/tmp/sparse_matrix.npz', tf_train)
    sp.sparse.save_npz('/tmp/labels.npz', sp.sparse.csr_matrix(labels))
    sparse_matrix = sp.sparse.load_npz('/tmp/sparse_matrix.npz')
    print(sparse_matrix.shape)
    labels = sp.sparse.load_npz('/tmp/labels.npz').toarray().ravel()
    print(labels[0])

    # # Required libraries

    # # Example documents
    # docs = [
    #     "this is the first document",
    #     "this document is the second document",
    #     "and this is the third one",
    #     "is this the first document",
    #     "another example text for testing t-sne in 3d",
    #     "more documents to see clustering"
    # ]

    # # 1. Compute TF-IDF
    # tfidf = TfidfVectorizer(stop_words='english', max_features=1000)  # limit vocab
    # X_sparse = tfidf.fit_transform(docs)  # shape: (n_docs, n_terms)

    # 2. (Optional) Reduce dimensionality before t-SNE (helps performance)
    # svd = TruncatedSVD(n_components=50, random_state=42)
    # X_reduced = svd.fit_transform(tf_train)  # (n_docs, 50)

    # # 3. Run t-SNE to get 3D embeddings
    # tsne = TSNE(n_components=3, random_state=42, perplexity=5, init='pca')
    # X_tsne3d = tsne.fit_transform(X_reduced)  # (n_docs, 3)

    # # 4. Plot with Plotly
    # # You could have labels or categories for color, here just index
    # labels = list(range(len(list_labels)))

    # fig = px.scatter_3d(
    #     x=X_tsne3d[:,0],
    #     y=X_tsne3d[:,1],
    #     z=X_tsne3d[:,2],
    #     color=labels,
    #     hover_name=list_labels,
    #     title="3D t-SNE of TF-IDF Vectors"
    # )
    # fig.update_traces(marker=dict(size=5))
    # fig.show()


    # tsne.fit(tf_train, list_labels)
    # tsne.show()
# encode_tokens(data)
# check_max_len_encodings(data)
# pad_encodings()
# convert_token_to_text()
make_embeddings()
# with open('encoded_tokens_w_data.json', 'r') as f:
#     data_check = json.load(f)

# convert_token_to_text()


# check_max_len_encodings(data_check)

# pad_encodings()
# encode_tokens(data)

# print(tests)
# sentiment_dataset = load_dataset('Ishat/SentimentAnalysisRAID', split='train')

# sentiment_dataset.to_json('dataset.json')
