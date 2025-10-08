from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from yellowbrick.text import TSNEVisualizer
import numpy as np
from sklearn.manifold import TSNE
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import TruncatedSVD
import plotly.express as px
# from tiktoken import 
import tiktoken
import json
from tqdm import tqdm
import plotly.graph_objects as go


# dataset = load_dataset('Ishat/SentimentAnalysisRAID', split='train')

# dataset.to_json('dataset.json', lines=False)


with open("dataset.json", "r") as f:
    data = [json.loads(line) for line in f]
len(data)

# enc = tiktoken.get_encoding("o200k_base")
# assert enc.decode(enc.encode("hello world")) == "hello world"

# To get the tokeniser corresponding to a specific model in the OpenAI API:
enc = tiktoken.encoding_for_model("gpt-4o")
test_sample = "Yolo"

check = enc.encode(test_sample)
print(enc.decode([check[1]]))
# tests = enc.encode_batch([test_sample, test_sample])


def visualize_decision_boundary_3d(model, X, y, title="3D Decision Boundary"):
    # Step 1: Dimensionality reduction
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_reduced = svd.fit_transform(X)

    tsne = TSNE(n_components=3, random_state=42, perplexity=30, init='pca')
    X_3d = tsne.fit_transform(X_reduced)

    # Step 2: Train classifier on 3D features
    model.fit(X_3d, y)

    # Step 3: Build grid for visualization
    x_min, x_max = X_3d[:, 0].min() - 5, X_3d[:, 0].max() + 5
    y_min, y_max = X_3d[:, 1].min() - 5, X_3d[:, 1].max() + 5
    z_min, z_max = X_3d[:, 2].min() - 5, X_3d[:, 2].max() + 5

    # Keep grid coarse to avoid huge computation
    grid_size = 25
    x_range = np.linspace(x_min, x_max, grid_size)
    y_range = np.linspace(y_min, y_max, grid_size)
    z_range = np.linspace(z_min, z_max, grid_size)
    xx, yy, zz = np.meshgrid(x_range, y_range, z_range)

    grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    preds = model.predict(grid_points)

    # Step 4: Prepare Plotly traces
    scatter_train = go.Scatter3d(
        x=X_3d[:, 0],
        y=X_3d[:, 1],
        z=X_3d[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=y,
            colorscale=['red', 'green'],
            opacity=0.8
        ),
        name='Training points'
    )

    scatter_pred = go.Scatter3d(
        x=grid_points[:, 0],
        y=grid_points[:, 1],
        z=grid_points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=preds,
            colorscale=['rgba(255,0,0,0.2)', 'rgba(0,255,0,0.2)'],
            opacity=0.2
        ),
        name='Decision region'
    )

    fig = go.Figure(data=[scatter_pred, scatter_train])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='t-SNE Component 1',
            yaxis_title='t-SNE Component 2',
            zaxis_title='t-SNE Component 3'
        ),
        showlegend=True,
        width=900,
        height=700
    )
    fig.show()


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
    with open('filter_encoded_padded_text.json', 'r') as f:
        check = json.load(f)

    sentiment_to_idx = {
        'Positive': 1,
        'Negative': 0
    }

    train_samples = 2700
    test_samples = 1000

    list_train_text = []
    list_train_labels = []
    list_train_labels_idx = []
    for x in tqdm(check[:train_samples]):
        list_train_text.append(x['text_tokens'])
        list_train_labels.append(x['sentiment'])
        list_train_labels_idx.append(sentiment_to_idx[x['sentiment']])

    list_test_text = []
    list_test_labels = []
    list_test_labels_idx = []
    for x in tqdm(check[train_samples:train_samples+test_samples]):
        list_test_text.append(x['text_tokens'])
        list_test_labels.append(x['sentiment'])
        list_test_labels_idx.append(sentiment_to_idx[x['sentiment']])

    vectorizer = TfidfVectorizer(
        ngram_range=(3, 5),
        lowercase=False,
        sublinear_tf=True,
        analyzer='word',
        tokenizer=dummy,
        preprocessor=dummy,
        token_pattern=None, strip_accents='unicode')

    tf_train = vectorizer.fit_transform(list_train_text)
    tf_test = vectorizer.transform(list_test_text)

    print(tf_train.shape)

    rf = RandomForestClassifier()
    rf.fit(tf_train, list_train_labels_idx)

    y_pred = rf.predict(tf_test)
    print(classification_report(list_test_labels_idx, y_pred))

    svd = TruncatedSVD(n_components=50, random_state=42)
    X_reduced = svd.fit_transform(tf_train)  # (n_docs, 50)

    # 3. Run t-SNE to get 3D embeddings
    tsne = TSNE(n_components=3, random_state=42, perplexity=5, init='pca')
    X_tsne3d = tsne.fit_transform(X_reduced)  # (n_docs, 3)

    # 4. Plot with Plotly
    # You could have labels or categories for color, here just index
    labels = list(range(len(list_train_labels)))

    fig = px.scatter_3d(
        x=X_tsne3d[:, 0],
        y=X_tsne3d[:, 1],
        z=X_tsne3d[:, 2],
        color=labels,
        hover_name=list_train_labels,
        title="3D t-SNE of TF-IDF Vectors"
    )
    fig.update_traces(marker=dict(size=5))
    fig.show()


make_embeddings()
# with open('/Users/gjyotin305/Desktop/scripts/NLP-Workshop/testing/filter_encoded_padded_text.json', 'r') as f:
#     data_check = json.load(f)



# convert_token_to_text()




# check_max_len_encodings(data_check)

# pad_encodings()
# encode_tokens(data)

# print(tests)
# sentiment_dataset = load_dataset('Ishat/SentimentAnalysisRAID', split='train')

# sentiment_dataset.to_json('dataset.json')
