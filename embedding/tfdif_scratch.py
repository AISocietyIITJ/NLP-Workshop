import math
from collections import Counter

docs = ["the sky is blue", "the sun is bright", "the sun in the sky is bright"]

tokenized_docs = []
for doc in docs:
    tokens = doc.lower().split()
    tokenized_docs.append(tokens)
    
def compute_tf(doc):
    count = Counter(doc)
    total = len(doc)
    tf = {}
    for word, c in count.items():
        tf[word] = c / total
    return tf
    
tf_list = []
for doc in tokenized_docs:
    res = compute_tf(doc)
    tf_list.append(res)

def compute_idf(docs):
    N = len(docs)
    all_words = set()
    for doc in docs:
        for word in doc:
            all_words.add(word)
            
    idf = {}
    for word in all_words:
        containing_docs = 0
        for doc in docs:
            if word in doc:
                containing_docs += 1
        idf[word] = math.log(N / (1 + containing_docs)) + 1
    return idf

idf = compute_idf(tokenized_docs)

tfidf_docs = []
for tf in tf_list:
    tfidf = {}
    for word in tf:
        tfidf[word] = tf[word] * idf[word]
    tfidf_docs.append(tfidf)

for i, tfidf in enumerate(tfidf_docs, 1):
    print(f"Document {i} TF-IDF:")
    for word, value in tfidf.items():
        print(f"{word:7s} : {value:.3f}")
    print()
