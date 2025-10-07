from typing import Set

sentences = [
    "it was the best of times",
    "it it was the worst of times",
    "it was the age of wisdom wisdom wisdom",
    "it was the age of of foolishness"
]

# Tokenize each sentence into a list of words
tokenized_sentences = [sentence.split(sep=' ') for sentence in sentences]

vocabulary: Set[str] = set()

for tok_sentence in tokenized_sentences:
    for tok in tok_sentence:
        vocabulary.add(tok)

# ordered list
vocab_list = list(vocabulary)

# Binary BOW
bowBinary = []
for tok_sentence in tokenized_sentences:
    sentenceVec = []  # Binary BOW Vector
    for word in vocab_list:
        if word in tok_sentence:
            sentenceVec.append(1)
        else:
            sentenceVec.append(0)
    bowBinary.append(sentenceVec)

print("Vocabulary:", vocab_list)
print("Binary BOW:", bowBinary)

# Word Freq BOW
bowCount = []
for tok_sentence in tokenized_sentences:
    sentenceVec = []
    for word in vocab_list:
        sentenceVec.append(tok_sentence.count(word))
    bowCount.append(sentenceVec)

print("Count BOW:", bowCount)
