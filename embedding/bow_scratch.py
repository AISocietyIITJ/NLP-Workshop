from typing import Dict

sentences = [
    "it was the best of times",
    "it it was the worst of times",
    "it was the age of wisdom wisdom wisdom",
    "it was the age of of foolishness"
]

# Tokenize each sentence into a list of words
tokenized_sentences = [sentence.split(sep=' ') for sentence in sentences]

wordFreq: Dict[str, int] = {}

for tok_sentence in tokenized_sentences:
    for tok in tok_sentence:
        if (tok not in wordFreq):
            wordFreq[tok] = 1  # add word to vocabulary
        else:
            wordFreq[tok] += 1  # increment frequency by 1

# Binary BOW
bowBinary = []
for tok_sentence in tokenized_sentences:
    sentenceVec = []  # Binary BOW Vector
    for word in wordFreq:
        if word in tok_sentence:
            sentenceVec.append(1)
        else:
            sentenceVec.append(0)
    bowBinary.append(sentenceVec)

print(bowBinary)

# Word Freq BOW
bowCount = []
for tok_sentence in tokenized_sentences:
    sentenceVec = []
    for word in wordFreq:
        sentenceVec.append(tok_sentence.count(word))

    bowCount.append(sentenceVec)

print(bowCount)
