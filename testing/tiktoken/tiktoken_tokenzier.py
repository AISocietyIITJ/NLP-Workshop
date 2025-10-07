import tiktoken
enc = tiktoken.get_encoding("o200k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"

# To get the tokeniser corresponding to a specific model in the OpenAI API:
enc = tiktoken.encoding_for_model("gpt-4o")

english_sample = "Yug Dalwadi" 
japanese_sample = "顕微鏡は目に見えない世界を探険できる素晴らしい道具です。科学の理解を深め、新たな発見をもたらします。"
kannda_sample = "ಕುರಿತಿಗೆ ನನ್ನಲ್ಲಿರುವ ಆಸಕ್ತಿ ಹುಟ್ಟುತ್ತಿದೆ. ವೈದ್ಯಕೀಯ ಕ್ಷೇತ್ರದಲ್ಲಿ ಹೊಸದ್ದನ್ನು ತಿಳಿದುಕೊಳ್ಳಲು ನಾನು ಉತ್ಸಾಹಿತನಾಗಿರುವೆ."

print(len(enc.encode(english_sample)))
for token in enc.encode(english_sample):
    print(token, enc.decode([token]))

print(len(enc.encode(japanese_sample)))
print(len(enc.encode(kannda_sample)))