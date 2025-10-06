from tokenizers import Tokenizer
from datasets import load_dataset
from huggingface_hub import login
import tiktoken
login(token="lol")

ds = load_dataset("Ishat/SentimentAnalysisRAID")
texts = ds['train']['text'] if 'train' in ds else ds[ds.keys()[0]  # type: ignore
                                                     ]['text']
# Example: GPT-3.5 tokenizer
enc = tiktoken.get_encoding("o200k_base")
tokenizer: Tokenizer = Tokenizer.from_file(
    "/data/bpe_mult_tok.json")

start = 6001
print(texts[start])
for text in texts[start:start+5]:
    tiktoken_tokens = enc.encode(text)
    my_tokens = tokenizer.encode(text)
    # print(f"Text: {text}")
    print(f"Tiktoken: (len={len(tiktoken_tokens)})")
    print(f"My Tokenizer: (len={len(my_tokens)})")
