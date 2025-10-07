from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(
    show_progress=True,
    vocab_size=120000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

files = [
    f"data/{lang}_12k.txt" for lang in ["eng", "kan", "jap"]]
tokenizer.train(files, trainer)

tokenizer.save("/data/bpe_mult_tok.json")
