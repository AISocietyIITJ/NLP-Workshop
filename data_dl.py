from datasets import load_dataset
import json


ds_kn = load_dataset(
    "ai4bharat/sangraha", data_dir="synthetic/kan_Knda", split="train", streaming=True)

# streaming=True ensures we don’t download everything up front
# ds_ja = load_dataset("HuggingFaceFW/fineweb-2", "jpn_Jpan",
#                      split="train", streaming=True)

# ds_en = load_dataset("agentlans/high-quality-english-sentences",
#                      split="train", streaming=True)
target = 400000  # was 30k
out = []
lang = "kan"
for item in ds_kn:
    text = item.get("text", None)  # type: ignore
    if text is None:
        continue
    # optionally filter: skip very short (<20 chars) or huge ones
    if len(text) < 20 or len(text) > 1000:
        continue
    out.append(text)
    if len(out) >= target:
        break

# write to a newline-delimited file
with open(f"data/{lang}_12k.txt", "x", encoding="utf-8") as f:
    for t in out:
        f.write(t.replace("\n", " ") + "\n")
