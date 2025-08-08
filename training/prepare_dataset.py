import pandas as pd
import json

# Load CSV
df = pd.read_csv('./data/medical_faq.csv')
df.dropna(subset=['question', 'answer'], inplace=True)
df.drop_duplicates(inplace=True)

# Convert to LoRA-ready format
records = []
for _, row in df.iterrows():
    prompt = f"Question: {row['question'].strip()}\nAnswer:"
    records.append({
        "instruction": prompt,
        "input": "",
        "output": row['answer'].strip()
    })

# Save to JSONL
with open('./data/medical_faq.jsonl', 'w') as f:
    for r in records:
        f.write(json.dumps(r) + '\n')

print(f"✅ Saved {len(records)} entries → medical_faq.jsonl")
