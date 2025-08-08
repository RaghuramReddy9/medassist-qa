### Fine-Tuning Summary (LoRA + Falcon)

This project uses LoRA adapters via Hugging Face PEFT to fine-tune a small LLM (falcon-rw-1b) for medical FAQs using our custom dataset.

<details>
<summary>🔧 View Fine-Tuning Script Snippet</summary>

```python
# 🔹 Load dataset from JSONL
dataset = load_dataset("json", data_files="data/medical_faq.jsonl", split="train")

# 🔹 Load Falcon model & tokenizer
model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

# 🔹 Apply LoRA adapters
lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["query_key_value"],
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# 🔹 Tokenize dataset
def tokenize(example):
    full_prompt = example["instruction"] + " " + example["output"]
    return tokenizer(full_prompt, truncation=True, padding="max_length", max_length=256)
tokenized_dataset = dataset.map(tokenize)

# 🔹 Configure training
training_args = TrainingArguments(
    output_dir="models/medassist-falcon-lora",
    per_device_train_batch_size=1, num_train_epochs=3,
    learning_rate=2e-4, save_strategy="no"
)

# 🔹 Train + Save
trainer = Trainer(
    model=model, args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
trainer.train()
model.save_pretrained("models/medassist-falcon-lora")
```
</details>