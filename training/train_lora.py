import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# Load Dataset
# This loads custom JSONL file with instruction-output format
dataset = load_dataset("json", data_files="./data/medical_faq.jsonl", split="train")

# Load Base Model + Tokenizerx
model_name = "tiiuae/falcon-rw-1b"  # This is enough for CPU
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Falcon models need this

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  #Use float32 for CPU
    trust_remote_code=True
)

# Apply LoRA via PEFT
lora_config = LoraConfig(
    r=8,                                   # Low-rank dimension (tweakable)
    lora_alpha=16,
    target_modules=["query_key_value"],    # Falcon uses this layer naming
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(base_model, lora_config)

# Tokenization
def tokenize(example):
    full_prompt = example["instruction"] + " " + example["output"]
    return tokenizer(full_prompt, truncation=True, padding="max_length", max_length=256)

tokenized_dataset = dataset.map(tokenize)

# Training Arguments
training_args = TrainingArguments(
    output_dir="models/medassist-falcon-lora",          # Where to save model
    per_device_train_batch_size=1,                      # Small batch (CPU)
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="no",                                 # Skip model checkpoints for now
    report_to="none",                                   # No wandb
    gradient_accumulation_steps=2
)

# Trainer Setup 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# Start Training
trainer.train()

# Save Final Model
model.save_pretrained("training/output")
tokenizer.save_pretrained("training/output")

print("LoRA fine-tuning complete. Model saved to: training/output")








