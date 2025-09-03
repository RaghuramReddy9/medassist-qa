from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

# 1. Set your fine-tuned model's adapter path (LoRA)
model_path = "training/output"

# 2. Load the LoRA adapter configuration
config = PeftConfig.from_pretrained(model_path)

# 3. Load the base model that was fine-tuned (e.g., Mistral, TinyLLaMA)
base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)

# 4. Load and apply the LoRA adapter on top of the base model
model = PeftModel.from_pretrained(base_model, model_path)

# 5. Load the tokenizer that matches the base model
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# 6. Use the model for inference
prompt = "What are the common symptoms of diabetes?"
inputs = tokenizer(prompt, return_tensors="pt")

# 7. Generate the model's response (disable gradients for speed)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

# 8. Decode and print the model's output
print("Model response:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
