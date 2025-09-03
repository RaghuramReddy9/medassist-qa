from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

app = FastAPI()

# Load your fine-tuned model
model_path = "training/output"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Define the input structure
class PromptRequest(BaseModel):
    prompt: str

# Inference endpoint
@app.post("/generate")
async def generate_text(req: PromptRequest):
    prompt = req.prompt
    output = pipe(prompt, max_new_tokens=100, do_sample=True)[0]["generated_text"]
    return {"response": output}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
