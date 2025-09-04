[![Hugging Face Model](https://img.shields.io/badge/HuggingFace-MedAssist--QA-yellow?logo=huggingface&style=flat-square)](https://huggingface.co/RaghuramReddyT/medassist-qa)


# 🩺 MedAssist-QA: Fine-Tuned Medical Question Answering Assistant

This project demonstrates fine-tuning a transformer model to answer medical-related questions using domain-specific text. It also includes an API built with FastAPI to serve predictions in real-time.

---

##  Project Overview

- **Objective**: Create a healthcare QA assistant by fine-tuning a language model.
- **Approach**: LoRA-based fine-tuning using Hugging Face Transformers on custom medical Q&A dataset.
- **Deployment**: Exposed model through a FastAPI endpoint (`/generate`) for real-time use.

---

##  Tech Stack

| Area        | Tools / Libraries                     |
|-------------|----------------------------------------|
| Model       | `AutoModelForCausalLM`, `LoRA`, `PEFT` |
| API         | `FastAPI`, `uvicorn`                  |
| Training    | `transformers`, `datasets`, `torch`   |
| Hosting     | [Hugging Face Model Repo](https://huggingface.co/RaghuramReddyT/medassist-qa) |
| Dev Env     | `Python 3.10+`, `.venv`                |

---

##  How to Run

### 1. Clone the Repo

```bash
git clone https://github.com/RaghuramReddy9/medassist-qa.git
cd medassist-qa
```
### 2. Set up virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate   # On Windows
```
### 3. Install requirements
```bash
pip install -r requirements.txt
```
### 4. Start the FastAPI Server
```bash
cd app
uvicorn main:app --reload
```
## To test
```
Go to http://127.0.0.1:8000/docs
```
##  Example Input
```bash
{
  "prompt": "What are the symptoms of high blood pressure?"
}
```
## Example Output
```bash
{
  "response": "Symptoms of high blood pressure may include headaches, shortness of breath, or nosebleeds, but it is often symptomless..."
}
```
##  Folder Structure
```bash
medassist-qa/
├── app/                # FastAPI app
│   ├── main.py         # API route
│   └── __init__.py
├── train/              # Fine-tuning scripts
│   ├── fine_tune.py
│   └── __init__.py
├── model/              # Saved model artifacts
├── test_model.py       # Script to load and test model
├── requirements.txt    
└── README.md
```
## Notes
```
1.LoRA adapters are uploaded separately to Hugging Face Hub

2.You can extend this with a Streamlit UI or deploy to cloud later
```
## Credits
Fine-tuned and deployed by `Raghuramreddy Thirumalareddy`


