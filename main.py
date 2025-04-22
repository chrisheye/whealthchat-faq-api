import openai
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from difflib import get_close_matches
import os

# Load your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load CSV with FAQs
df = pd.read_csv("faq.csv", encoding="utf-8")
questions = df["Question"].tolist()

@app.get("/faq")
def get_faq(q: str = Query(...)):
    matches = get_close_matches(q, questions, n=1, cutoff=0.4)
    if not matches:
        return "I'm sorry, I couldn't find an answer to that question."

    matched_question = matches[0]
    matched_answer = df[df["Question"] == matched_question]["Answer"].values[0]

    # GPT prompt formatting
    prompt = f"""You are an expert assistant. Use the information below to answer the user's question clearly and helpfully.

Question: {q}
Relevant Info: {matched_answer}

Answer:"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.5
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"
