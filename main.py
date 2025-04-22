from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from difflib import get_close_matches

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = pd.read_csv("faq.csv", encoding="utf-8")
questions = df["Question"].tolist()

@app.get("/faq")
def get_faq(q: str = Query(...)):
    matches = get_close_matches(q, questions, n=1, cutoff=0.4)
    if matches:
        match = matches[0]
        answer = df[df["Question"] == match]["Answer"].values[0]
        return {"match": match, "answer": answer}
    else:
        return {"match": None, "answer": "I'm sorry, I couldn't find an answer."}
