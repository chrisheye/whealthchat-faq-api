from fastapi import FastAPI, Query
import pandas as pd
import openai
from rapidfuzz import process, fuzz

app = FastAPI()

df = pd.read_csv("faq.csv", encoding="utf-8-sig")
questions = df["Question"].tolist()

@app.get("/faq")
def get_faq(q: str = Query(...)):
    match, score, _ = process.extractOne(q, questions, scorer=fuzz.ratio)
    if score < 85:
        return "I do not possess the information to answer that question. Try asking me something about financial, retirement, estate, or healthcare planning."

    matched_question = match
    matched_answer = df[df["Question"] == matched_question]["Answer"].values[0]
    coaching_tip = df[df["Question"] == matched_question]["Coaching Tip"].values[0]

    prompt = (
        f"You are a helpful assistant. Always use the provided answer exactly as written. "
        f"If a coaching tip is included, repeat it at the end under a heading called 'Coaching Tip.'\n\n"
        f"Question: {q}\n"
        f"Answer: {matched_answer}\n"
        f"Coaching Tip: {coaching_tip}"
    )

    print("\U0001F9EA DEBUG INFO:")
    print(f"Matched Question: {matched_question} (Score: {score})")
    print(f"Matched Answer: {matched_answer}")
    print(f"Coaching Tip: {coaching_tip}")
    print("------ Full Prompt Sent to GPT ------")
    print(prompt)
    print("-------------------------------------")

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
