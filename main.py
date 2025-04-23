from fastapi import FastAPI, Query
import pandas as pd
import openai
from rapidfuzz import process, fuzz

app = FastAPI()

# Load the FAQ data
df = pd.read_csv("faq.csv", encoding="utf-8-sig")
questions = df["Question"].tolist()

@app.get("/faq")
def get_faq(q: str = Query(...)):
    # Find best match using RapidFuzz
    match, score, _ = process.extractOne(q, questions, scorer=fuzz.ratio)
    if score < 85:
        return "I do not possess the information to answer that question. Try asking me something about financial, retirement, estate, or healthcare planning."

    matched_question = match
    matched_answer = df[df["Question"] == matched_question]["Answer"].values[0].strip()
    coaching_tip = df[df["Question"] == matched_question]["Coaching Tip"].values[0].strip()

    # Fallback if the answer is missing or just whitespace
    if not matched_answer.strip():
        return "I'm sorry, I couldn't find a full answer to that question. Try rephrasing or asking something else about financial, estate, or health-related planning."

    # Compose prompt
    prompt = (
        "You are a helpful assistant. Respond in plain text only. Do not use Markdown, bullets, or HTML.\n\n"
        "Always use the provided answer exactly as written. "
        "If a coaching tip is included, repeat it at the end under a heading called 'Coaching Tip.'\n\n"
        f"Question: {q}\n"
        f"Answer: {matched_answer}\n"
        f"Coaching Tip: {coaching_tip}"
    )

    # Debug output
    print("🧪 DEBUG INFO:")
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
