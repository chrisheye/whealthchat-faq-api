from fastapi import FastAPI, Query
import os
import openai
import weaviate

app = FastAPI()

# Connect to Weaviate
client = weaviate.connect_to_wcs(
    cluster_url=os.getenv("WEAVIATE_CLUSTER_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_APIKEY")},
)
collection = client.collections.get("Whealthchat_rag")

@app.get("/faq")
def get_faq(q: str = Query(...)):
    response = collection.query.near_text(query=q, limit=3).objects
    matches = response if response else []

    if not matches:
        return "I do not possess the information to answer that question. Try asking me something about financial, retirement, estate, or healthcare planning."

    prompt_parts = [
        "You are a helpful assistant. Respond in plain text only. Do not use Markdown, bullets, or HTML.",
        "Summarize and combine the information below to answer the user's question in a clear, supportive, and actionable way.",
        f"User's Question: {q}"
    ]

    for i, obj in enumerate(matches):
        question = obj.properties.get("question", "")
        answer = obj.properties.get("answer", "")
        coaching = obj.properties.get("coachingTip", "")
        prompt_parts.append(f"\nFAQ #{i+1}:\nQ: {question}\nA: {answer}\nCoaching Tip: {coaching}")

    prompt = "\n\n".join(prompt_parts)

    try:
        chat_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.5
        )
        return chat_response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"
