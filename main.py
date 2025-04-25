from fastapi import FastAPI, Query
import os
import weaviate
import openai

app = FastAPI()

# Connect to Weaviate (RAG backend)
client = weaviate.connect_to_wcs(
    cluster_url=os.getenv("WEAVIATE_CLUSTER_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
)

collection = client.collections.get("Whealthchat_rag")

@app.get("/faq")
def get_faq(q: str = Query(...)):
    response = collection.query.near_text(
        query=q,
        limit=1,
        return_metadata=["score"]
    )

    if not response.objects:
        return "I do not possess the information to answer that question. Try asking me something about financial, retirement, estate, or healthcare planning."

    obj_full = response.objects[0]
    score = obj_full.additional.get("score", 1.0)

    if score < 0.75:
        return "I do not possess the information to answer that question. Try asking me something about financial, retirement, estate, or healthcare planning."

    obj = obj_full.properties
    question = obj.get("question", "").strip()
    answer = obj.get("answer", "").strip()
    coaching_tip = obj.get("coachingTip", "").strip()

    prompt = (
        "You are a helpful assistant. Respond in plain text only. Do not use Markdown, bullets, or HTML.\n\n"
        "Always use the provided answer exactly as written. "
        "If a coaching tip is included, repeat it at the end under a heading called 'Coaching Tip.'\n\n"
        f"Question: {q}\n"
        f"Answer: {answer}\n"
        f"Coaching Tip: {coaching_tip}"
    )

    try:
        reply = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.5
        )
        return reply.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"
