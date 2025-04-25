from fastapi import FastAPI, Query
import os
import weaviate
import openai

app = FastAPI()

# Connect to Weaviate
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
        return_metadata=["distance"]
    )

    if not response.objects:
        return "I do not possess the information to answer that question. Try asking me something about financial, retirement, estate, or healthcare planning."

    obj = response.objects[0]
    score = obj.metadata.distance if obj.metadata and obj.metadata.distance is not None else 0.5

    if score > 0.3:
        return "I do not possess the information to answer that question. Try asking me something about financial, retirement, estate, or healthcare planning."

    props = obj.properties
    question = props.get("question", "").strip()
    answer = props.get("answer", "").strip()
    coaching_tip = props.get("coachingTip", "").strip()

    prompt = (
        "You are a helpful assistant. Respond in plain text only. Do not use Markdown, bullets, or HTML.\n\n"
        "Always use the provided answer exactly as written. "
        "If a coaching tip is included, start a new paragraph with 'Coaching Tip:' and show it.\n\n"
        f"Question: {q}\n"
        f"Answer: {answer}\n"
        f"Coaching Tip: {coaching_tip}"
    )

    try:
        reply = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.5
        )
        return reply.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"
