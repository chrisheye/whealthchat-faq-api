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
        limit=3  # Fetch top 3 matches
    )

    if not response.objects:
        return "I do not possess the information to answer that question. Try asking me something about financial, retirement, estate, or healthcare planning."

    blocks = []
    for obj in response.objects:
        props = obj.properties
        question = props.get("question", "").strip()
        answer = props.get("answer", "").strip()
        tip = props.get("coachingTip", "").strip()

        if not answer:
            continue

        block = f"Answer: {answer}"
        if tip:
            block += f"\n\nCoaching Tip: {tip}"
        blocks.append(block)

    full_prompt = (
        "You are a helpful assistant. Respond in plain text only. Do not use Markdown, bullets, or HTML.\n\n"
        "The user asked a question. You were given multiple relevant answers. Create a single, coherent response based on them.\n\n"
        f"Question: {q}\n\n" + "\n\n---\n\n".join(blocks)
    )

    try:
        reply = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=400,
            temperature=0.5
        )
        return reply.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"
