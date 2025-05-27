from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import weaviate
import openai
from weaviate.collections.classes.filters import Filter

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://whealthchat.ai"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Weaviate connection
client = weaviate.connect_to_wcs(
    cluster_url=os.getenv("WEAVIATE_CLUSTER_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
)
collection = client.collections.get("FAQ")

@app.get("/version")
def version_check():
    return {"status": "Running", "message": "✅ CORS enabled version"}

@app.post("/faq")
async def get_faq(request: Request):
    body = await request.json()
    q = body.get("query", "").strip()
    print(f"Received question: {q}")

    # ✅ Step 1: Exact match by full string
    filters = Filter.by_property("question").equal(q)
    result = collection.query.fetch_objects(filters=filters, limit=1)

    if result.objects:
        obj = result.objects[0]
        print("✅ Exact match found:", obj.properties["question"])
        answer = obj.properties.get("answer", "").strip()
        coaching_tip = obj.properties.get("coachingTip", "").strip()
    else:
        # ✅ Step 2: Fall back to vector search
        response = collection.query.near_text(
            query=q,
            limit=1,
            return_metadata=["distance"]
        )
        print("🧠 Fallback vector search results:", response)

        if not response.objects:
            return "I don’t have an answer to that yet. Try asking about estate, retirement, or healthcare planning."

        obj = response.objects[0]
        distance = getattr(obj.metadata, "distance", 1.0)
        if distance > 0.6:
            return "I don’t have an answer to that yet. Try asking about estate, retirement, or healthcare planning."

        print("✅ Vector fallback question used:", obj.properties["question"])
        answer = obj.properties.get("answer", "").strip()
        coaching_tip = obj.properties.get("coachingTip", "").strip()

    # ✅ Format final prompt
    prompt = (
        "You are a helpful assistant. Respond using Markdown with consistent formatting.\n"
        "Do NOT include the word 'Answer:' in your response.\n"
        "Bold the words 'Coaching Tip:' exactly as shown.\n"
        "Do not bold any other parts of the answer text.\n"
        "Keep 'Coaching Tip:' inline with the rest of the text, followed by a colon.\n"
        "Use line breaks only to separate paragraphs.\n\n"
        f"Question: {q}\n"
        f"{answer}\n"
        f"Coaching Tip: {coaching_tip}"
    )

    print("📨 Prompt sent to OpenAI:", repr(prompt))
    try:
        reply = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.5
        )
        clean = reply.choices[0].message.content.strip()
        return clean.strip('"').replace("\\n", "\n")
    except Exception as e:
        return f"An error occurred: {str(e)}"
