from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from weaviate.collections.classes.filters import Filter
import os
import weaviate
import openai

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://whealthchat.ai"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/version")
def version_check():
    return {"status": "Running", "message": "✅ Exact match + fallback version"}

# Connect to Weaviate
client = weaviate.connect_to_wcs(
    cluster_url=os.getenv("WEAVIATE_CLUSTER_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
)

collection = client.collections.get("FAQ")

@app.post("/faq")
async def get_faq(request: Request):
    body = await request.json()
    q = body.get("query", "").strip()
    print(f"Received question: {q}")

    # ✅ STEP 1: Check for exact match
    exact_filter = Filter.by_property("question").equal(q)
    exact_result = collection.query.fetch_objects(filters=exact_filter, limit=1)

    if exact_result.objects:
        obj = exact_result.objects[0]
        print("✅ Exact match found:", obj.properties["question"])
    else:
        # 🧠 STEP 2: Fallback to vector search
        response = collection.query.near_text(query=q, limit=1, return_metadata=["distance"])
        print("🧠 Fallback vector search results:", response)

        if not response.objects:
            return "I do not possess the information to answer that question. Try asking about financial, estate, or healthcare planning."

        obj = response.objects[0]
        if obj.metadata.distance > 0.6:
            return "I do not possess the information to answer that question. Try asking about financial, estate, or healthcare planning."

    # Format the response
    props = obj.properties
    answer = props.get("answer", "").strip()
    coaching_tip = props.get("coachingTip", "").strip()

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
        clean_response = reply.choices[0].message.content.strip()
        if clean_response.startswith('"') and clean_response.endswith('"'):
            clean_response = clean_response[1:-1]
        clean_response = clean_response.replace("\\n", "\n").strip()
        return clean_response

    except Exception as e:
        return f"An error occurred: {str(e)}"
