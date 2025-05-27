from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import weaviate
import os
import openai
from weaviate.collections.classes.config import Property
from weaviate.collections.classes.filters import Filter
from weaviate.collections.classes.data import DataObject

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to Weaviate
client = weaviate.connect_to_wcs(
    cluster_url=os.getenv("WEAVIATE_CLUSTER_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
)

collection = client.collections.get("FAQ")

# POST endpoint
@app.post("/faq")
async def get_faq(request: Request):
    body = await request.json()
    q = body.get("query", "").strip()

    print(f"Received question: {q}")

    # Exact match lookup (case-insensitive)
    print(f"🔍 Performing exact match for: {q}")
    exact_results = collection.query.fetch_objects(
        filters=Filter.by_property("question").is_equal(q),
        limit=1,
    )

    if exact_results.objects:
        obj = exact_results.objects[0]
        print(f"✅ Exact match found: {obj.properties['question']}")
        answer = obj.properties.get("answer", "")
        coaching = obj.properties.get("coachingTip", "")
        return {
            "matchType": "exact",
            "question": obj.properties["question"],
            "answer": answer,
            "coachingTip": coaching
        }

    # Fallback to vector search
    print("⚠️ No exact match found. Performing vector search...")
    vector_results = collection.query.near_text(q, limit=1)

    if vector_results.objects:
        obj = vector_results.objects[0]
        print(f"🧠 Fallback vector match: {obj.properties['question']}")
        answer = obj.properties.get("answer", "")
        coaching = obj.properties.get("coachingTip", "")
        return {
            "matchType": "vector",
            "question": obj.properties["question"],
            "answer": answer,
            "coachingTip": coaching
        }

    # No results found
    print("❌ No FAQ entry found.")
    return {
        "matchType": "none",
        "message": "No relevant FAQ found."
    }
