from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import weaviate
import openai

# Load environment variables
WEAVIATE_CLUSTER_URL = os.getenv("WEAVIATE_CLUSTER_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI
app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to Weaviate
client = weaviate.connect_to_wcs(
    cluster_url=WEAVIATE_CLUSTER_URL,
    auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),
    headers={"X-OpenAI-Api-Key": OPENAI_API_KEY}
)

faq_collection = client.collections.get("FAQ")

@app.post("/faq")
async def get_faq(request: Request):
    body = await request.json()
    q = body.get("query", "").strip()
    print(f"Received question: {q}")

    # --- Step 1: Manual exact match ---
    print(f"🔍 Performing exact match for: {q}")
    all_objects = faq_collection.query.fetch_objects(limit=1500).objects
    exact_match_obj = next(
        (obj for obj in all_objects if obj.properties.get("question", "").strip().lower() == q.lower()),
        None
    )

    if exact_match_obj:
        print("✅ Exact match found.")
        return {
            "matchType": "exact",
            "question": exact_match_obj.properties.get("question", ""),
            "answer": exact_match_obj.properties.get("answer", ""),
            "coachingTip": exact_match_obj.properties.get("coachingTip", "")
        }

    # --- Step 2: Fallback vector search ---
    print("❌ No exact match. Performing vector search.")
    vector_results = faq_collection.query.near_text(query=q, limit=1)
    obj = vector_results.objects[0] if vector_results.objects else None

    if obj:
        print("🧠 Fallback vector search results:", obj)
        return {
            "matchType": "vector",
            "question": obj.properties.get("question", ""),
            "answer": obj.properties.get("answer", ""),
            "coachingTip": obj.properties.get("coachingTip", "")
        }

    # --- Step 3: Nothing found ---
    print("⚠️ No results found at all.")
    return {
        "matchType": "none",
        "question": q,
        "answer": "I'm sorry, I couldn't find an answer to that question.",
        "coachingTip": ""
    }
