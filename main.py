from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import weaviate
import os

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for all origins
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
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
)

collection = client.collections.get("FAQ")

@app.post("/faq")
async def get_faq(request: Request):
    body = await request.json()
    query = body.get("query", "").strip()
    print(f"✅ Received question: {query}")

    # Try exact match first
    try:
        exact = collection.query.bm25().with_where({
            "path": ["question"],
            "operator": "Equal",
            "valueText": query
        }).with_limit(1).do()

        if exact and exact.objects:
            obj = exact.objects[0]
            answer = obj.properties.get("answer", "")
            tip = obj.properties.get("coachingTip", "")
            print("🎯 Exact match found.")
            return {"answer": answer, "coachingTip": tip}
    except Exception as e:
        print(f"❌ Exact match error: {e}")

    # Fall back to vector search
    try:
        vector = collection.query.near_text(query=query, limit=1).do()
        if vector and vector.objects:
            obj = vector.objects[0]
            answer = obj.properties.get("answer", "")
            tip = obj.properties.get("coachingTip", "")
            print("🧠 Fallback vector search result used.")
            return {"answer": answer, "coachingTip": tip}
    except Exception as e:
        print(f"❌ Vector search error: {e}")

    return {
        "answer": "Sorry, I couldn’t find a good match for your question.",
        "coachingTip": ""
    }
