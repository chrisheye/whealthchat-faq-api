from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import weaviate
import os

app = FastAPI()

# Allow CORS for all origins (adjust if needed)
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

    # Try to find exact match (case-insensitive)
    results = collection.query.where(
        filter={"path": ["question"], "operator": "Equal", "valueText": query}
    ).with_limit(1).do()

    if results.objects:
        obj = results.objects[0]
        answer = obj.properties.get("answer", "")
        tip = obj.properties.get("coachingTip", "")
        return {"answer": answer, "coachingTip": tip}
    else:
        return {"answer": "No exact match found.", "coachingTip": ""}


