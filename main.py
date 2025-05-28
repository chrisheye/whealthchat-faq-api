from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import weaviate
import os
import openai
from weaviate.collections.classes.filters import Filter

app = FastAPI()

@app.get("/version")
def version_check():
    return {"status": "Running", "message": "✅ CORS enabled version"}

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://whealthchat.ai"],
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
    q = body.get("query", "").strip()
    print(f"✅ Received question: {q}")

    # Step 1: Exact match test with debug info
    filters = Filter.by_property("questionExact").equal(q)
    print("🔍 Running exact match filter:", filters)

    exact_match = collection.query.fetch_objects(
        filters=filters,
        limit=3
    )

    print("🧠 Exact match results:")
    for obj in exact_match.objects:
        print("-", obj.properties.get("question"))

    if exact_match.objects:
        obj = exact_match.objects[0]
        answer = obj.properties.get("answer", "")
        tip = obj.properties.get("coachingTip", "")
        return {"answer": answer, "coachingTip": tip}
    else:
        return {"answer": "No exact match found.", "coachingTip": ""}
