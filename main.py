from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import weaviate
import os

# Create FastAPI app
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
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
)

collection = client.collections.get("FAQ")

@app.post("/faq")
async def get_faq(request: Request):
    body = await request.json()
    question = body.get("query", "").strip()
    print(f"✅ Received question: {question}")

    # TEMP: Just return a stub response for now
    return {
        "answer": "We received your question, but exact match and vector logic aren't implemented yet.",
        "coachingTip": "This is just a placeholder."
    }

