from fastapi import FastAPI, Query
import pandas as pd
import os
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.init import WeaviateClientConfig
from weaviate.classes.query import Filter
from weaviate.classes.query import MetadataQuery

app = FastAPI()

# === Weaviate Configuration ===
WEAVIATE_URL = "https://wm1lowcq6jljdkqldaxq.c0.us-west3.gcp.weaviate.cloud"
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")  # make sure this is set in Render secrets
OPENAI_API_KEY = os.getenv("OPENAI_APIKEY")       # also must be set in Render secrets

client = weaviate.WeaviateClient(
    config=WeaviateClientConfig(
        endpoint=WEAVIATE_URL,
        auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
        headers={"X-OpenAI-Api-Key": OPENAI_API_KEY}
    )
)

COLLECTION_NAME = "Whealthchat_rag"

@app.get("/faq")
def faq(q: str = Query(...)):
    # Perform a vector search with hybrid scoring
    try:
        response = client.collections.get(COLLECTION_NAME).query.near_text(
            query=q,
            limit=1,
            return_metadata=MetadataQuery(distance=True),
        )

        if not response.objects:
            return "I do not possess the information to answer that question. Try asking something about financial, estate, or healthcare planning."

        obj = response.objects[0]
        question = obj.properties["question"]
        answer = obj.properties["answer"]
        tip = obj.properties.get("coachingTip", "").strip()

        result = f"**{question}**\n\n{answer}"
        if tip:
            result += f"\n\n**Coaching Tip:** {tip}"

        return result

    except Exception as e:
        return f"An error occurred: {str(e)}"
