from fastapi import FastAPI, Query
import weaviate
import os

# === CONFIG ===
WEAVIATE_URL = "https://wm1lowcq6jljdkqldaxq.c0.us-west3.gcp.weaviate.cloud"
WEAVIATE_API_KEY = "8Slc79F6PAqNuyhTspOK5kDJURBqgzwIdX26"
OPENAI_API_KEY = os.getenv("OPENAI_APIKEY")
COLLECTION_NAME = "Whealthchat_rag"
SIMILARITY_THRESHOLD = 0.75

# === INIT ===
app = FastAPI()

client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY),
    additional_headers={"X-Openai-Api-Key": OPENAI_API_KEY}
)

# === ENDPOINT ===
@app.get("/faq")
def get_faq(q: str = Query(...)):
    response = client.query.get(COLLECTION_NAME, ["question", "answer", "coachingTip"])\
        .with_near_text({"concepts": [q]})\
        .with_limit(1)\
        .with_additional(["certainty"])\
        .do()

    try:
        result = response["data"]["Get"][COLLECTION_NAME][0]
        certainty = result["_additional"]["certainty"]

        if certainty >= SIMILARITY_THRESHOLD:
            answer = result["answer"].strip()
            coaching_tip = result.get("coachingTip", "").strip()

            final_response = f"{answer}"
            if coaching_tip:
                final_response += f"\n\nCoaching Tip: {coaching_tip}"

            return final_response

        else:
            return (
                "I’m not confident enough to answer that. "
                "Try asking something related to financial, estate, retirement, or healthcare planning."
            )

    except Exception as e:
        return f"An error occurred: {str(e)}"
