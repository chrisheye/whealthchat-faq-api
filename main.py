from fastapi import FastAPI, Query
import weaviate
import os

# Initialize FastAPI app
app = FastAPI()

# Set up Weaviate client connection
client = weaviate.connect_to_wcs(
    cluster_url=os.getenv("WEAVIATE_CLUSTER_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
)

# Get the collection
collection = client.collections.get("Whealthchat_rag")

@app.get("/faq")
def get_faq(q: str = Query(...)):
    try:
        response = (
            collection.query
            .near_text(q)
            .with_limit(1)
            .with_generate(single_prompt=f"Answer this question:\n{q}")
            .do()
        )

        result = response["data"]["Get"]["Whealthchat_rag"][0]["_additional"]["generate"]["singleResult"]
        return result

    except Exception as e:
        return f"Sorry, I couldn’t find a good answer for that. Try rephrasing or ask about financial, retirement, estate, or healthcare planning."

