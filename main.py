from fastapi import FastAPI, Query
import os
import weaviate
import openai

from fastapi.middleware.cors import CORSMiddleware

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
    cluster_url=os.getenv("WEAVIATE_CLUSTER_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
)

collection = client.collections.get("FAQ")

@app.get("/faq")
def get_faq(q: str = Query(...)):
response = collection.query.near_text(
    query=q,
    limit=1,
    return_metadata=["distance"],
    return_properties=["question", "answer", "coachingTip"]
)
    if not response.objects:
        return "I do not possess the information to answer that question. Try asking me something about financial, retirement, estate, or healthcare planning."

    obj = response.objects[0]
    distance = obj.metadata.distance if obj.metadata and hasattr(obj.metadata, "distance") else 1.0

    if distance > 0.45:
        return "I do not possess the information to answer that question. Try asking me something about financial, retirement, estate, or healthcare planning."

    props = obj.properties
    question = props.get("question", "").strip()
    answer = props.get("answer", "").strip()
    coaching_tip = props.get("coachingTip", "").strip()

    prompt = (
        "You are a helpful assistant. Respond in plain text only. Do not use Markdown, bullets, or HTML.\n\n"
        "Always use the provided answer exactly as written. "
        "If a coaching tip is included, repeat it under a heading 'Coaching Tip.'\n\n"
        f"Question: {q}\n"
        f"Answer: {answer}\n"
        f"Coaching Tip: {coaching_tip}"
    )

    try:
        reply = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.5
        )
        clean_response = reply.choices[0].message.content.strip()

        # 🛠 Clean up the output properly
        if clean_response.startswith('"') and clean_response.endswith('"'):
            clean_response = clean_response[1:-1]

        clean_response = clean_response.replace("\\n", "\n").strip()

        return clean_response
    except Exception as e:
        return f"An error occurred: {str(e)}"
