import os
import weaviate
from weaviate.classes.query import Filter
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import openai

# Set up FastAPI app
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
WEAVIATE_URL = os.environ.get("WEAVIATE_URL")
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
COLLECTION_NAME = "FAQ"

# Connect to Weaviate
client = weaviate.connect_to_wcs(
    cluster_url=WEAVIATE_URL,
    auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),
    headers={"X-OpenAI-Api-Key": OPENAI_API_KEY},
)

collection = client.collections.get(COLLECTION_NAME)


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    q = data.get("question", "").strip()
    user = data.get("user", "").strip().lower()

    if not q:
        return {"answer": "Please enter a question."}

    try:
        # Step 1: Exact match
        if user:
            exact_results = collection.query.fetch_objects(
                filters=Filter.by_all([
                    Filter.by_property("questionExact").equal(q),
                    Filter.by_property("user").equal(user)
                ]),
                limit=1
            )
        else:
            exact_results = collection.query.fetch_objects(
                filters=Filter.by_property("questionExact").equal(q),
                limit=1
            )

        if exact_results.objects:
            obj = exact_results.objects[0].properties
            return {
                "answer": obj["answer"],
                "coachingTip": obj.get("coachingTip", ""),
                "source": obj.get("source", "")
            }

        # Step 2: Vector search fallback
        near_text = collection.query.near_text(q, limit=1, filters=Filter.by_property("user").equal(user)) if user else collection.query.near_text(q, limit=1)
        if not near_text.objects:
            return {"answer": "Sorry, I couldn't find a relevant answer."}

        obj = near_text.objects[0].properties
        question = obj["question"]
        answer = obj["answer"]
        tip = obj.get("coachingTip", "")
        source = obj.get("source", "")

        # Step 3: Format with OpenAI
        openai.api_key = OPENAI_API_KEY
        messages = [
            {"role": "system", "content": "You are a helpful assistant who answers questions with clear facts followed by supportive, behavioral coaching tips. Format as Markdown with bold headers. Never invent information."},
            {"role": "user", "content": f"Q: {q}\n\nBest match:\nQ: {question}\nA: {answer}\nTip: {tip}"}
        ]

        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.4,
            max_tokens=800
        )

        response_text = completion.choices[0].message.content
        return {"answer": response_text, "source": source}

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"answer": "Sorry, something went wrong."}
