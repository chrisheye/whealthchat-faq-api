from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from weaviate.collections.classes.filters import Filter
import weaviate
import openai
import os

# --- PROMPT TEMPLATES ---
SYSTEM_PROMPT = (
    "You are a helpful assistant. Respond using Markdown with consistent formatting.\n"
    "Do NOT include the word 'Answer:' in your response.\n"
    "Bold the words 'Coaching Tip:' exactly as shown.\n"
    "Do not bold any other parts of the answer text.\n"
    "Keep 'Coaching Tip:' inline with the rest of the text, followed by a colon.\n"
    "Use line breaks only to separate paragraphs."
)

# --- APP SETUP ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://whealthchat.ai"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONNECT TO WEAVIATE & OPENAI ---
client = weaviate.connect_to_wcs(
    cluster_url=os.getenv("WEAVIATE_CLUSTER_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
)
collection = client.collections.get("FAQ")

openai.api_key = os.getenv("OPENAI_API_KEY")

# --- HEALTH CHECK ---
@app.get("/version")
def version_check():
    return {"status": "Running", "message": "✅ CORS enabled version"}

# --- FAQ ENDPOINT ---
@app.post("/faq")
async def get_faq(request: Request):
    body = await request.json()
    q = body.get("query", "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Missing 'query' in request body.")
    print(f"Received question: {q}")

    # 1. Exact match on questionExact
    try:
        filters = Filter.by_property("question").equal(q)
        exact = collection.query.fetch_objects(
            filters=filters,
            limit=1
        )
        if exact.objects:
            props = exact.objects[0].properties
            answer = props.get("answer", "").strip()
            coaching = props.get("coachingTip", "").strip()

            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Question: {q}\n"
                f"{answer}\n"
                f"Coaching Tip: {coaching}"
            )
            print("Exact match found. Prompt sent to OpenAI:", repr(prompt))

            reply = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.5
            )
            content = reply.choices[0].message.content.strip()

            # unwrap quotes and fix escaped newlines
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1]
            return content.replace("\\n", "\n").strip()
    except Exception as e:
        print("Exact-match error:", e)

    # 2. Vector search fallback
    try:
        response = collection.query.near_text(
            query=q,
            limit=1,
            return_metadata=["distance"]
        )
        print("Weaviate vector response:", response)

        if response.objects:
            obj = response.objects[0]
            distance = getattr(obj.metadata, "distance", 1.0)
            if distance <= 0.6:
                props = obj.properties
                answer = props.get("answer", "").strip()
                coaching = props.get("coachingTip", "").strip()

                prompt = (
                    f"{SYSTEM_PROMPT}\n\n"
                    f"Question: {q}\n"
                    f"{answer}\n"
                    f"Coaching Tip: {coaching}"
                )
                print("Vector match found. Prompt sent to OpenAI:", repr(prompt))

                reply = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400,
                    temperature=0.5
                )
                content = reply.choices[0].message.content.strip()
                if content.startswith('"') and content.endswith('"'):
                    content = content[1:-1]
                return content.replace("\\n", "\n").strip()
    except Exception as e:
        print("Vector-search error:", e)

    # 3. No match
    return (
        "I do not possess the information to answer that question. "
        "Try asking me something about financial, retirement, estate, or healthcare planning."
    )
