from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# --- CONNECT TO WEAVIATE & OPENAI (v3 client) ---
from weaviate import Client
from weaviate.auth import AuthApiKey

client = Client(
    url=os.getenv("WEAVIATE_CLUSTER_URL"),
    auth_client_secret=AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    additional_headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
)
# (no `collection = client.collections.get(...)` in v3)

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

    # 1. Exact match using a 'where' filter (v3)
    try:
        exact_res = (
            client.query
            .get("FAQ", ["question", "answer", "coachingTip"])
            .with_where({
                "path": ["question"],
                "operator": "Equal",
                "valueText": q
            })
            .with_limit(1)
            .do()
        )
        faq_list = exact_res.get("data", {}).get("Get", {}).get("FAQ", [])
        if faq_list:
            obj = faq_list[0]
            answer   = obj.get("answer", "").strip()
            coaching = obj.get("coachingTip", "").strip()

            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Question: {q}\n"
                f"{answer}\n"
                f"Coaching Tip: {coaching}"
            )
            print("Exact match found. Prompt sent to OpenAI:", repr(prompt))

            import time
            start = time.time()

            reply = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.5
            )

            end = time.time()
            print(f"⏱️ OpenAI response time: {end - start:.2f} seconds")

            content = reply.choices[0].message.content.strip()
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1]
            return content.replace("\\n", "\n").strip()

    except Exception as e:
        print("Exact-match (Python) error:", e)

    # 2. Vector search fallback (v3 syntax)
    try:
        vec_res = (
            client.query
            .get("FAQ", ["question", "answer", "coachingTip"])
            .with_near_text({"concepts": [q]})
            .with_additional(["distance"])
            .with_limit(1)
            .do()
        )
        faq_vec_list = vec_res.get("data", {}).get("Get", {}).get("FAQ", [])
        if faq_vec_list:
            obj = faq_vec_list[0]
            distance = obj.get("_additional", {}).get("distance", 1.0)
            if distance <= 0.6:
                answer   = obj.get("answer", "").strip()
                coaching = obj.get("coachingTip", "").strip()

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
                    max_tokens=500,
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
