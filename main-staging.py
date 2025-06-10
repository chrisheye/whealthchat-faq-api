from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import weaviate
import openai
import os
from weaviate import Client
from weaviate.auth import AuthApiKey

# --- PROMPT TEMPLATES ---
SYSTEM_PROMPT = (
    "You are a helpful assistant. Respond using Markdown with consistent formatting.\n"
    "Do NOT include the word 'Answer:' in your response.\n"
    "Bold the words 'Coaching Tip:' exactly as shown.\n"
    "Do not bold any other parts of the answer text.\n"
    "Keep 'Coaching Tip:' inline with the rest of the text, followed by a colon.\n"
    "Use line breaks only to separate paragraphs.\n"
    "If multiple Coaching Tips are provided, summarize them into ONE final Coaching Tip for the user."
)


# --- APP SETUP ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://whealthchat.ai",
        "https://staging.whealthchat.ai"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONNECT TO WEAVIATE & OPENAI ---
client = Client(
    url=os.getenv("WEAVIATE_CLUSTER_URL"),
    auth_client_secret=AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    additional_headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
)

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

    # 1. Exact match
    try:
        exact_res = (
            client.query
            .get("FAQ", ["question", "answer", "coachingTip"])
            .with_where({
                "path": ["question"],
                "operator": "Equal",
                "valueText": q
            })
            .with_limit(3)
            .do()
        )
        faq_list = exact_res.get("data", {}).get("Get", {}).get("FAQ", [])
        print(f"🔍 Exact match returned {len(faq_list)} result(s)")
        if faq_list:
            obj = faq_list[0]
            answer   = obj.get("answer", "").strip()
            coaching = obj.get("coachingTip", "").strip()
            print("✅ Exact match found. Returning answer without OpenAI call.")
            return f"{answer}\n\n**Coaching Tip:** {coaching}"
    except Exception as e:
        print("Exact-match (Python) error:", e)

    # 2. Vector search fallback with summarization
    try:
        vec_res = (
            client.query
            .get("FAQ", ["question", "answer", "coachingTip"])
            .with_near_text({"concepts": [q]})
            .with_additional(["distance"])
            .with_limit(3)
            .do()
        )
        faq_vec_list = vec_res.get("data", {}).get("Get", {}).get("FAQ", [])
        print(f"🔍 Retrieved {len(faq_vec_list)} vector matches:")

        from rapidfuzz import fuzz
        unique_faqs = []
        questions_seen = []

        for obj in faq_vec_list:
            q_text = obj.get("question", "").strip()
            is_duplicate = any(fuzz.ratio(q_text, seen_q) > 90 for seen_q in questions_seen)
            if not is_duplicate:
                unique_faqs.append(obj)
                questions_seen.append(q_text)

        faq_vec_list = unique_faqs
        print(f"🧹 After deduplication: {len(faq_vec_list)} match(es) kept.")

        for i, obj in enumerate(faq_vec_list):
            q_match = obj.get("question", "")
            d = obj.get("_additional", {}).get("distance", "?")
            print(f"{i+1}. {q_match} (distance: {d})")

        if faq_vec_list:
            blocks = []
            for i, obj in enumerate(faq_vec_list):
                answer   = obj.get("answer", "").strip()
                coaching = obj.get("coachingTip", "").strip()
                blocks.append(f"Answer {i+1}:\n{answer}\n\nCoaching Tip {i+1}: {coaching}")
            combined = "\n\n---\n\n".join(blocks)

            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Question: {q}\n\n"
                f"Here are multiple answers and coaching tips from similar questions. "
                f"Summarize them into a single helpful response for the user:\n\n{combined}"
            )
            print("🌀 Vector match found. Prompt sent to OpenAI:\n", repr(prompt))

            import time
            start = time.time()

            reply = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.5
            )

            end = time.time()
            print(f"⏱️ OpenAI response time: {end - start:.2f} seconds")

            content = reply.choices[0].message.content.strip()
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1]
            return content.replace("\\n", "\n").strip()
    except Exception as e:
        print("Vector-search error:", e)

    # 3. No match fallback
    return (
        "I do not possess the information to answer that question. "
        "Try asking me something about financial, retirement, estate, or healthcare planning."
    )
