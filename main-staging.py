from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import weaviate
import openai
import os
from weaviate import Client
from weaviate.auth import AuthApiKey

# --- PROMPT TEMPLATE ---
SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the question and the related answers to create a concise, unified summary.
"
    "Format using Markdown. Bold the words 'Coaching Tip:' but nothing else.
"
    "If answers conflict or overlap, synthesize the most important ideas into a clear and accurate response.
"
    "Add only one Coaching Tip at the end, drawing from the tips provided."
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
    return {"status": "Running", "message": "✅ Staging with multi-answer summarization"}

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
            .with_where({"path": ["question"], "operator": "Equal", "valueText": q})
            .with_limit(1)
            .do()
        )
        faq_list = exact_res.get("data", {}).get("Get", {}).get("FAQ", [])
        if faq_list:
            obj = faq_list[0]
            answer = obj.get("answer", "").strip()
            coaching = obj.get("coachingTip", "").strip()
            print("✅ Exact match found. Returning answer without OpenAI call.")
            return f"{answer}\n\n**Coaching Tip:** {coaching}"
    except Exception as e:
        print("Exact-match error:", e)

    # 2. Vector fallback with summary of top 3
    try:
        vec_res = (
            client.query
            .get("FAQ", ["question", "answer", "coachingTip"])
            .with_near_text({"concepts": [q]})
            .with_additional(["distance"])
            .with_limit(3)
            .do()
        )
        faq_list = vec_res.get("data", {}).get("Get", {}).get("FAQ", [])
        if not faq_list:
            raise ValueError("No vector matches found.")

        combined_answers = ""
        combined_tips = ""
        for i, obj in enumerate(faq_list, 1):
            a = obj.get("answer", "").strip()
            t = obj.get("coachingTip", "").strip()
            combined_answers += f"Answer {i}: {a}\n"
            combined_tips += f"Coaching Tip {i}: {t}\n"

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Question: {q}\n\n"
            f"{combined_answers}\n"
            f"{combined_tips}"
        )

        print("🌀 Sending multi-answer summary prompt to OpenAI")
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
        print("Vector-summary error:", e)

    # 3. No match
    return (
        "I do not possess the information to answer that question. "
        "Try asking me something about financial, retirement, estate, or healthcare planning."
    )
