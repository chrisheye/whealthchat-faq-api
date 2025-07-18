import weaviate; print("✅ weaviate version:", weaviate.__version__)
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import weaviate
import openai
import os
import re
from weaviate.auth import AuthApiKey
from rapidfuzz import fuzz
import time

SYSTEM_PROMPT = (
    "You are a helpful assistant. Respond using Markdown with consistent formatting.\n"
    "Do NOT include the word 'Answer:' in your response.\n"
    "Bold the words 'Coaching Tip:' exactly as shown.\n"
    "Do not bold any other parts of the answer text.\n"
    "Keep 'Coaching Tip:' inline with the rest of the text, followed by a colon.\n"
    "Use line breaks to break long answers into clear, readable paragraphs – ideally no more than 3 sentences.\n"
    "Preserve all emojis in both the answer and the Coaching Tip exactly as they appear in the source material.\n"
    "Use a warm, supportive tone that acknowledges the emotional weight of sensitive topics like aging, illness, or financial stress.\n"
    "Avoid clinical or robotic phrasing. Use gentle, encouraging language that helps the user feel heard and empowered.\n"
    "Show empathy through wording — not by pretending to be human, but by offering reassurance and thoughtful framing of difficult issues.\n"
    "**If the original answers include links or downloads (e.g., checklists or tools), make sure to include those links in the final summarized answer. Do not omit them.**\n"
    "**Do not include links, downloads, or tools in the Coaching Tip — those must go in the main answer only.**\n"
    "**Preserve bold formatting from the source answers wherever it appears in the summary.**\n"
    "When appropriate, encourage users not to isolate themselves when facing difficult decisions. You may include the phrase **never worry alone** (in bold). Use sentence case unless it begins a sentence. Do not use the phrase in every response—only when it is contextually appropriate and feels natural.\n"
    "If multiple Coaching Tips are provided, summarize them into ONE final Coaching Tip for the user.\n"
    "If a long-term care calculator is mentioned, refer only to the custom calculator provided by WhealthChat — not generic online tools."
)

def normalize(text):
    return (
        text.lower().strip()
        .replace("’", "'").replace("‘", "'")
        .replace("“", '"').replace("”", '"')
        .replace("—", "-").replace("–", "-")
        .replace("…", "...")
    )

# --- APP SETUP ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://whealthchat.ai", "https://staging.whealthchat.ai"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEAVIATE_CLUSTER_URL = os.environ.get("WEAVIATE_CLUSTER_URL")
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# --- CONNECT TO WEAVIATE & OPENAI ---
client = weaviate.Client(
    url=WEAVIATE_CLUSTER_URL,
    auth_client_secret=AuthApiKey(WEAVIATE_API_KEY),
    additional_headers={"X-OpenAI-Api-Key": OPENAI_API_KEY}
)

openai.api_key = OPENAI_API_KEY

@app.get("/version")
def version_check():
    return {"status": "Running", "message": "✅ using weaviate.Client()"}

@app.post("/faq")
async def get_faq(request: Request):
    body = await request.json()
    raw_q = body.get("query", "").strip()
    user_type = body.get("user", "").strip().lower()

    if not raw_q:
        raise HTTPException(status_code=400, detail="Missing 'query' in request body.")

    print(f"👤 User type: {user_type}")
    print(f"Received question: {raw_q}")

    normalized = normalize(raw_q)

    # --- 1. EXACT MATCH ---
    try:
        exact_res = client.query.get("FAQ", ["question", "answer", "coachingTip"]).with_where({
            "operator": "And",
            "operands": [
                {"path": ["question"], "operator": "Equal", "valueText": raw_q},
                {
                    "path": ["user"],
                    "operator": "Or",
                    "operands": [
                        {"operator": "Equal", "valueText": "both"},
                        {"operator": "Equal", "valueText": user_type}
                    ]
                }
            ]
        }).with_limit(3).do()

        results = exact_res.get("data", {}).get("Get", {}).get("FAQ", [])
        for obj in results:
            db_norm = normalize(obj.get("question", ""))
            if db_norm == normalized:
                print("✅ Exact match found.")
                return f"{obj.get('answer','').strip()}\n\n**Coaching Tip:** {obj.get('coachingTip','').strip()}"

        print("⚠️ No exact match. Trying vector search.")
    except Exception as e:
        print("Exact match error:", e)

    # --- 2. VECTOR SEARCH ---
    try:
        vec_res = client.query.get("FAQ", ["question", "answer", "coachingTip"]).with_where({
            "operator": "Or",
            "operands": [
                {"path": ["user"], "operator": "Equal", "valueText": "both"},
                {"path": ["user"], "operator": "Equal", "valueText": user_type}
            ]
        }).with_near_text({"concepts": [raw_q]}).with_additional(["distance"]).with_limit(3).do()

        results = vec_res.get("data", {}).get("Get", {}).get("FAQ", [])
        unique = []
        seen = []

        for obj in results:
            q = obj.get("question", "").strip()
            if not any(fuzz.ratio(q, s) > 90 for s in seen):
                seen.append(q)
                unique.append(obj)

        if unique and float(unique[0].get("_additional", {}).get("distance", 1.0)) <= 0.6:
            blocks = [f"Answer {i+1}:\n{obj['answer'].strip()}\n\nCoaching Tip {i+1}: {obj['coachingTip'].strip()}" for i, obj in enumerate(unique)]
            prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {raw_q}\n\nHere are multiple answers and coaching tips from similar questions. Summarize them into a single helpful response for the user:\n\n" + "\n\n---\n\n".join(blocks)

            print("🌀 Sending to OpenAI")
            start = time.time()
            reply = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.5
            )
            print(f"⏱️ Time: {time.time() - start:.2f}s")
            return reply.choices[0].message.content.strip()
        else:
            print("❌ No strong vector match.")
    except Exception as e:
        print("Vector search error:", e)

    # --- 3. DEFAULT RESPONSE ---
    return (
        "I do not possess the information to answer that question. "
        "Try asking me something about financial, retirement, estate, or healthcare planning."
    )
