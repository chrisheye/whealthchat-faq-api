from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from weaviate import Client
from weaviate.auth import AuthApiKey
import weaviate
import openai
import os
import re
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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://whealthchat.ai", "https://staging.whealthchat.ai"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from weaviate import WeaviateClient, AuthApiKey

client = WeaviateClient.connect_to_wcs(
    cluster_url=os.getenv("WEAVIATE_CLUSTER_URL"),
    auth_credentials=AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
)


openai.api_key = os.getenv("OPENAI_API_KEY")

@app.get("/version")
def version_check():
    return {"status": "Running", "message": "✅ Fuzzy logic + user filter version"}

def normalize(text):
    return (
        text.lower()
        .strip()
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("—", "-")
        .replace("–", "-")
        .replace("…", "...")
    )

@app.post("/faq")
async def get_faq(request: Request):
    body = await request.json()
    raw_q = body.get("query", "").strip()
    user = body.get("user", "").strip().lower()

    if not raw_q or not user:
        raise HTTPException(status_code=400, detail="Missing 'query' or 'user'.")

    print(f"👤 User type: {user}")
    print(f"Received question: {raw_q}")
    normalized = normalize(raw_q)
    print(f"🔎 Checking exact match for normalized question: {normalized}")

    # --- 1. Exact Match ---
    try:
        exact = (
            client.query
            .get("FAQ", ["question", "answer", "coachingTip"])
            .with_where({
                "operator": "And",
                "operands": [
                    {"path": ["question"], "operator": "Equal", "valueText": raw_q},
                    {"path": ["user"], "operator": "Equal", "valueText": user}
                ]
            })
            .with_limit(1)
            .do()
        )
        items = exact.get("data", {}).get("Get", {}).get("FAQ", [])
        for obj in items:
            db_q = normalize(obj.get("question", ""))
            if db_q == normalized:
                print("✅ Exact match confirmed.")
                return f"{obj['answer'].strip()}\n\n**Coaching Tip:** {obj['coachingTip'].strip()}"
        print("⚠️ No strict match. Proceeding to vector search.")
    except Exception as e:
        print("Exact-match error:", e)

    # --- 2. Vector Fallback ---
    try:
        vector = (
            client.query
            .get("FAQ", ["question", "answer", "coachingTip"])
            .with_where({
                "path": ["user"],
                "operator": "Equal",
                "valueText": user
            })
            .with_near_text({"concepts": [raw_q]})
            .with_additional(["distance"])
            .with_limit(5)
            .do()
        )
        items = vector.get("data", {}).get("Get", {}).get("FAQ", [])
        print(f"🔍 Retrieved {len(items)} vector matches")

        seen = set()
        deduped = []
        for obj in items:
            q = obj.get("question", "")
            if all(fuzz.ratio(q, prev) <= 90 for prev in seen):
                deduped.append(obj)
                seen.add(q)
        print(f"🧪 Filtered to {len(deduped)} usable fuzzy matches")

        if deduped and float(deduped[0].get("_additional", {}).get("distance", 1.0)) <= 0.6:
            blocks = []
            for i, obj in enumerate(deduped):
                a = obj.get("answer", "").strip()
                c = obj.get("coachingTip", "").strip()
                blocks.append(f"Answer {i+1}:\n{a}\n\nCoaching Tip {i+1}: {c}")
            combined = "\n\n---\n\n".join(blocks)

            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Question: {raw_q}\n\n"
                f"Here are multiple answers and coaching tips from similar questions. "
                f"Summarize them into a single helpful response for the user:\n\n{combined}"
            )

            print("🌀 Sending summarization prompt to OpenAI...")
            start = time.time()
            reply = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.5,
            )
            end = time.time()
            print(f"⏱️ OpenAI response time: {end - start:.2f} seconds")

            text = reply.choices[0].message.content.strip()
            return text
        else:
            print("❌ No high-quality vector match. Returning fallback.")
    except Exception as e:
        print("Vector-search error:", e)

    return "I do not possess the information to answer that question. Try asking me something about financial, retirement, estate, or healthcare planning."
