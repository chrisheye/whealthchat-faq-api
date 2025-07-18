import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.query import Filter
from weaviate.classes.init import AdditionalConfig
from weaviate.classes.query import NearText
from weaviate.connect import ConnectionParams
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
    "If multiple Coaching Tips are provided, summarize them into ONE final Coaching Tip for the user."
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://whealthchat.ai", "https://staging.whealthchat.ai"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEAVIATE_CLUSTER_URL = os.environ.get("WEAVIATE_CLUSTER_URL", "")
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

client = weaviate.WeaviateClient(
    connection_params=ConnectionParams.from_url(WEAVIATE_CLUSTER_URL),
    auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
    additional_config=AdditionalConfig(
        headers={"X-OpenAI-Api-Key": OPENAI_API_KEY}
    )
)

openai.api_key = OPENAI_API_KEY

@app.get("/version")
def version_check():
    return {"status": "Running", "message": "✅ Using Weaviate v4"}

@app.post("/faq")
async def get_faq(request: Request):
    body = await request.json()
    user_type = body.get("user", "").strip().lower()
    q = body.get("query", "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Missing 'query' in request body.")
    q_norm = re.sub(r"[^\w\s]", "", q).lower()

    try:
        results = client.query.get("FAQ", ["question", "answer", "coachingTip"]).with_where(
            Filter.by_property("question").equal(q).by_property("user").any_of(["both", user_type])
        ).with_limit(3).do()

        faqs = results.data.get("FAQ", [])
        for obj in faqs:
            db_q = obj.get("question", "")
            if re.sub(r"[^\w\s]", "", db_q).lower().strip() == q_norm:
                print("✅ Exact match.")
                return f"{obj['answer'].strip()}\n\n**Coaching Tip:** {obj['coachingTip'].strip()}"

    except Exception as e:
        print("Exact-match error:", e)

    try:
        vector_results = client.query.get("FAQ", ["question", "answer", "coachingTip"])\
            .with_where(Filter.by_property("user").any_of(["both", user_type]))\
            .with_near_text(NearText(concepts=[q]))\
            .with_additional(["distance"]).with_limit(3).do()

        matches = vector_results.data.get("FAQ", [])
        seen = []
        unique = []
        for obj in matches:
            q_text = obj.get("question", "").strip()
            if not any(fuzz.ratio(q_text, s) > 90 for s in seen):
                unique.append(obj)
                seen.append(q_text)

        if unique and float(unique[0].get("_additional", {}).get("distance", 1)) <= 0.6:
            parts = []
            for i, o in enumerate(unique):
                parts.append(f"Answer {i+1}:\n{o['answer'].strip()}\n\nCoaching Tip {i+1}: {o['coachingTip'].strip()}")
            prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {q}\n\nHere are multiple answers and coaching tips:\n\n" + "\n\n---\n\n".join(parts)

            reply = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.5
            )
            return reply.choices[0].message.content.strip()

    except Exception as e:
        print("Vector-search error:", e)

    return (
        "I do not possess the information to answer that question. "
        "Try asking me something about financial, retirement, estate, or healthcare planning."
    )
