import weaviate; print("✅ weaviate version:", weaviate.__version__)
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from weaviate.classes.query import Filter
from weaviate.classes.init import AdditionalConfig
from weaviate.auth import AuthApiKey
import weaviate
import openai
import os
import re
from rapidfuzz import fuzz
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    "If a long-term care calculator is mentioned, refer only to the custom calculator provided by WhealthChat — not generic online tools."
)

def normalize(text):
    return re.sub(r"[^\w\s]", "", text.lower().strip())

# --- APP SETUP ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://whealthchat.ai", "https://staging.whealthchat.ai"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEAVIATE_CLUSTER_URL = os.getenv("WEAVIATE_CLUSTER_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = weaviate.connect_to_wcs(
    cluster_url=WEAVIATE_CLUSTER_URL,
    auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
    additional_config=AdditionalConfig(grpc_port_experimental=50051)
)

openai.api_key = OPENAI_API_KEY
collection = client.collections.get("FAQ")
print("🔍 Available collections:", client.collections.list_all())

@app.get("/version")
def version_check():
    return {"status": "Running", "message": "✅ CORS enabled version"}

@app.post("/faq")
async def get_faq(request: Request):
    body = await request.json()
    raw_q = body.get("query", "").strip()
    requested_user = body.get("user", "").strip().lower()
    q_norm = normalize(raw_q)

    if not raw_q:
        raise HTTPException(status_code=400, detail="Missing 'query' in request body.")

    print(f"👤 User type: {requested_user}")
    print(f"Received question: {raw_q}")
    print(f"🔎 Checking exact match for normalized question: {q_norm}")

    try:
        filter = Filter.by_property("question").equal(raw_q.strip()) & (
            Filter.by_property("user").equal("both") | Filter.by_property("user").equal(requested_user)
        )
        exact_res = collection.query.fetch_objects(
            filters=filter,
            return_properties=["question", "answer", "coachingTip"],
            limit=3
        )
        for obj in exact_res.objects:
            db_q = obj.properties.get("question", "").strip()
            db_q_norm = normalize(db_q)
            if db_q_norm == q_norm:
                print("✅ Exact match confirmed.")
                return {"response": format_response(obj)}
        print("⚠️ No strict match. Proceeding to vector search.")
    except Exception as e:
        print("Exact-match error:", e)

    # ✅ Vector diagnostic step
    try:
        vec_res = collection.query.near_text(
            query=raw_q,
            return_metadata=["distance"],
            return_properties=["question", "answer", "coachingTip", "user"],
            limit=5
        )
        objects = vec_res.objects
        print("🔬 Raw vector results:")
        for i, obj in enumerate(objects):
            print(f"{i+1}. Q: {obj.properties.get('question', '')} | user: {obj.properties.get('user', '')} | distance: {obj.metadata.get('distance', '?')}")
        print(f"🧪 Retrieved {len(objects)} vector matches.")
        results = []
        for obj in objects:
            results.append({
                "question": obj.properties.get("question", ""),
                "user": obj.properties.get("user", ""),
                "distance": obj.metadata.get("distance", "?")
            })
        return {"vector_matches": results}
    except Exception as e:
        print("Vector-search error:", e)
        raise HTTPException(status_code=500, detail="Vector search failed.")

def format_response(obj):
    answer = obj.properties.get("answer", "").strip()
    tip = obj.properties.get("coachingTip", "").strip()
    if tip:
        return f"{answer}\n\n**Coaching Tip:** {tip}"
    return answer

@app.get("/faq-count")
def count_faqs():
    try:
        count = client.collections.get("FAQ").aggregate.over_all(total_count=True).metadata.total_count
        return {"count": count}
    except Exception as e:
        logger.exception("❌ Error counting FAQs")
        raise HTTPException(status_code=500, detail=str(e))
