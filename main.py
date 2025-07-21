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
import time
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant. Respond using Markdown with consistent formatting.\n\n"
    "Answer the user's question clearly and supportively.\n"
    "Then provide ONE **Coaching Tip** in bold using this exact label: **Coaching Tip:** (inline, not as a heading).\n\n"
    "🚫 Do NOT include checklists, links, downloads, or tools in the Coaching Tip. Those belong in the main answer ONLY.\n"
    "✅ Preserve links and bold formatting in the main answer.\n"
    "✅ Include emojis if they appear in the source content.\n\n"
    "🔁 FORMATTING RULES:\n"
    "1. Break both the main answer and the Coaching Tip into short, readable paragraphs.\n"
    "2. Use line breaks between paragraphs.\n"
    "3. No paragraph should be more than 3 sentences long.\n"
    "4. NEVER place links or tools inside the Coaching Tip.\n\n"
    "💬 TONE:\n"
    "Use warm, encouraging language. Avoid robotic or clinical phrasing.\n"
    "Acknowledge that many users are navigating emotional or sensitive topics.\n"
    "Encourage users to seek help and **never worry alone** when appropriate.\n\n"
    "**IMPORTANT REMINDER:**\n"
    "Break long Coaching Tips into multiple short paragraphs, each no more than 3 sentences.\n"
    "Summarize multiple tips into one helpful, well-structured Coaching Tip for the user.\n"
    "If a long-term care calculator is mentioned, refer ONLY to the WhealthChat custom calculator."
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
    headers={"X-OpenAI-Api-Key": OPENAI_API_KEY},
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

    try:
        vec_res = collection.query.near_text(
            query=raw_q,
            return_metadata=["distance"],
            return_properties=["question", "answer", "coachingTip", "user"],
            limit=4
        )
        objects = vec_res.objects
        print(f"🔍 Retrieved {len(objects)} vector matches:")

        unique_faqs = []
        questions_seen = []
        for obj in objects:
            if obj.properties.get("user", "").lower() not in [requested_user, "both"]:
                continue
            q_text = obj.properties.get("question", "").strip()
            is_duplicate = any(fuzz.ratio(q_text, seen_q) > 90 for seen_q in questions_seen)
            if not is_duplicate:
                unique_faqs.append(obj)
                questions_seen.append(q_text)

        print(f"🫹 After filtering and deduplication: {len(unique_faqs)} match(es) kept.")

        for i, obj in enumerate(unique_faqs):
            distance = getattr(obj.metadata, "distance", '?')
            print(f"{i+1}. {obj.properties.get('question', '')} (distance: {distance})")


        if unique_faqs and getattr(unique_faqs[0].metadata, "distance", 1.0) <= 0.6:
            blocks = []
            for i, obj in enumerate(unique_faqs):
                answer = obj.properties.get("answer", "").strip()
                coaching = obj.properties.get("coachingTip", "").strip()
                blocks.append(f"Answer {i+1}:\n{answer}\n\nCoaching Tip {i+1}: {coaching}")
            combined = "\n\n---\n\n".join(blocks)
            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Question: {raw_q}\n\n"
                f"Here are multiple answers and coaching tips from similar questions.\n\n"
                f"1. Summarize the answers into one helpful response.\n"
                f"2. Then write ONE Coaching Tip that is no more than 3 sentences long. It should be clear, supportive, and behaviorally insightful.\n"
                f"3. 👉 Break all text into readable paragraphs of no more than 3 sentences each — especially the Coaching Tip.\n"
                f"4. ❌ Do NOT include any links, downloads, or tools in the Coaching Tip. Those belong in the answer only.\n\n"
                f"{combined}"
            )

            print("Sending prompt to OpenAI.")
            reply = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.5
            )
            return {"response": reply.choices[0].message.content.strip()}
        else:
            print("❌ No high-quality vector match. Returning fallback message.")
    except Exception as e:
        print("Vector-search error:", e)

    return {
        "response": (
            "I do not possess the information to answer that question. "
            "Try asking me something about financial, retirement, estate, or healthcare planning."
        )
    }

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

   
from fastapi.responses import JSONResponse

@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
def root():
    return JSONResponse({"status": "WhealthChat API is running"})

