import weaviate 
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
import requests

import json
from pathlib import Path

ACCESS_MAP_PATH = os.getenv("ACCESS_MAP_PATH", "access_map.json")
with open(Path(ACCESS_MAP_PATH), "r", encoding="utf-8") as f:
    ACCESS_MAP = json.load(f)

def allowed_sources_for_request(request):
    tenant = request.headers.get("X-Tenant") or request.query_params.get("tenant") or "public"
    return ACCESS_MAP.get(tenant, ACCESS_MAP["public"])

def source_filter(allowed_sources: list[str]):
    return Filter.by_property("source").contains_any(allowed_sources)

def and_filters(*filters):
    filt_list = [f for f in filters if f is not None]
    if not filt_list:
        return None
    if len(filt_list) == 1:
        return filt_list[0]
    return Filter.all_of(filt_list)

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

PROTECTED_BRANDS = {"pendleton", "pendleton square"}  # lowercase
BRAND_TO_SOURCE = {
    "pendleton": "pendleton",
    "pendleton square": "pendleton",  # alias → source slug
}

def sanitize_question_for_disallowed_brands(question: str, allowed_sources: list[str]) -> str:
    allowed_lower = {s.lower() for s in allowed_sources}

    def replace_brand(q: str, brand: str, alias_patterns: list[tuple[str, str]]) -> str:
        for pattern, repl in alias_patterns:
            q = re.sub(pattern, repl, q, flags=re.IGNORECASE)
        return re.sub(r"\s{2,}", " ", q).strip()  # tidy extra spaces

    q = question
    for brand in PROTECTED_BRANDS:
        src_slug = BRAND_TO_SOURCE.get(brand, brand)
        if src_slug not in allowed_lower:
            # covers “<brand> trust”, possessives, and standalone brand
            patterns = [
                (rf"\b{re.escape(brand)}\s+trust('?s)?\b", "the trust company"),
                (rf"\b{re.escape(brand)}('?s)?\b", "the firm"),
            ]
            q = replace_brand(q, brand, patterns)
    return q


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
    allowed = allowed_sources_for_request(request)
    tenant_filt = source_filter(allowed)

    if not raw_q:
        raise HTTPException(status_code=400, detail="Missing 'query' in request body.")

    print(f"👤 User type: {requested_user}")
    print(f"Received question: {raw_q}")
    print(f"🔎 Checking exact match for normalized question: {q_norm}")


    try:
        user_filt = Filter.by_property("user").equal("both") | Filter.by_property("user").equal(requested_user)
        combined_filt = and_filters(user_filt, tenant_filt)

        filter = Filter.by_property("question").equal(raw_q.strip()) & combined_filt
        print("🔎 exact-match allowed_sources:", allowed)

        exact_res = collection.query.fetch_objects(
            filters=filter,
            return_properties=["question", "answer", "coachingTip", "source"],  # add source
            limit=3
        )
        print("📦 exact sources:", [o.properties.get("source") for o in exact_res.objects])

        for obj in exact_res.objects:
            db_q = obj.properties.get("question", "").strip()
            db_q_norm = normalize(db_q)
            if db_q_norm == q_norm:
                src = (obj.properties.get("source") or "").strip()
                if src not in allowed:
                    print("⛔ blocked exact-match source:", src, "allowed:", allowed)
                    continue
                print("✅ Exact match confirmed.")
                return {"response": format_response(obj)}

        print("⚠️ No strict match. Proceeding to vector search.")

    except Exception as e:
        print("Exact-match error:", e)

    try:
        user_filt = Filter.by_property("user").equal("both") | Filter.by_property("user").equal(requested_user)
        combined_filt = and_filters(user_filt, tenant_filt)

        vec_res = collection.query.near_text(
            query=raw_q,
            filters=combined_filt,
            return_metadata=["distance"],
            return_properties=["question", "answer", "coachingTip", "user", "source"],  # include source for debugging
            limit=3
        )
        objects = vec_res.objects
        print("📦 vector sources:", [o.properties.get("source") for o in objects])
        print(f"🔍 Retrieved {len(objects)} vector matches:")

        unique_faqs = []
        questions_seen = []
        for obj in objects:
            src = (obj.properties.get("source") or "").strip()
            if src not in allowed:
                print("⛔ blocked vector source:", src, "allowed:", allowed)
                continue

            if obj.properties.get("user", "").lower() not in [requested_user, "both"]:
                continue
            q_text = obj.properties.get("question", "").strip()
            is_duplicate = any(fuzz.ratio(q_text, seen_q) > 90 for seen_q in questions_seen)
            if not is_duplicate:
                unique_faqs.append(obj)
                questions_seen.append(q_text)

        print("🧾 used sources/questions:", [
        (o.properties.get("source"), o.properties.get("question")) for o in unique_faqs
        ])

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
            # ... after you build `combined` from blocks, right before prompt:
            safe_q = sanitize_question_for_disallowed_brands(raw_q, allowed)
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

# --- persona-classify (staging) — CLEAN BLOCK ---
# --- persona-classify (staging) — GUARDED BLOCK ---
from typing import Dict
from pydantic import BaseModel

class PersonaRequest(BaseModel):
    answers: Dict
    personasUrl: str

@app.post("/persona-classify")
def persona_classify(req: PersonaRequest):
    """
    Classify a user's answers into the best persona using OpenAI,
    with strict prompt guardrails and server-side validations.
    Returns: {"persona":{"id":...}, "meta":{"id":..., "confidence":..., "rationale":...}}
    """
    # 1) Load personas JSON from the provided URL
try;    
    fetch_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36 WhealthChat/1.0"
        ),
        "Accept": "application/json, text/plain, */*",
    }
    resp = requests.get(req.personasUrl, headers=fetch_headers, timeout=10)
    resp.raise_for_status()
    personas = resp.json()

    except Exception as e:
        return {
            "persona": {"id": "Error"},
            "meta": {"id": "Error", "confidence": 0, "rationale": f"Could not fetch personas: {e}"}
        }

    # 2) Build a compact catalog to keep the prompt stable
    catalog = []
    if isinstance(personas, list):
        for p in personas:
            pid = p.get("id") or p.get("name") or p.get("title")
            if not pid:
                continue
            catalog.append({
                "id": pid,
                "name": p.get("name") or pid,
                "overview": p.get("overview", ""),
                "keyCharacteristics": p.get("keyCharacteristics", []),
                "primaryGoals": p.get("primaryGoals", [])
            })
    else:
        for key, p in (personas or {}).items():
            pid = (p or {}).get("id") or (p or {}).get("name") or (p or {}).get("title") or key
            catalog.append({
                "id": pid,
                "name": (p or {}).get("name") or pid,
                "overview": (p or {}).get("overview", ""),
                "keyCharacteristics": (p or {}).get("keyCharacteristics", []),
                "primaryGoals": (p or {}).get("primaryGoals", [])
            })

    if not catalog:
        return {
            "persona": {"id": "Error"},
            "meta": {"id": "Error", "confidence": 0, "rationale": "No personas found in JSON"}
        }

    # 3) Prompt with explicit field limits + rules (no invented fields)
    allowed_ids = [p["id"] for p in catalog]
    FIELDS_ALLOWED = [
        "age","gender","marital_status","life_stage","health_status",
        "caregiving","risk_tolerance","planning_horizon","decision_style",
        "memory_status","life_event","primary_concerns"
    ]

    prompt = (
        "You are PersonaClassifier. Pick EXACTLY ONE persona from ALLOWED_IDS using ONLY FIELDS_ALLOWED.\n"
        "NEVER invent fields, ages, or personas.\n\n"

        "HARD CONSTRAINTS:\n"
        "- Use ONLY these user fields: " + ", ".join(FIELDS_ALLOWED) + ".\n"
        "- Do NOT infer children unless the user's 'caregiving' TEXT explicitly contains 'child' or 'children'.\n"
        "- 'Sandwich Generation Planner' REQUIRES BOTH: 'parent'/'parents' AND 'child'/'children' in caregiving text.\n"
        "- If age is provided and clearly conflicts with a persona’s typical profile, down-rank that persona.\n"
        "- Prefer 'Responsible Supporter' over 'Sandwich' when age is ~55–70 AND caregiving mentions parent(s) BUT NOT child/children.\n"
        "- 'Empowered Widow' REQUIRES widowhood signals (e.g., 'widow', 'death of spouse').\n"
        "- 'Business Owner Nearing Exit' REQUIRES business/exit/sale intent.\n"
        "- 'Self-Directed Investor' aligns with high risk tolerance + research/analyze decision style.\n"
        "- 'HENRY (High Earner, Not Rich Yet)' typically ~35–50 and working full-time; tax/earnings focus is a useful signal.\n"
        "- If NO persona satisfies constraints, return id:'no_match'.\n\n"

        "TIE-BREAKERS:\n"
        "1) Highest coverage of explicit user fields. 2) Age proximity (if present). 3) Goals/concerns alignment.\n\n"

        "OUTPUT STRICTLY AS JSON (no prose):\n"
        "{ \"id\": \"<one of ALLOWED_IDS or 'no_match'>\", \"confidence\": <0..1>, \"rationale\": \"<=30 words\" }\n\n"

        "USER_ANSWERS:\n" + json.dumps(req.answers, indent=2) + "\n\n"
        "ALLOWED_IDS:\n" + json.dumps(allowed_ids, indent=2) + "\n"
    )


# TODO: insert deterministic shortcuts here (e.g., Responsible Supporter) before building prompt


    # 4) Call OpenAI and force JSON output
    try:
        reply = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0,
            response_format={"type": "json_object"}
        )
        text = (reply.choices[0].message.content or "").strip()

        # Parse JSON (LLM guaranteed JSON mode, but we still guard)
        try:
            result = json.loads(text)
        except Exception as e:
            return {
                "persona": {"id": "Error"},
                "meta": {"id": "Error", "confidence": 0, "rationale": f"parse fail: {e}; raw starts: {text[:120]}"}
            }

        # 5) Server-side validations & corrections
        persona_id = result.get("id") or "Unknown"
        try:
            confidence = float(result.get("confidence", 0))
        except Exception:
            confidence = 0.0
        rationale = result.get("rationale") or "No rationale provided"

        # Clamp confidence to [0,1]
        confidence = max(0.0, min(1.0, confidence))

        # Guard A: id must be allowed or 'no_match'
        if persona_id not in allowed_ids and persona_id != "no_match":
            persona_id, confidence, rationale = "no_match", 0.0, "LLM returned an id not in catalog."

        # Compute caregiving flags once for Guard B
        careg = (str(req.answers.get("caregiving") or "")).lower()
        has_parent = ("parent" in careg or "parents" in careg)
        has_child  = ("child" in careg or "children" in careg)

        # ... after Guard B ...
        if persona_id == "Sandwich Generation Planner" and not (has_parent and has_child):
            persona_id, confidence, rationale = "no_match", 0.0, "Sandwich requires both parent and child caregiving."

        # Guard C: minimum confidence required
        MIN_CONFIDENCE = 0.40
        if confidence < MIN_CONFIDENCE:
            persona_id, confidence, rationale = "no_match", 0.0, f"Confidence below {MIN_CONFIDENCE}."

        # 6) Return in the shape your frontend expects
        return {
            "persona": {"id": persona_id},
            "meta": {"id": persona_id, "confidence": confidence, "rationale": rationale}
        }

    except Exception as e:
        return {
            "persona": {"id": "Error"},
            "meta": {"id": "Error", "confidence": 0, "rationale": f"AI classification failed: {e}"}
        }
# --- end guarded block ---


   
from fastapi.responses import JSONResponse

@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
def root():
    return JSONResponse({"status": "WhealthChat API is running"})
