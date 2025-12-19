import weaviate; print("✅ weaviate version:", weaviate.__version__)
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from weaviate.classes.query import Filter
from weaviate.classes.init import AdditionalConfig
from weaviate.auth import AuthApiKey
import openai
import requests
import os
import re
from rapidfuzz import fuzz
import time
import logging
import asyncio
from contextlib import suppress
import json, os
from pathlib import Path

ACCESS_MAP_PATH = os.getenv("ACCESS_MAP_PATH", "access_map.json")
with open(Path(ACCESS_MAP_PATH), "r", encoding="utf-8") as f:
    ACCESS_MAP = json.load(f)


# ---- Persona file load (SAFE) ----
from pathlib import Path
import json
import os

PERSONAS_PATH = os.getenv("PERSONAS_PATH", "financial_personas.json")

_PERSONAS_RAW = []          # ✅ always defined
PERSONA_BY_ID = {}         # ✅ always defined

try:
    with open(Path(PERSONAS_PATH), "r", encoding="utf-8") as f:
        _PERSONAS_RAW = json.load(f)
    print(f"✅ Loaded personas file: {PERSONAS_PATH} (type={type(_PERSONAS_RAW).__name__})")
except Exception as e:
    print(f"⚠️ Could not load personas file: {PERSONAS_PATH} — {e}")
    _PERSONAS_RAW = []  # keep it safe

# Build PERSONA_BY_ID from whatever we loaded (list or dict)
if isinstance(_PERSONAS_RAW, list):
    for p in _PERSONAS_RAW:
        if isinstance(p, dict):
            pid = (p.get("id") or p.get("name") or "").strip()
            if pid:
                PERSONA_BY_ID[pid.lower()] = p
elif isinstance(_PERSONAS_RAW, dict):
    for k, p in _PERSONAS_RAW.items():
        if isinstance(p, dict):
            pid = (p.get("id") or p.get("name") or k or "").strip()
            if pid:
                PERSONA_BY_ID[pid.lower()] = p

print(f"🧠 PERSONA_BY_ID size: {len(PERSONA_BY_ID)}")
# ---- end persona load ----

SEEN_FAQ_CLIENTS = set()
SEEN_SESSIONS = set()


def allowed_sources_for_request(request):
    tenant = request.headers.get("X-Tenant") or request.query_params.get("tenant") or "public"
    allowed = ACCESS_MAP.get(tenant, ACCESS_MAP["public"])
    # ✅ Make the global source always allowed
    if "WhealthChat" not in allowed:
        allowed = allowed + ["WhealthChat"]
    return allowed

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
    "Then provide ONE Coaching Tip using this exact label: **💡 COACHING TIP:** (inline, not as a heading).\n\n"

    "STRICT COACHING TIP LIMITS:\n"
    "- Maximum 2 paragraphs total.\n"
    "- Each paragraph must be 1–2 sentences.\n"
    "- Do NOT repeat persona background already stated in the main answer.\n"
    "- Focus on advisor behavior, not client biography.\n"
    "- Do NOT include planning steps (e.g., consolidation, beneficiaries, timelines) in the Coaching Tip.\n\n"

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
    "Summarize multiple tips into one helpful, well-structured Coaching Tip for the user.\n"
    "If a long-term care calculator is mentioned, refer ONLY to the WhealthChat custom calculator."
)

def persona_fields_for_question(raw_q: str) -> dict:
    """
    Returns a dict:
      {"topic": "<label>", "fields": ["life_stage", "primary_concerns", ...]}
    The goal: only pass the *relevant* persona fields to the LLM for this question.
    """
    q = (raw_q or "").lower()

    TOPICS = [
        # Estate / authority / documents
        ("authority_docs", [
            r"\bpoa\b", r"power of attorney", r"healthcare proxy", r"proxy",
            r"advance directive", r"living will", r"trust", r"will\b",
            r"beneficiar", r"executor", r"trustee", r"incapac", r"guardianship"
        ], ["life_stage", "primary_concerns"]),

        # Health events / caregiving / LTC
        ("health_caregiving", [
            r"caregiv", r"memory care", r"assisted living", r"skilled nursing",
            r"long[-\s]?term care", r"\bltc\b", r"dementia", r"alz", r"stroke",
            r"hospital", r"diagnos", r"declin", r"cognitive"
        ], ["life_stage", "primary_concerns"]),

        # Cash-flow / income / retirement mechanics
        ("income_cashflow", [
            r"cash flow", r"budget", r"spend", r"debt", r"emergency fund",
            r"retire", r"social security", r"pension", r"annuit", r"rmd",
            r"withdraw", r"income", r"tax"
        ], ["primary_concerns", "decision_style"]),

        # Behavior / decision overwhelm / anxiety
        ("decision_behavior", [
            r"overwhelm", r"anx", r"stress", r"panic", r"avoid",
            r"confident", r"decision", r"procrast", r"motivat",
            r"impuls", r"regret"
        ], ["decision_style", "primary_concerns"]),
    ]

    for topic, patterns, fields in TOPICS:
        if any(re.search(p, q) for p in patterns):
            return {"topic": topic, "fields": fields}

    # Default: keep it light
    return {"topic": "general", "fields": ["decision_style"]}

PERSONAS_PATH = os.getenv("PERSONAS_PATH", "financial_personas.json")  # or whatever file you have
with open(PERSONAS_PATH, "r", encoding="utf-8") as f:
    _PERSONAS_RAW = json.load(f)

# Build an ID->persona dict that works for either list or dict JSON shapes
PERSONA_BY_ID = {}
if isinstance(_PERSONAS_RAW, list):
    for p in _PERSONAS_RAW:
        pid = (p.get("id") or p.get("name") or "").strip()
        if pid:
            PERSONA_BY_ID[pid.lower()] = p
elif isinstance(_PERSONAS_RAW, dict):
    for k, p in _PERSONAS_RAW.items():
        pid = ((p or {}).get("id") or (p or {}).get("name") or k or "").strip()
        if pid and isinstance(p, dict):
            PERSONA_BY_ID[pid.lower()] = p



def slice_persona(persona: dict, fields: list[str]) -> dict:
    """Return only the persona fields we want the model to see for this question."""
    if not isinstance(persona, dict) or not persona:
        return {}

    keep = {}

    # Always keep a usable name/id for referencing
    name = (persona.get("client_name") or persona.get("name") or persona.get("id") or "").strip()
    if name:
        keep["name"] = name

    # Only include allowed fields if present
    if "life_stage" in fields and (persona.get("life_stage") or "").strip():
        keep["life_stage"] = (persona.get("life_stage") or "").strip()

    if "primary_concerns" in fields and (persona.get("primary_concerns") or "").strip():
        keep["primary_concerns"] = (persona.get("primary_concerns") or "").strip()

    if "decision_style" in fields and (persona.get("decision_style") or "").strip():
        keep["decision_style"] = (persona.get("decision_style") or "").strip()

    return keep


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
    allow_origins=["https://whealthchat.ai", "https://staging.whealthchat.ai", "https://horizons.whealthchat.ai", "https://demo.whealthchat.ai","https://demo1.whealthchat.ai","https://pendleton.whealthchat.ai"],
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

# --- KEEP-ALIVE: keep Weaviate gRPC warm to avoid long first-query stalls -----
KEEPALIVE_INTERVAL_SEC = 150  # 2.5 minutes; you can use 120–240

async def _weaviate_keepalive_loop():
    # small delay so the app and client finish starting
    await asyncio.sleep(5)
    while True:
        try:
            coll = client.collections.get("FAQ")
            # cheap "are you there?" call
            _ = coll.aggregate.over_all(total_count=True)
            logging.info("Weaviate keep-alive: OK")
        except Exception as e:
            logging.warning(f"Weaviate keep-alive error: {e}")
        await asyncio.sleep(KEEPALIVE_INTERVAL_SEC)

@app.on_event("startup")
async def _start_keepalive():
    # fire-and-forget background task
    with suppress(Exception):
        asyncio.create_task(_weaviate_keepalive_loop())

@app.get("/version")
def version_check():
    return {"status": "Running", "message": "✅ CORS enabled version"}


async def rewrite_with_tone(text, audience_block, persona_block: str = ""):
    if not audience_block and not persona_block:
        return text

    persona_instruction = ""
    if persona_block.strip():
        persona_instruction = (
            "ADD ONE explicit persona-tailored sentence in the MAIN ANSWER (not the Coaching Tip).\n"
            "The rewritten answer may include ONE brief sentence that explicitly names the persona, using natural language such as "
            "'For a client like the <persona name>,' or 'When working with a <persona name>,'. "
            "Do NOT use the word 'lens' or describe a framework.\n"
            "Do NOT force a persona mention if it feels unnatural.\n"
        )

    prompt = (
        f"{audience_block}\n\n"
        f"{persona_block}\n\n"
        "Rewrite the following answer so it matches this audience"
        + (" and persona.\n" if persona_block.strip() else ".\n")
        f"{persona_instruction}"
        "It should reflect constraints around tone, pacing, support level, and examples.\n"
        "Keep all key facts and recommendations the same.\n"
        "Do NOT restate persona background.\n"
        "Do NOT add new tools or links, and do NOT remove any that are already present.\n\n"
        f"ANSWER:\n{text}"
    )


    reply = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=700,
        temperature=0
    )

    return reply.choices[0].message.content.strip()


async def finalize_response(
    text: str,
    row_user: str,
    audience_block: str,
    persona: dict,
    persona_block: str,
    raw_q: str = "",
    allow_rewrite: bool = True
) -> str:

    # 1) Rewrite tone ONLY if allowed (never for exact matches)
    if allow_rewrite:
        if persona_block:
            text = await rewrite_with_tone(text, audience_block, persona_block)
        elif audience_block:
            text = await rewrite_with_tone(text, audience_block)

    # 3) OPTIONAL: show persona label (visible) so you can confirm it’s being applied
    if persona and isinstance(persona, dict):
        pname = (persona.get("client_name") or persona.get("name") or persona.get("id") or "").strip()
        if pname:
            text = f"Persona applied: {pname}\n\n{text}"

    return text


def enforce_coaching_tip_rules(text: str) -> str:
    """
    Enforces Coaching Tip constraints WITHOUT changing retrieval or logic.
    Rules:
    - Max 3 paragraphs
    - Each paragraph max 3 sentences
    """
    marker = "**💡 COACHING TIP:**"
    if marker not in text:
        return text

    before, after = text.split(marker, 1)

    # Normalize paragraphs
    paragraphs = [p.strip() for p in after.split("\n\n") if p.strip()]

    trimmed_paragraphs = []
    for p in paragraphs[:3]:  # max 3 paragraphs
        sentences = re.split(r'(?<=[.!?])\s+', p)
        trimmed = " ".join(sentences[:3])  # max 3 sentences
        trimmed_paragraphs.append(trimmed)

    cleaned_tip = "\n\n".join(trimmed_paragraphs)

    return f"{before}{marker} {cleaned_tip}"




def is_default_persona(p: dict) -> bool:
    pid = (p.get("id") or "").strip().lower()
    nm  = (p.get("name") or p.get("client_name") or "").strip().lower()

    # Explicit placeholders only
    bad = {
        "", "default", "default persona", "persona",
        "select persona", "choose persona",
        "none", "no persona", "unknown", "n/a"
    }

    # 🚫 DO NOT treat "|template" as default
    if pid in bad or nm in bad:
        return True

    if pid.startswith("select") or nm.startswith("select"):
        return True

    return False

# --- Step B: topic-gated persona_block (no taglines) ---

def detect_topic(raw_q: str) -> str:
    q = (raw_q or "").lower()
    if any(t in q for t in ["poa", "power of attorney", "healthcare proxy", "incapac", "capacity", "diminish", "dementia", "alz"]):
        return "capacity"
    if any(t in q for t in ["caregiving", "caregiver", "memory care", "assisted living", "nursing home", "home care"]):
        return "caregiving"
    if any(t in q for t in ["long-term care", "ltc", "long term care", "care costs", "medicaid"]):
        return "care_costs"
    if any(t in q for t in ["will", "trust", "estate", "inheritance", "beneficiary", "probate"]):
        return "estate"
    if any(t in q for t in ["retirement", "income", "social security", "annuity", "rmd"]):
        return "retirement_income"
    return "general"

def persona_slice_for_topic(persona: dict, topic: str) -> dict:
    # Pull only the persona fields that matter for THIS topic.
    # (These keys assume your persona JSON already contains these fields;
    # missing keys just become empty strings.)
    def g(k): return (persona.get(k) or "").strip()

    common = {
        "persona_name": (persona.get("client_name") or persona.get("name") or persona.get("id") or "").strip(),
        "life_stage": g("life_stage"),
        "primary_concerns": g("primary_concerns"),
        "decision_style": g("decision_style"),
    }

    topic_map = {
        "capacity": {
            "capacity_triggers": g("capacity_triggers"),
            "trusted_contacts": g("trusted_contacts"),
            "decision_support": g("decision_support"),
        },
        "caregiving": {
            "caregiving_situation": g("caregiving_situation"),
            "caregiving_constraints": g("caregiving_constraints"),
            "family_dynamics": g("family_dynamics"),
        },
        "care_costs": {
            "cost_anxieties": g("cost_anxieties"),
            "coverage_gaps": g("coverage_gaps"),
            "risk_tolerance": g("risk_tolerance"),
        },
        "estate": {
            "legacy_goals": g("legacy_goals"),
            "family_complexity": g("family_complexity"),
            "document_readiness": g("document_readiness"),
        },
        "retirement_income": {
            "income_style": g("income_style"),
            "spending_patterns": g("spending_patterns"),
            "security_preferences": g("security_preferences"),
        },
        "general": {
            # keep it light: only common fields
        }
    }

    sliced = {**common, **topic_map.get(topic, {})}

    # Drop empty values so the block stays compact
    return {k: v for k, v in sliced.items() if str(v).strip()}


# --- end Step B ---

@app.post("/faq")
async def get_faq(request: Request):
    body = await request.json()
    session_id = request.headers.get("X-Session-Id") or body.get("session_id") or ""
    is_new_session = False
    if session_id and session_id not in SEEN_SESSIONS:
    SEEN_SESSIONS.add(session_id)
    is_new_session = True

    raw_q = body.get("query", "").strip()

    # 🧠 BACKEND FIRST-REQUEST GUARD (exact placement)
    client_ip = request.client.host if request.client else "unknown"
    ua = request.headers.get("user-agent", "unknown")
    client_key = f"{client_ip}|{ua}"

    first_time = client_key not in SEEN_FAQ_CLIENTS
    if first_time:
        SEEN_FAQ_CLIENTS.add(client_key)


    # ✅ SAFEGUARD: if frontend prepends persona text into query, extract the real question
    if raw_q.lower().startswith("persona context"):
        m = re.search(r"\b(what|how|why|when|where|who|can|should|do|is|are)\b.*$", raw_q, re.IGNORECASE)
        if m:
            raw_q = m.group(0).strip()


    print("📤 payload body:", body)  # <-- add this line
    print("🧾 RAW /faq body keys:", list(body.keys()))
    print("🧾 RAW query (first 200 chars):", raw_q[:200])
    print("🧾 RAW persona present?:", bool(body.get("persona")))
    print("🧾 RAW persona payload:", body.get("persona"))

    if raw_q.startswith("{"):
        return {"response": "⚠️ Invalid query format. Please ask a plain language question."}
    requested_user = body.get("user", "").strip().lower()
    q_norm = normalize(raw_q)
    allowed = allowed_sources_for_request(request)
    allowed_lower = {s.lower() for s in allowed}
    # make source filtering tolerant to case/spacing variants
    allowed_ci = list({
    s for a in allowed for s in (
        a,
        a.lower(),
        a.upper(),
        a.replace(" ", "").lower()
    ) if s
    })
    tenant_filt = source_filter(allowed_ci)

    
    # ---- Audience tone block (insert after requested_user parsing) ----
    audience_block = ""
    if requested_user == "professional":
        audience_block = (
            "You are advising a financial professional on how to talk with their clients.\n"
            "Write directly TO THE ADVISOR using 'you' for the advisor and 'your client' or 'your clients' when referring to the people they serve.\n"
            "Do NOT write as the advisor in the first person. Do NOT use 'I', 'we', or 'I'll' as if you are speaking to the client.\n"
            "When you give example phrases, introduce them as guidance to the advisor, for example: 'You might say, \"I know this can be a tough subject…\"'.\n"
            "Do NOT address the reader as if they are the client."
        )
    elif requested_user == "consumer":
        audience_block = (
            "You are advising an individual or family.\n"
            "Write directly to them using 'you'.\n"
            "Be clear, empathetic, and action-oriented with practical next steps."
        )

    # ---- Persona context block ----
    persona = body.get("persona")
# ✅ Clear persona automatically on new sessions
    if is_new_session:
        persona = {}

    # HARD BACKEND GUARD: drop placeholder/default personas
    if not isinstance(persona, dict):
        persona = {}

    pid = (persona.get("id") or "").strip().lower()
    pname = (persona.get("name") or persona.get("client_name") or "").strip().lower()

    if pid in {"", "default", "default persona"}:
        persona = {}


    # ✅ Drop placeholder persona so answers don't start with “default”
    if isinstance(persona, dict) and persona and is_default_persona(persona):
        persona = {}

    # ✅ ENRICH PERSONA: if frontend only sends {"id": "..."} load full persona fields
    # (Requires PERSONA_BY_ID to be defined at startup; I'll show that next if needed.)
    if isinstance(persona, dict) and persona:
        pid_full = (persona.get("id") or persona.get("name") or persona.get("client_name") or "").strip()
        if pid_full:
            full = PERSONA_BY_ID.get(pid_full.lower())
            if isinstance(full, dict):
                persona = {**full, **persona}  # keep any fields frontend already sent

    print("🧩 persona keys after enrichment:", sorted(list(persona.keys()))[:30])

    persona_block = ""  # define once

    raw_q_original = (body.get("query") or "").strip()
    persona_applied_in_query = raw_q_original.lower().startswith("persona context")
    # ✅ BACKEND FIX: ignore persona unless the query explicitly includes persona context
    if not persona_applied_in_query:
        persona = {}
        persona_block = ""


    def is_template_persona(p: dict) -> bool:
        pid = (p.get("id") or "").strip().lower()
        nm  = (p.get("name") or p.get("client_name") or "").strip().lower()
        return ("template" in pid) or ("template" in nm) or ("|template" in pid)

    def has_real_persona_fields(p: dict) -> bool:
        # Treat "id" as real so personas like {"id":"Resilient Partner"} are preserved
        return any([
            (p.get("id") or "").strip(),
            (p.get("client_name") or "").strip(),
            (p.get("name") or "").strip(),
            (p.get("life_stage") or "").strip(),
            (p.get("primary_concerns") or "").strip(),
            (p.get("decision_style") or "").strip(),
        ])

    # Step B: compute topic + persona_slice (topic-gated)
    topic = detect_topic(raw_q)
    persona_slice = persona_slice_for_topic(persona, topic)


    # Build persona_block only if persona is real (after the guard)
    if isinstance(persona, dict) and persona and has_real_persona_fields(persona):
        # Use only the topic-gated slice (not the full persona)
        persona_block = "Persona context (use lightly; do not restate):\n"
        persona_block += f"- Topic: {topic}\n"
        for k, v in persona_slice.items():
            persona_block += f"- {k}: {v}\n"
    else:
        persona = {}
        persona_block = ""

 
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
            return_properties=["question", "answer", "coachingTip", "source", "user"],
            limit=12
        )
        
        print("📦 exact sources:", [o.properties.get("source") for o in exact_res.objects])
        print("🧪 persona_applied_in_query:", persona_applied_in_query, "persona_kept:", bool(persona), "pid:", (persona.get("id") if isinstance(persona, dict) else None))

        for obj in exact_res.objects:
            db_q = obj.properties.get("question", "").strip()
            db_q_norm = normalize(db_q)
            if db_q_norm == q_norm:
                src = (obj.properties.get("source") or "").strip().lower()
                row_user = (obj.properties.get("user") or "").strip().lower()

                if src not in allowed_lower:
                    print("⛔ blocked exact-match source:", src, "allowed:", allowed)
                    continue
                print("✅ Exact match confirmed.")
                resp_text = format_response(obj)
                print("🧪 EXACT PATH persona_present:", bool(persona))
                resp_text = await finalize_response(
                    resp_text, row_user, audience_block, persona, persona_block,
                    raw_q=raw_q,
                    allow_rewrite=True
                )
              
                return {"response": resp_text}

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
        
        # 🔒 Exact-match override (case/punctuation tolerant)
        for obj in objects:
            db_q = (obj.properties.get("question") or "").strip()
            if normalize(db_q) == q_norm:
                src_ok = ((obj.properties.get("source") or "").strip().lower() in allowed_lower)
                row_user = (obj.properties.get("user","") or "").strip().lower()
                aud_ok = [requested_user, "both"]
                if requested_user == "professional":
                    aud_ok.append("professional")
                user_ok = (row_user in aud_ok)

                if src_ok and user_ok:
                    print("✅ Exact-match override via vector results.")
                    resp_text = format_response(obj)
                    resp_text = await finalize_response(
                        resp_text, row_user, audience_block, persona, persona_block,
                        raw_q=raw_q,
                        allow_rewrite=True
                    )


                    return {"response": resp_text}


        print("📦 vector sources:", [o.properties.get("source") for o in objects])
        print(f"🔍 Retrieved {len(objects)} vector matches:")

        unique_faqs = []
        questions_seen = []
        for obj in objects:
            src = (obj.properties.get("source") or "").strip()
            if src.lower() not in allowed_lower:
                print("⛔ blocked vector source (ci):", src, "allowed:", allowed)
                continue
            row_user = (obj.properties.get("user", "") or "").strip().lower()
            aud_ok = [requested_user, "both"]
            if requested_user == "professional":
                aud_ok.append("professional")
            if row_user not in aud_ok:
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
        
        # ----- RANKING RULES -----
        allowed_lower = {s.lower() for s in allowed}

        def is_brand_specific_question(q: str) -> bool:
            return re.search(r"\b(what are your|your|do you|can you|where are you|which do you|who are you)\b",
                             q, re.IGNORECASE) is not None

        def is_institutional_voice(ans: str) -> bool:
            # Heuristic for “we/our” voice that could impersonate a tenant
            return re.search(r"\b(we|our|our team|we offer|we provide|our services|our clients)\b",
                             ans, re.IGNORECASE) is not None

        ranked = []
        for obj in unique_faqs:
            src = (obj.properties.get("source") or "").strip()
            usr = (obj.properties.get("user") or "").strip().lower()
            qtxt = (obj.properties.get("question") or "").strip()
            dist = getattr(obj.metadata, "distance", 1.0)
            score = 1.0 - float(dist)

            answer_text = (obj.properties.get("answer") or "").strip()

            # Bonus: tenant-specific sources outrank global
            if src.lower() in allowed_lower and src.lower() != "whealthchat":
                score += 0.12

            # 🧠 Generic brand-specific guardrail (apply only to global content)
            if src.lower() == "whealthchat" and is_brand_specific_question(raw_q) and is_institutional_voice(answer_text):
                score -= 0.5

            ranked.append((score, obj))


        ranked.sort(key=lambda t: t[0], reverse=True)
        RANK_SCORE_MIN = 0.40
        top = [obj for sc, obj in ranked if sc >= RANK_SCORE_MIN][:3]
        print(f"📊 After ranking: {len(top)} kept above threshold {RANK_SCORE_MIN}")
        # --------------------------

        for i, obj in enumerate(unique_faqs):
            distance = getattr(obj.metadata, "distance", '?')
            print(f"{i+1}. {obj.properties.get('question', '')} (distance: {distance})")

        if top:
            blocks = []
            for i, obj in enumerate(top):

                answer = obj.properties.get("answer", "").strip()
                coaching = obj.properties.get("coachingTip", "").strip()
                blocks.append(f"Answer {i+1}:\n{answer}\n\nCoaching Tip {i+1}: {coaching}")
            combined = "\n\n---\n\n".join(blocks)
            safe_q = sanitize_question_for_disallowed_brands(raw_q, allowed)

            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"{audience_block}\n\n"
                f"{persona_block}\n\n"
                f"Question: {safe_q}\n\n"
                f"Here are multiple answers and coaching tips from similar questions, contained inside the block below.\n"
                f"The block is delimited by <<FAQ_BLOCK_START>> and <<FAQ_BLOCK_END>>.\n"
                f"Use ONLY that block as your source. Do NOT copy or repeat 'Answer 1', 'Answer 2', or 'Coaching Tip 3' literally; rewrite and summarize them instead.\n\n"
                f"1. Summarize the answers into one helpful response.\n"
                f"2. Then write ONE Coaching Tip with STRICT LIMITS: max 2 paragraphs, each 1–2 sentences.\n"
                f"3. Do NOT repeat persona background already stated in the main answer. Focus on advisor behavior.\n"
                f"4. The Coaching Tip should be clear, supportive, and behaviorally insightful, matching the correct audience (advisor or consumer).\n"
                f"5. ❌ Do NOT include any links, downloads, or tools in the Coaching Tip. Those belong in the answer only.\n\n"
                f"OUTPUT FORMAT (STRICT): Return ONLY JSON with this schema:\n"
                f'{{"answer_markdown":"...","coaching_tip_paragraphs":["p1","p2","p3"]}}\n'
                f"Rules for coaching_tip_paragraphs:\n"
                f"- MUST be an array of 1 to 3 strings.\n"
                f"- Each string must be 1 to 3 sentences.\n"
                f"- Do NOT include blank lines inside a paragraph string.\n"
                f"- Do NOT include the label '💡 COACHING TIP' inside the paragraphs.\n\n"
                f"<<FAQ_BLOCK_START>>\n{combined}\n<<FAQ_BLOCK_END>>"
            )


            print("Sending prompt to OpenAI.")
            reply = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=900,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            print("🧾 OpenAI raw JSON (first 500):", (reply.choices[0].message.content or "")[:500])
            
            # ✅ Extract JSON result
            data = json.loads((reply.choices[0].message.content or "").strip())

            answer_md = (data.get("answer_markdown") or "").strip()
            if not answer_md:
                answer_md = "I'm not finding a strong enough match in my knowledge base to answer that clearly."
            paras = data.get("coaching_tip_paragraphs") or []

            # ✅ Enforce: 1–3 paragraphs, each 1–3 sentences (server-side guard)
            cleaned_paras = []
            for p in paras[:3]:
                p = re.sub(r"\s+", " ", str(p).strip())
                if not p:
                    continue
                sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', p) if s.strip()]
                cleaned_paras.append(" ".join(sents[:3]))

            coaching_tip = "\n\n".join(cleaned_paras).strip()

            # ✅ Build final Markdown in your standard format
            text = answer_md
            if coaching_tip:
                text = f"{answer_md}\n\n**💡 COACHING TIP:** {coaching_tip}"

            # ✅ Apply persona injection (does NOT touch exact-match path)
            text = await finalize_response(
                text, "both", audience_block, persona, persona_block,
                raw_q=raw_q
            )

            return {"response": text}


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


# --- audience post-processor (temporary no-op) ---
def apply_audience_tone(text: str, audience: str) -> str:
    """Placeholder so apply_audience_tone() calls don't break execution."""
    return text
# -------------------------------------------------


def format_response(obj):
    answer = obj.properties.get("answer", "").strip()
    tip = obj.properties.get("coachingTip", "").strip()
    if tip:
        return f"{answer}\n\n**💡 COACHING TIP:** {tip}"
    return answer

def enforce_coaching_tip_limits(text: str) -> str:
    """
    Enforce strict limits on the Coaching Tip ONLY.
    Does not modify the main answer.
    """
    marker = "**💡 COACHING TIP:**"
    if marker not in text:
        return text

    before, after = text.split(marker, 1)

    # Split into paragraphs
    paragraphs = [p.strip() for p in after.strip().split("\n\n") if p.strip()]

    # Keep at most 2 paragraphs
    paragraphs = paragraphs[:2]

    cleaned_paragraphs = []
    for p in paragraphs:
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', p)
        # Keep at most 2 sentences per paragraph
        cleaned_paragraphs.append(" ".join(sentences[:2]))

    cleaned_tip = "\n\n".join(cleaned_paragraphs)
    return f"{before}{marker} {cleaned_tip}"

def format_coaching_tip_paragraphs(text: str) -> str:
    """
    Post-process ONLY the Coaching Tip so it's not a single blob.
    Rules:
    - Max 3 paragraphs
    - Each paragraph max 3 sentences
    - Does NOT modify the main answer
    """
    marker = "**💡 COACHING TIP:**"
    if marker not in text:
        return text

    before, after = text.split(marker, 1)
    tip = after.strip()

    # If the tip already has paragraph breaks, keep them and just enforce limits.
    if "\n\n" in tip:
        paragraphs = [p.strip() for p in tip.split("\n\n") if p.strip()]
    else:
        # Otherwise: split into sentences and rebuild paragraphs (3 sentences each)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', tip) if s.strip()]
        paragraphs = []
        for i in range(0, len(sentences), 3):
            paragraphs.append(" ".join(sentences[i:i+3]))

    # Enforce max 3 paragraphs
    paragraphs = paragraphs[:3]

    cleaned = "\n\n".join(paragraphs).strip()
    return f"{before}{marker} {cleaned}"


@app.get("/faq-count")
def count_faqs():
    try:
        resp = client.collections.get("FAQ").aggregate.over_all(total_count=True)
        return {"count": resp.total_count}
    except Exception as e:
        logger.exception("❌ Error counting FAQs")
        raise HTTPException(status_code=500, detail=str(e))


from typing import Dict
from pydantic import BaseModel

class PersonaRequest(BaseModel):
    answers: Dict
    personasUrl: str
# ✅ helper function (place this ABOVE @app.post("/persona-classify"))
def norm_answers(raw: dict) -> dict:
    """Map Formidable keys → clean fields, coerce types, and trim text."""
    def s(v): return (str(v or "").strip())
    def i(v):
        try:
            return int(v)
        except:
            return None

    return {
        "gender": s(raw.get("gender")),
        "age": i(raw.get("age")),
        "marital_status": s(raw.get("marital_status")),
        "life_stage": s(raw.get("life_stage")),
        "how_confident": i(raw.get("how_confident")),
        "planning_style": s(raw.get("planning_style")),
        "medical_conditions": s(raw.get("medical_conditions")),
    }
    
@app.post("/persona-classify")
def persona_classify(req: PersonaRequest):
    """
    Classify a user's answers into the best persona using OpenAI,
    with strict prompt guardrails and server-side validations.
    Returns: {"persona":{"id":...}, "meta":{"id":..., "confidence":..., "rationale":...}}
    """
    user_answers = norm_answers(req.answers)

    # 1) Load personas JSON from the provided URL
    try:
        fetch_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36 WhealthChat/1.0"
            ),
            "Accept": "application/json, text/plain, */*",
        }
        resp = requests.get(req.personasUrl, headers=fetch_headers, timeout=10)
        if resp.status_code >= 400:
            print(f"[personas fetch] {resp.status_code} {resp.reason} — url={req.personasUrl} body_start={resp.text[:200]}")
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
        "gender",
        "age",
        "marital_status",
        "life_stage",
        "how_confident",
        "planning_style",
        "medical_conditions",
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
        "- 'Solo Ager' REQUIRES clear signals of having no spouse AND no children. "
        "-  Do NOT classify as 'Empowered Widow' if there is no evidence of a past partnership.\n"
        "- 'Business Owner Nearing Exit' REQUIRES business/exit/sale intent.\n"
        "- 'Self-Directed Investor' aligns with high risk tolerance + research/analyze decision style.\n"
        "- 'HENRY (High Earner, Not Rich Yet)' typically ~35–50 and working full-time; tax/earnings focus is a useful signal.\n"
        "- If NO persona satisfies constraints, return id:'no_match'.\n\n"

        "TIE-BREAKERS:\n"
        "1) Highest coverage of explicit user fields. 2) Age proximity (if present). 3) Goals/concerns alignment.\n\n"

        "OUTPUT STRICTLY AS JSON (no prose):\n"
        "{ \"id\": \"<one of ALLOWED_IDS or 'no_match'>\", \"confidence\": <0..1>, \"rationale\": \"<=30 words\" }\n\n"

        "USER_ANSWERS:\n" + json.dumps(user_answers, indent=2) + "\n\n"
        "ALLOWED_IDS:\n" + json.dumps(allowed_ids, indent=2) + "\n"
    )


    # TODO: insert deterministic shortcuts here (e.g., Responsible Supporter) before building prompt
    # --- Deterministic shortcuts before calling OpenAI ---
    w = (user_answers.get("marital_status") or "").lower()
    m = (user_answers.get("medical_conditions") or "").lower()

    # Shortcut 1 – Empowered Widow
    if "widow" in w or "widowed" in w:
        return {
            "persona": {"id": "Empowered Widow"},
            "meta": {"id": "Empowered Widow", "confidence": 1.0, "rationale": "Marital status indicates widowhood."}
        }

    # Shortcut 2 – Diminished Decision-Maker
    if any(x in m for x in ["cognitive", "memory", "dementia", "alz"]):
        return {
            "persona": {"id": "Diminished Decision-Maker"},
            "meta": {"id": "Diminished Decision-Maker", "confidence": 1.0, "rationale": "Medical condition suggests cognitive decline."}
        }
        
    # Shortcut 3 – Business Owner Nearing Exit
    ls = (user_answers.get("life_stage") or "").lower()
    if any(k in ls for k in ["business owner", "owner", "succession", "exit", "selling business", "sale of business"]):
        return {
            "persona": {"id": "Business Owner Nearing Exit"},
            "meta": {"id": "Business Owner Nearing Exit", "confidence": 1.0,
                     "rationale": "Life stage indicates business ownership and exit/succession intent."}
        }

    # Shortcut 4 – Self-Directed Investor
    conf = user_answers.get("how_confident") or 0
    ps = (user_answers.get("planning_style") or "").lower()

    if conf >= 4 and any(k in ps for k in ["self", "independent", "do it myself", "analyze", "research", "hands-on"]):
        return {
            "persona": {"id": "Self-Directed Investor"},
            "meta": {"id": "Self-Directed Investor", "confidence": 0.95,
                     "rationale": "High confidence and self-directed planning style."}
        }

    # Shortcut 5 – Responsible Supporter
    ls = (user_answers.get("life_stage") or "").lower()
    age = user_answers.get("age") or 0

    if any(k in ls for k in ["caring for parent", "caregiving for parent", "supporting parent", "aging parent"]) \
       or ("caregiver" in ls and "parent" in ls):
        # gentle age nudge, but not required
        if 50 <= age <= 72 or age == 0:
            return {
                "persona": {"id": "Responsible Supporter"},
                "meta": {"id": "Responsible Supporter", "confidence": 0.9,
                         "rationale": "Life stage indicates caregiving for a parent."}
            }

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

