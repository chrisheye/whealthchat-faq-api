from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import weaviate
import openai
import os
import re
from weaviate import Client
from weaviate.auth import AuthApiKey
from rapidfuzz import fuzz
import time

# --- PROMPT TEMPLATES ---
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
    "**If the original answers include links or downloads (e.g., checklists or tools), make sure to include those links in the final summarized answer. Do not omit them.**"
    "**Do not include links, downloads, or tools in the Coaching Tip — those must go in the main answer only.**\n"
    "**Preserve bold formatting from the source answers wherever it appears in the summary.**\n"
    "Always keep the phrase “**Never worry alone**” in bold exactly as shown, even when rewriting or summarizing content."
    "If multiple Coaching Tips are provided, summarize them into ONE final Coaching Tip for the user."
    "If a long-term care calculator is mentioned, refer only to the custom calculator provided by WhealthChat — not generic online tools."
)
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

# --- APP SETUP ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://whealthchat.ai", "https://staging.whealthchat.ai"],
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
    raw_q = body.get("query", "").strip()
    q_norm = re.sub(r"[^\w\s]", "", raw_q).lower()

    if not raw_q:
        raise HTTPException(status_code=400, detail="Missing 'query' in request body.")

    print(f"Received question: {raw_q}")
    print(f"🔎 Checking exact match for normalized question: {q_norm}")

    # 1. Exact match
    try:
        # Pull all FAQs so we can compare normalized text
        all_faqs = (
            client.query
            .get("FAQ", ["question", "answer", "coachingTip"])
            .with_limit(500)  # Adjust based on your total count
            .do()
        )
        faq_list = all_faqs.get("data", {}).get("Get", {}).get("FAQ", [])
    
        clean_q = normalize(raw_q)
        strict_match = next(
            (obj for obj in faq_list if normalize(obj.get("question", "")) == clean_q),
            None
        )
    
        if strict_match:
            print("✅ Exact match confirmed. Returning answer without OpenAI call.")
            answer = strict_match.get("answer", "").strip()
            coaching = strict_match.get("coachingTip", "").strip()
            return f"{answer}\n\n**Coaching Tip:** {coaching}"
        else:
            print("⚠️ No strict match found. Will proceed to vector search.")

except Exception as e:
    print("Exact-match error:", e)


    # 2. Vector search fallback with summarization
    try:
        vec_res = (
            client.query
            .get("FAQ", ["question", "answer", "coachingTip"])
            .with_near_text({"concepts": [raw_q]})
            .with_additional(["distance"])
            .with_limit(3)
            .do()
        )
        faq_vec_list = vec_res.get("data", {}).get("Get", {}).get("FAQ", [])
        print(f"🔍 Retrieved {len(faq_vec_list)} vector matches:")

        unique_faqs = []
        questions_seen = []
        for obj in faq_vec_list:
            q_text = obj.get("question", "").strip()
            is_duplicate = any(fuzz.ratio(q_text, seen_q) > 90 for seen_q in questions_seen)
            if not is_duplicate:
                unique_faqs.append(obj)
                questions_seen.append(q_text)

        faq_vec_list = unique_faqs
        print(f"🪩 After deduplication: {len(faq_vec_list)} match(es) kept.")

        for i, obj in enumerate(faq_vec_list):
            q_match = obj.get("question", "")
            d = obj.get("_additional", {}).get("distance", "?")
            print(f"{i+1}. {q_match} (distance: {d})")

        if faq_vec_list and float(faq_vec_list[0].get("_additional", {}).get("distance", 1.0)) <= 0.6:
            blocks = []
            for i, obj in enumerate(faq_vec_list): 
                answer = obj.get("answer", "").strip()
                coaching = obj.get("coachingTip", "").strip()
                blocks.append(f"Answer {i+1}:\n{answer}\n\nCoaching Tip {i+1}: {coaching}")

            combined = "\n\n---\n\n".join(blocks)
            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Question: {raw_q}\n\n"
                f"Here are multiple answers and coaching tips from similar questions. Summarize them into a single helpful response for the user:\n\n{combined}"
            )

            print("🌀 Vector match found. Prompt sent to OpenAI:\n", repr(prompt))
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
        else:
            print("❌ No high-quality vector match. Returning fallback message.")

    except Exception as e:
        print("Vector-search error:", e)

    # 3. No match fallback
    return (
        "I do not possess the information to answer that question. "
        "Try asking me something about financial, retirement, estate, or healthcare planning."
    )
