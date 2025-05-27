from weaviate.collections.classes.filters import Filter
from fastapi import FastAPI, Query, Request
import os
import weaviate
import openai
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

@app.get("/version")
def version_check():
    return {"status": "Running", "message": "✅ CORS enabled version"}

# ✅ Fix CORS for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://whealthchat.ai"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to Weaviate
client = weaviate.connect_to_wcs(
    cluster_url=os.getenv("WEAVIATE_CLUSTER_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
)

collection = client.collections.get("FAQ")

@app.post("/faq")
async def get_faq(request: Request):
    body = await request.json()
    q = body.get("query", "").strip()
    print(f"Received question: {q}")

    # ✅ Step 1: Exact match check
    filters = Filter.by_property("question").equal(q)
    print("🔍 Performing match lookup for:", q)

    exact_match = collection.query.fetch_objects(filters=filters, limit=10)

    matched_obj = None
    for obj in exact_match.objects:
        if obj.properties.get("question", "").strip() == q:
            matched_obj = obj
            break

    if matched_obj:
        print("✅ Exact match question from DB:", matched_obj.properties["question"])
        props = matched_obj.properties
        answer = props.get("answer", "").strip()
        coaching_tip = props.get("coachingTip", "").strip()

        prompt = (
            "You are a helpful assistant. Respond using Markdown with consistent formatting.\n"
            "Do NOT include the word 'Answer:' in your response.\n"
            "Bold the words 'Coaching Tip:' exactly as shown.\n"
            "Do not bold any other parts of the answer text.\n"
            "Keep 'Coaching Tip:' inline with the rest of the text, followed by a colon.\n"
            "Use line breaks only to separate paragraphs.\n\n"
            f"Question: {q}\n"
            f"{answer}\n"
            f"Coaching Tip: {coaching_tip}"
        )

        print("Exact match found. Prompt sent to OpenAI:", repr(prompt))

        try:
            reply = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.5
            )
            clean_response = reply.choices[0].message.content.strip()
            if clean_response.startswith('"') and clean_response.endswith('"'):
                clean_response = clean_response[1:-1]
            clean_response = clean_response.replace("\\n", "\n").strip()
            return clean_response

        except Exception as e:
            return f"An error occurred: {str(e)}"

    # ✅ Step 2: Vector fallback
    response = collection.query.near_text(
        query=q,
        limit=1,
        return_metadata=["distance"]
    )
    print(f"Weaviate response: {response}")

    if not response.objects:
        return "I do not possess the information to answer that question. Try asking me something about financial, retirement, estate, or healthcare planning."

    obj = response.objects[0]
    distance = obj.metadata.distance if obj.metadata and hasattr(obj.metadata, "distance") else 1.0

    if distance > 0.6:
        return "I do not possess the information to answer that question. Try asking me something about financial, retirement, estate, or healthcare planning."

    props = obj.properties
    answer = props.get("answer", "").strip()
    coaching_tip = props.get("coachingTip", "").strip()

    prompt = (
        "You are a helpful assistant. Respond using Markdown with consistent formatting.\n"
        "Do NOT include the word 'Answer:' in your response.\n"
        "Bold the words 'Coaching Tip:' exactly as shown.\n"
        "Do not bold any other parts of the answer text.\n"
        "Keep 'Coaching Tip:' inline with the rest of the text, followed by a colon.\n"
        "Use line breaks only to separate paragraphs.\n\n"
        f"Question: {q}\n"
        f"{answer}\n"
        f"Coaching Tip: {coaching_tip}"
    )

    print("Prompt sent to OpenAI:", repr(prompt))

    try:
        reply = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.5
        )
        clean_response = reply.choices[0].message.content.strip()
        if clean_response.startswith('"') and clean_response.endswith('"'):
            clean_response = clean_response[1:-1]
        clean_response = clean_response.replace("\\n", "\n").strip()
        print("Backend answer sent:", answer)
        return clean_response

    except Exception as e:
        return f"An error occurred: {str(e)}"

