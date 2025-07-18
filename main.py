from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import weaviate
from weaviate.classes.query import Filter
from weaviate.classes.config import Configure
from openai import OpenAI
import os

app = FastAPI()

# Allow CORS from any origin (adjust for production if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Weaviate v4 client
client = weaviate.WeaviateClient(
    url=os.getenv("WEAVIATE_URL"),
    auth_client_secret=weaviate.auth.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY")),
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
)

collection = client.collections.get("whealthchat-faqs")

# Initialize OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    "Show empathy through wording — not by pretending to be human, but by offering clarity, calm, and actionable suggestions."
)

@app.post("/faq")
async def get_faq(request: Request):
    data = await request.json()
    q = data.get("query", "").strip()
    user_type = data.get("user", "consumer")

    if not q:
        return "Please enter a valid question."

    normalized_q = q.lower().strip("?").strip()
    print(f"👤 User type: {user_type}")
    print(f"Received question: {q}")
    print(f"🔎 Checking exact match for normalized question: {normalized_q}")

    try:
        exact_match = collection.query.fetch_objects(
            filters=Filter.by_property("question").equal(normalized_q)
            .and_filter(Filter.by_property("user").any_of([user_type, "both"]))
        )
        if exact_match.objects:
            print("✅ Exact match found")
            obj = exact_match.objects[0].properties
        else:
            print("⚠️ No exact match. Trying vector search...")
            vector_match = collection.query.near_text(
                query=Configure.near_text(concepts=[q]),
                filters=Filter.by_property("user").any_of([user_type, "both"]),
                limit=1,
            )
            if vector_match.objects:
                print("✅ Vector match found")
                obj = vector_match.objects[0].properties
            else:
                print("❌ No matches found at all")
                return "I do not possess the information to answer that question. Try asking me something about financial, retirement, estate, or healthcare planning."
    except Exception as e:
        print("❌ Error querying Weaviate:", str(e))
        return "⚠️ There was a problem processing your question."

    question = obj.get("question", "")
    answer = obj.get("answer", "")
    tip = obj.get("tip", "")

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Q: {question}\nA: {answer}\n\nCoaching Tip: {tip}"},
        ]
        chat_response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.4,
        )
        return chat_response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ OpenAI error:", str(e))
        return f"⚠️ Unable to generate a response right now. Please try again later."
