from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/faq")
async def get_faq(request: Request):
    body = await request.json()
    question = body.get("query", "").strip()
    print(f"✅ Received question: {question}")
    return {"status": "OK", "received": question}
