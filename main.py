from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from weaviate import WeaviateClient
from weaviate.auth import AuthApiKey

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = WeaviateClient(
    url=os.environ.get("WEAVIATE_CLUSTER_URL", "https://7p26cwfhtawdfxv4j906a.c0.us-west3.gcp.weaviate.cloud"),
    auth_credentials=AuthApiKey(os.environ.ge_
