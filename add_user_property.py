from weaviate import WeaviateClient
from weaviate.auth import AuthApiKey
from weaviate.classes.config import Property, DataType

client = WeaviateClient(
    url="https://7p26cwfhtawdfxv4j906a.c0.us-west3.gcp.weaviate.cloud",
    auth_credentials=AuthApiKey("l08xptCQlzFutKWkusOTzvwPN2s4Scpbi7UJ")
)

# Define the new 'user' property
new_property = Property(
    name="user",
    data_type=DataType.TEXT
)

# Add it to the existing FAQ class
client.schema.properties.add("FAQ", new_property)

print("✅ 'user' property added to FAQ class.")
