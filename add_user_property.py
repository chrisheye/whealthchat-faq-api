from weaviate import Client
from weaviate.auth import AuthApiKey
from weaviate.classes.config import Property, DataType

client = Client(
    url="https://7p26cwfhtawdfxv4j906a.c0.us-west3.gcp.weaviate.cloud",
    auth_client_secret=AuthApiKey("l08xptCQlzFutKWkusOTzvwPN2s4Scpbi7UJ")
)

# Define the new 'user' property
new_property = Property(
    name="user",
    data_type=DataType.TEXT
)

# Add it to the existing FAQ class
client.schema.property.create("FAQ", new_property)

print("✅ 'user' property added to FAQ class.")
