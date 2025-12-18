import os
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()

endpoint = "https://opneaipocs.openai.azure.com/"
model_name = "text-embedding-3-large"
deployment = "text-embedding-3-large"

api_version = "2024-02-01"

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=endpoint,
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)


def extract_embeddings(texts, batch_size=100):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            input=batch,
            model=deployment
        )
        all_embeddings.extend(response.data)
    return all_embeddings

if __name__ == "__main__":
    texts = ["first phrase","second phrase","third phrase"]
    embeddings = extract_embeddings(texts)
    vectors = np.array([item.embedding for item in embeddings])
    print(vectors)
