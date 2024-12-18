import os
import asyncio
import json
from aiohttp import ClientSession
import tiktoken

# Configuration from environment variables
LM_STUDIO_URL = os.getenv('LM_STUDIO_URL', 'http://localhost:1234')
EMBEDDING_MODEL_ID = os.getenv('EMBEDDING_MODEL_ID', 'nomic-embed-text-v1.5.Q8_0.gguf')
EMBEDDING_CTX_LENGTH = 2048
EMBEDDING_ENCODING = 'cl100k_base'
EMBEDDING_DIMENSION = 768  # Correct dimension for nomic-embed-text-v1.5


def truncate_text_tokens(text, encoding_name=EMBEDDING_ENCODING, max_tokens=EMBEDDING_CTX_LENGTH):
    """Truncate a string to have `max_tokens` according to the given encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(text)[:max_tokens]

async def generate_embeddings(text: str, session: ClientSession):
    # Use LM Studio's embedding endpoint

    # Truncate the text if it's too long
    truncated_text_tokens = truncate_text_tokens(text, max_tokens=EMBEDDING_CTX_LENGTH)
    encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)
    truncated_text = encoding.decode(truncated_text_tokens)

    payload = {
        "model": EMBEDDING_MODEL_ID,
        "input": truncated_text,
        "embedding_type": "float"
    }

    async with session.post(f"{LM_STUDIO_URL}/v1/embeddings", json=payload) as response:
        if response.status == 200:
            response_json = await response.json()
            return response_json['data'][0]['embedding']
        else:
            response_text = await response.text()
            raise Exception(f"Error generating embeddings: {response_text}")

async def main():
    YOUR_KNOWLEDGE_BASE = [
        "Paris is the capital of France.",
        "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles.",
        "Albert Einstein was a German-born theoretical physicist, widely ranked among the greatest and most influential scientists of all time.",
         "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France." ,
         "The Mona Lisa is a half-length portrait painting by Italian artist Leonardo da Vinci.",
         "The Large Hadron Collider (LHC) is the world's largest and most powerful particle accelerator.",
         "E=mc² is a famous equation developed by Albert Einstein.",
         "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
         "The solar system consists of the Sun and the celestial objects that orbit it.",
         "The Earth's atmosphere is composed of nitrogen and oxygen."

    ]
    async with ClientSession() as session:
        with open("knowledge_base.jsonl", "w") as f:
            for item in YOUR_KNOWLEDGE_BASE:
                embedding = await generate_embeddings(item, session)
                json_line = json.dumps({"text": item, "embedding": embedding})
                f.write(json_line + "\n")
        print("knowledge_base.jsonl has been generated")

if __name__ == "__main__":
    asyncio.run(main())
