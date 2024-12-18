import os
import asyncio
import json
from typing import List, Tuple
import gradio as gr
from aiohttp import ClientSession
import tiktoken
import numpy as np
from itertools import islice
import faiss  # For efficient similarity search

# Configuration from environment variables
LM_STUDIO_URL = os.getenv('LM_STUDIO_URL', 'http://localhost:1234')
CHAT_MODEL_ID = os.getenv('CHAT_MODEL_ID', 'Qwen2.5-Coder-32B-Instruct-IQ2_M.gguf')
EMBEDDING_MODEL_ID = os.getenv('EMBEDDING_MODEL_ID', 'nomic-embed-text-v1.5.Q8_0.gguf')

# Constants
MAX_TOKENS = 32768
EMBEDDING_CTX_LENGTH = 2048  # Context length for embeddings in LM Studio
EMBEDDING_ENCODING = 'cl100k_base'
EMBEDDING_DIMENSION = 768  # Correct dimension for nomic-embed-text-v1.5
K_RETRIEVAL = 3  # Number of similar items to retrieve from the knowledge base

# --- Knowledge Base and Embedding Index (using FAISS) ---

class KnowledgeBase:
    def __init__(self, dimension: int, filepath: str = None):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
        self.knowledge_dict = {}
        self.counter = 0
        if filepath:
            self.load_from_file(filepath)

    def add_item(self, text: str, embedding: List[float]):
        self.index.add(np.array([embedding], dtype=np.float32))
        self.knowledge_dict[self.counter] = text
        self.counter += 1

    def search(self, query_embedding: List[float], k: int) -> List[str]:
        distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), k)
        return [self.knowledge_dict[i] for i in indices[0]]

    def load_from_file(self, filepath: str):
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    content = json.loads(line.strip())
                    text = content['text']
                    embedding = content['embedding']
                    self.add_item(text, embedding)
        except FileNotFoundError:
             print("Knowledge base file not found. Please ensure 'knowledge_base.jsonl' exists in the same folder as the script")

# --- Embedding Functions ---

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

# --- Chat Completion Functions ---

async def generate_chat_completion(prompt: str, conversation_history: List[dict], max_tokens: int, session: ClientSession, retrieved_context: List[str]):
    # Convert conversation history to a string format, excluding empty messages
    formatted_history = "\n".join([f"{message['role']}: {message['content']}" for message in conversation_history if message.get('content')])

    # Construct the context prompt using retrieved information
    context_prompt = "Here is some relevant information from the knowledge base:\n"
    context_prompt += "\n".join(retrieved_context)
    context_prompt += "\n\nNow, continue the conversation with the user."

    payload = {
        "model": CHAT_MODEL_ID,
        "messages": [{"role": "system", "content": context_prompt}],
    }

    if formatted_history:
        payload["messages"].append({"role": "user", "content": formatted_history})

    payload["messages"].append({"role": "user", "content": prompt})

    payload["max_tokens"] = max_tokens
    payload["stream"] = True

    async with session.post(f"{LM_STUDIO_URL}/v1/chat/completions", json=payload) as response:
        if response.status == 200:
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith("data:"):
                    try:
                        data = json.loads(line[5:])
                        if 'choices' in data and data['choices']:
                            token = data['choices'][0]['delta'].get('content', '')
                            if token:
                                yield token
                    except json.JSONDecodeError:
                        print(f"Error decoding json {line}")
        else:
            response_text = await response.text()
            raise Exception(f"Error generating chat completion: {response_text}")

async def generate_intermediate_completion(prompt: str, conversation_history: List[dict], session: ClientSession):
    # Convert conversation history to a string format, excluding empty messages
    formatted_history = "\n".join([f"{message['role']}: {message['content']}" for message in conversation_history if message.get('content')])

    payload = {
        "model": CHAT_MODEL_ID,
        "messages": [{"role": "system", "content": "You are a strategic assistant tasked with planning the optimal response to user requests. Analyze the current conversation to identify the user's intent, and then formulate the necessary steps to fulfill that intent. Detail the reasoning behind these steps and extract key information to provide context for the final response. Your analysis should clearly outline the required actions and their justifications to ensure a comprehensive and well-reasoned final answer is created from this intermediate planning session."}],
    }

    if formatted_history:
        payload["messages"].append({"role": "user", "content": formatted_history})

    payload["messages"].append({"role": "user", "content": prompt})

    payload["max_tokens"] = 1000
    payload["stream"] = True  # Enable streaming for intermediate completion

    async with session.post(f"{LM_STUDIO_URL}/v1/chat/completions", json=payload) as response:
        if response.status == 200:
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith("data:"):
                   try:
                      data = json.loads(line[5:])
                      if 'choices' in data and data['choices']:
                          token = data['choices'][0]['delta'].get('content', '')
                          if token:
                            # Add "assistant:" prefix to intermediate responses
                            yield "assistant: " + token
                   except json.JSONDecodeError:
                       print(f"Error decoding json {line}")
        else:
            response_text = await response.text()
            raise Exception(f"Error generating intermediate completion: {response_text}")

# --- Main Chat Interface ---

async def chat_interface(prompt: str, conversation_history: List[dict], max_tokens_slider: int):
    try:
        async with ClientSession() as session:
            # Generate intermediate completion
            intermediate_response = ""
            async for token in generate_intermediate_completion(prompt, conversation_history, session):
                intermediate_response += token
                # Update conversation history with intermediate tokens
                yield conversation_history + [{"role": "user", "content": prompt}, {"role": "assistant", "content": intermediate_response}]

            # Generate embedding for intermediate response (without the "assistant:" prefix)
            intermediate_embedding = await generate_embeddings(intermediate_response.replace("assistant: ", ""), session)

            # Retrieve relevant information from knowledge base
            retrieved_context = knowledge_base.search(intermediate_embedding, K_RETRIEVAL)

            # Append intermediate response to conversation history
            conversation_history.append({"role": "assistant", "content": intermediate_response})

            # Generate final chat completion with retrieved context
            final_response = ""
            async for token in generate_chat_completion(prompt, conversation_history, max_tokens_slider, session, retrieved_context):
                final_response += token
                # Update conversation history with final response tokens
                yield conversation_history + [{"role": "user", "content": prompt}, {"role": "assistant", "content": final_response}]

    except Exception as e:
        yield conversation_history + [{"role": "assistant", "content": f"Error: {str(e)}"}]

async def chatbot_interface(prompt: str, conversation_history: List[dict], max_tokens_slider: int):
    async for history in chat_interface(prompt, conversation_history, max_tokens_slider):
        yield history

# --- Gradio Interface ---

# Initialize Knowledge Base (load from the generated file)
knowledge_base = KnowledgeBase(dimension=EMBEDDING_DIMENSION, filepath="knowledge_base.jsonl")

with gr.Blocks(title="LLM Chat Interface") as iface:
    gr.Markdown("# 🚀 High-Performance Chat Interface with Embeddings (RAG)")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot_history = gr.Chatbot(label="Conversation", height=400, type='messages')

            user_input = gr.Textbox(
                label="Your Message",
                placeholder="Type your message here...",
                lines=2
            )

            max_tokens_slider = gr.Slider(
                label="Max Tokens",
                minimum=100,
                maximum=MAX_TOKENS,
                step=100,
                value=1024
            )

            send_button = gr.Button("Send", variant="primary")

    with gr.Row():
        example_inputs = gr.Examples([
            ["What is the capital of France?"],
            ["Explain quantum mechanics in simple terms."],
            ["Tell me about Albert Einstein."]
        ], inputs=[user_input, max_tokens_slider])

    # Event handling
    send_button.click(chatbot_interface, [user_input, chatbot_history, max_tokens_slider], chatbot_history)

# Launch the interface
iface.launch()
