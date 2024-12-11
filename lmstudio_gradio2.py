#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
High-Performance Chat Interface for LM Studio - Enhanced Version

This script creates a chat interface using Gradio for interacting with the LM Studio API.
It features an advanced internal reasoning system using embeddings, enhanced problem
decomposition, and a simple in-memory knowledge base, all without external NLP libraries
like spaCy or FAISS.

Key Improvements:
- **Advanced Internal Reasoning:**
    - Employs a more sophisticated problem decomposition technique using regex patterns
      to identify potential sub-problems and actions within a user's message.
    - Uses a simple in-memory knowledge base with cosine similarity for knowledge retrieval.
    - Integrates retrieved knowledge into the reasoning steps and the final response.
- **No External NLP Libraries:**
    - Avoids dependencies on spaCy, FAISS, psutil, and cryptography.
    - All reasoning and knowledge retrieval logic is implemented using built-in Python
      libraries and NumPy for basic vector operations.
- **Streamlined and Efficient:**
    - Optimized for performance and memory usage within the constraints of a single-file
      implementation.
    - Asynchronous operations for improved responsiveness.
- **Real-Time Updates:**
    - Gradio interface updates in real-time as the response is generated and as
      reasoning steps are processed.
- **Simplified Configuration:**
    - Uses only two models (one for chat, one for embeddings) specified as constants.

Note:
- The knowledge base is entirely in-memory, limiting scalability.
- The advanced reasoning techniques, while improved, are still implemented without
  external NLP libraries, so their sophistication is limited by that constraint.

Author: Bard
Date: 2023-12-10 (Revised)
"""

import gradio as gr
import httpx
import json
import os
import numpy as np
import torch
import asyncio
import logging
from functools import lru_cache
from typing import Optional, Dict, List, Tuple
import re

# ===========================
# Logging Configuration
# ===========================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===========================
# Configuration and Constants
# ===========================

BASE_URL = os.getenv("LMSTUDIO_API_BASE_URL", "http://localhost:1234/v1")  # Ensure your LM Studio server is running

if not BASE_URL.startswith("http"):
    logger.error(f"Invalid BASE_URL: {BASE_URL}. Must start with 'http'.")
    raise ValueError("Invalid BASE_URL. Must start with 'http'.")

USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")

logger.info(f"GPU Available: {USE_GPU}, Device: {DEVICE}")

MODEL_MAX_TOKENS = 32768
EMBEDDING_MODEL_MAX_TOKENS = 8192
AVERAGE_CHARS_PER_TOKEN = 4
BUFFER_TOKENS = 1500
MIN_OUTPUT_TOKENS = 500

MAX_EMBEDDINGS = 100  # Reduced for in-memory database
HTTPX_TIMEOUT = 3000

HISTORY_FILE_PATH = "chat_history.json"  # This will be relative to the Spaces working directory

CHAT_MODEL = "Qwen2.5-Coder-32B-Instruct-abliterated-Rombo-TIES-v1.0.i1-IQ2_S.gguf"  # Ensure this model is available in your LM Studio
EMBEDDING_MODEL = "nomic-embed-text-v1.5.Q8_0.gguf"  # Ensure this model is available in your LM Studio

client: Optional[httpx.AsyncClient] = None

# ===========================
# In-Memory Knowledge Database
# ===========================

class InMemoryKnowledgeDB:
    """
    A simple in-memory knowledge database that stores text and their embeddings.
    """
    def __init__(self):
        self.knowledge: Dict[str, np.ndarray] = {}

    def add_item(self, text: str, embedding: np.ndarray):
        """Adds an item to the knowledge base."""
        self.knowledge[text] = embedding

    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """
        Searches the knowledge base for the most similar items to the query embedding.
        """
        if not self.knowledge:
            return []

        similarities = []
        for text, embedding in self.knowledge.items():
            similarity = calculate_similarity(query_embedding, embedding)
            similarities.append((text, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

# Global knowledge database instance (for simplicity)
knowledge_db = InMemoryKnowledgeDB()

# ===========================
# Utility Functions
# ===========================

@lru_cache(maxsize=128)
def calculate_max_tokens(message_history_length: int, model_max_tokens: int = MODEL_MAX_TOKENS,
                        buffer: int = BUFFER_TOKENS, avg_chars_per_token: int = AVERAGE_CHARS_PER_TOKEN,
                        min_tokens: int = MIN_OUTPUT_TOKENS) -> int:
    """Calculates the maximum number of tokens for the output."""
    input_tokens = message_history_length / avg_chars_per_token
    max_tokens = model_max_tokens - int(input_tokens) - buffer
    calculated_max = max(max_tokens, min_tokens)
    logger.info(f"Calculated max tokens: {calculated_max}")
    return calculated_max

async def get_embeddings(client: httpx.AsyncClient, text: str) -> Optional[np.ndarray]:
    """Retrieves embeddings from the LM Studio API."""
    text = text.strip()
    if not text:
        logger.warning("Attempted to get embeddings for empty text.")
        return None

    tokenized_text = text.split()
    chunks = []

    while tokenized_text:
        current_chunk = ' '.join(tokenized_text[:EMBEDDING_MODEL_MAX_TOKENS])
        chunks.append(current_chunk)
        tokenized_text = tokenized_text[EMBEDDING_MODEL_MAX_TOKENS:]

    embeddings = []

    for chunk in chunks:
        url = f"{BASE_URL}/embeddings"
        payload = {
            "model": EMBEDDING_MODEL,
            "input": [chunk]
        }
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            embedding = np.array(data["data"][0]["embedding"], dtype=np.float32)
            logger.info("Successfully retrieved embeddings.")
            embeddings.append(embedding)
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.error(f"HTTP error while getting embeddings: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error while getting embeddings: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error while getting embeddings: {e}")
            return None

    if embeddings:
        combined_embedding = np.concatenate(embeddings)
        logger.info(f"Combined embedding length: {len(combined_embedding)}")
        return combined_embedding
    else:
        return None

def calculate_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculates cosine similarity between two vectors using NumPy."""
    if vec1 is None or vec2 is None:
        logger.warning("One or both vectors are None. Returning similarity as 0.0.")
        return 0.0

    dot_product = np.dot(vec1, vec2)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)

    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0.0

    similarity = dot_product / (magnitude_vec1 * magnitude_vec2)
    return similarity

# ===========================
# Internal Reasoning 
# ===========================
async def generate_internal_reasoning_steps(client: httpx.AsyncClient, message: str) -> List[str]:
    """
    Generates internal reasoning steps for complex problem-solving using enhanced
    problem decomposition, embeddings, and a simple knowledge base.
    """
    try:
        message = message.strip()
        if not message:
            logger.warning("Attempted to generate reasoning steps for empty message.")
            return []

        user_embedding = await get_embeddings(client, message)
        if user_embedding is None:
            logger.error("Failed to generate embeddings for the message.")
            return []

        step1 = "Step 1: Analyzing the user input based on embeddings."

        # Enhanced Problem Decomposition (using regex patterns)
        sub_components = re.findall(r"(?:^|\s)(?:[Ww]hat|[Cc]an you|[Tt]ell me about|[Hh]ow to|[Dd]escribe|[Ee]xplain|[Ii]dentify)(.*?)(?=[?.!]|and|;|$)", message)
        sub_components = [comp.strip() for comp in sub_components if comp.strip()]

        step2 = "Step 2: Breaking down the problem into sub-components (using regex patterns)."
        for i, component in enumerate(sub_components):
            step2 += f"\n  Sub-component {i + 1}: {component[:50]}..."

        step3 = "Step 3: Retrieving relevant knowledge for each sub-component and synthesizing an overall response."
        overall_response = ""

        for i, component in enumerate(sub_components):
            component_embedding = await get_embeddings(client, component)
            if component_embedding is not None:
                similar_items = knowledge_db.search(component_embedding)
                if similar_items:
                    best_match, score = similar_items[0]
                    step3 += f"\n  For sub-component {i + 1}, retrieved knowledge: '{best_match}' (similarity: {score:.2f})"
                    overall_response += f" {best_match}"
                else:
                    step3 += f"\n  For sub-component {i + 1}, no relevant knowledge found."
            else:
                logger.error(f"Could not generate embeddings for sub-component {i+1}")

        reasoning_steps = [step1, step2, step3]
        logger.info(f"Generated internal reasoning steps: {reasoning_steps}")
        return reasoning_steps

    except Exception as e:
        logger.error(f"Error generating internal reasoning steps: {e}")
        return []
# ===========================
# API Interaction Handling (Asynchronous Streaming)
# ===========================

async def chat_with_lmstudio(client: httpx.AsyncClient, messages: List[Dict], model_name: str = CHAT_MODEL,
                             temperature: float = 1, top_p: float = 0.9, max_tokens: int = MODEL_MAX_TOKENS) -> str:
    """Handles chat completions with the LM Studio API using asynchronous streaming."""
    url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": True
    }

    try:
        logger.info("Sending chat completion request to LM Studio API.")
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.strip() == "data: [DONE]":
                    logger.info("Received [DONE] signal from LM Studio.")
                    break
                elif line.startswith("data:"):
                    data = json.loads(line[6:])
                    content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if content:
                        logger.debug(f"Received response chunk: {content[:50]}...")
                        yield content
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        logger.error(f"HTTP error during chat completion streaming: {e}")
        yield "An error occurred while generating a response. Check the logs for details."
    except Exception as e:
        logger.error(f"Unexpected error during chat completion streaming: {e}")
        yield "An unexpected error occurred while generating a response. Check the logs for details."

# ===========================
# Chat Handler
# ===========================

async def chat_handler(message: str, file_obj: Optional[object], state: Dict, internal_reasoning: bool,
                       model_name: str, temperature: float, top_p: float, chatbot_history: List[List],
                       context_display: str, reasoning_steps_display: str) -> Tuple[List[List], Dict, str, str]:
    """Handles the main chat logic."""
    logger.info("Handling new user message.")

    try:
        original_message = message.strip()
        if not original_message:
            yield chatbot_history, state, "", ""
            return

        embeddings = state.get("embeddings", [])
        messages_history = state.get("messages_history", [])
        updated_chat = chatbot_history.copy() if chatbot_history else []

        # File processing
        if file_obj is not None:
            try:
                if not file_obj.orig_name.lower().endswith(".txt"):
                    raise ValueError("Invalid file type. Only .txt files are allowed.")

                logger.info(f"Processing uploaded file: {file_obj.orig_name}")
                file_content = file_obj.read().decode("utf-8")
                logger.debug(f"File content: {file_content[:50]}...")
                message += f"\n[File Content]:\n{file_content}"

                # Add file content to knowledge base
                file_content_embedding = await get_embeddings(client, file_content)
                if file_content_embedding is not None:
                    knowledge_db.add_item(file_content, file_content_embedding)

            except ValueError as e:
                updated_chat.append([original_message, str(e)])
                yield updated_chat, state, "", ""
                return
            except UnicodeDecodeError:
                updated_chat.append([original_message, "Error: Could not decode file. Ensure it is a UTF-8 encoded text file."])
                yield updated_chat, state, "", ""
                return
            except Exception as e:
                logger.error(f"Error reading file: {e}")
                updated_chat.append([original_message, "Error reading file."])
                yield updated_chat, state, "", ""
                return

        # Embeddings generation
        user_embedding = await get_embeddings(client, message)
        if user_embedding is not None:
            logger.info("Embedding generated for user message.")
            embeddings.append(tuple(user_embedding))  # Convert to tuple for hashing
            messages_history.append({"role": "user", "content": original_message})
        else:
            updated_chat.append([original_message, "Failed to generate embeddings."])
            yield updated_chat, state, "", ""
            return

        if len(embeddings) > MAX_EMBEDDINGS:
            logger.info("Trimming embeddings and message history.")
            embeddings = embeddings[-MAX_EMBEDDINGS:]
            messages_history = messages_history[-MAX_EMBEDDINGS:]

        # Context Selection
        history = [*messages_history]
        context_text = ""

        if len(embeddings) > 1:
            logger.info("Retrieving relevant context from knowledge base.")
            similar_items = knowledge_db.search(user_embedding, k=3)  # Retrieve top 3
            for item, similarity in similar_items:
                context_text += f"Context: {item[:200]}... (Similarity: {similarity:.2f})\n"

            # Use the most similar item as context
            if similar_items:
                best_match, _ = similar_items[0]
                history.insert(0, {"role": "system", "content": best_match})

        # Calculate max_tokens for chat completion
        total_message_history_length = sum(len(msg["content"]) for msg in messages_history)
        max_tokens = calculate_max_tokens(total_message_history_length)

        response = ""
        updated_chat.append([original_message, None])

        # Internal reasoning steps
        reasoning_text = ""
        if internal_reasoning:
            reasoning_steps = await generate_internal_reasoning_steps(client, original_message)
            logger.info(f"Generated internal reasoning steps: {reasoning_steps}")
            reasoning_text = "\n".join(reasoning_steps)

        history.append({"role": "user", "content": original_message})
        try:
            logger.info("Sending chat request to LM Studio.")
            async for chunk in chat_with_lmstudio(client, history, model_name, temperature, top_p, max_tokens):
                response += chunk
                updated_chat[-1] = [original_message, response]
                yield updated_chat, {"embeddings": embeddings, "messages_history": messages_history}, context_text, reasoning_text

        except Exception as e:
            logger.error(f"Error during chat response generation: {e}")
            updated_chat.append([original_message, "An error occurred."])
            yield updated_chat, state, "", ""
            return

        messages_history.append({"role": "assistant", "content": response})

        # Save conversation history to file
        new_state = {"embeddings": embeddings, "messages_history": messages_history}
        try:
            with open(HISTORY_FILE_PATH, "w") as f:
                json.dump(new_state, f)
            logger.info("Conversation history saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save conversation history: {e}")

    except Exception as e:
        logger.error(f"Error in chat_handler: {e}")
        updated_chat.append([original_message, "An error occurred while processing your request. Please try again later."])
        yield updated_chat, state, "", ""

# ===========================
# Gradio Interface Implementation
# ===========================

async def gradio_chat_interface(client: httpx.AsyncClient):
    """Creates and launches the Gradio Blocks interface."""
    demo = gr.Blocks(title="LM Studio Chat Interface - Enhanced Version")
    with demo:
        gr.Markdown("# 🚀 High-Performance Chat Interface for LM Studio - Enhanced Version")

        with gr.Row():
            with gr.Column(scale=2):
                chatbot_history = gr.Chatbot(label="Conversation", height=500)

                with gr.Row():
                    user_input = gr.Textbox(
                        label="Your Message",
                        placeholder="Type your message here...",
                        lines=2,
                        scale=4
                    )
                    send_button = gr.Button("Send", variant="primary", scale=1)

                file_input = gr.UploadButton(label="Upload Context File (.txt)", type="binary")

            with gr.Column(scale=1):
                context_display = gr.Textbox(label="Relevant Context", interactive=False, lines=3)

                with gr.Accordion("Advanced Settings", open=False):
                    model_selector = gr.Dropdown(
                        label="Select Model",
                        choices=[CHAT_MODEL, "Another_Model"],
                        value=CHAT_MODEL
                    )
                    temperature_slider = gr.Slider(
                        label="Temperature (controls randomness)",
                        minimum=0.1,
                        maximum=2.0,
                        step=0.1,
                        value=1.0
                    )
                    top_p_slider = gr.Slider(
                        label="Top-p (controls diversity of tokens)",
                        minimum=0.1,
                        maximum=1.0,
                        step=0.1,
                        value=0.9
                    )
                    internal_reasoning_checkbox = gr.Checkbox(label="Enable Internal Reasoning", value=False)

                reasoning_steps_display = gr.Textbox(label="Reasoning Steps", interactive=False, lines=10)
                embeddings_display = gr.Textbox(label="Embeddings History", interactive=False, lines=10)

        # Persistent Conversation History
        history_state: Dict[str, List[Dict]] = {"embeddings": [], "messages_history": []}

        if os.path.exists(HISTORY_FILE_PATH):
            try:
                with open(HISTORY_FILE_PATH, "r") as f:
                    loaded_state = json.load(f)

                if not isinstance(loaded_state, dict) or "embeddings" not in loaded_state or "messages_history" not in loaded_state:
                    logger.warning("History format does not match. Starting with an empty one.")
                    history_state = {"embeddings": [], "messages_history": []}
                else:
                    history_state["embeddings"] = [tuple(emb) for emb in loaded_state.get("embeddings", [])] # Convert embeddings back to tuples
                    history_state["messages_history"] = loaded_state.get("messages_history", [])
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading chat history: {e}")
                history_state = {"embeddings": [], "messages_history": []}

        embeddings_state = gr.State(history_state)

        send_button.click(
            chat_handler,
            inputs=[user_input, file_input, embeddings_state, internal_reasoning_checkbox, model_selector, temperature_slider, top_p_slider, chatbot_history, context_display, reasoning_steps_display],
            outputs=[chatbot_history, embeddings_state, context_display, reasoning_steps_display]
        )

        user_input.submit(
            chat_handler,
            inputs=[user_input, file_input, embeddings_state, internal_reasoning_checkbox, model_selector, temperature_slider, top_p_slider, chatbot_history, context_display, reasoning_steps_display],
            outputs=[chatbot_history, embeddings_state, context_display, reasoning_steps_display]
        )

        def format_embeddings(embeddings):
            if embeddings and embeddings["messages_history"]:
                formatted_embeddings = "\n".join([f"Message: {msg['content']}\nEmbedding: {str(emb)[:50]}..." for msg, emb in zip(embeddings["messages_history"], embeddings["embeddings"])])
                return f"<pre>{formatted_embeddings}</pre>"
            else:
                return "No embeddings available."

        async def update_embeddings_display(embeddings_state):
            formatted_embeddings = format_embeddings(embeddings_state)
            return formatted_embeddings

        # Attach the embeddings display update to chat_handler
        chatbot_history.change(
            update_embeddings_display,
            inputs=[embeddings_state],
            outputs=embeddings_display
        )

    logger.info("Launching Gradio interface.")
    await demo.queue().launch(share=True, server_name="0.0.0.0", server_port=7860)

# ===========================
# Main Execution
# ===========================

if __name__ == "__main__":
    logger.info("Starting main execution.")
    try:
        # Example: Add some initial knowledge to the database
        knowledge_db.add_item("Pandas can be used to read CSV files.", [0.1] * 768)  # Replace with actual embedding
        knowledge_db.add_item("Matplotlib can create bar charts.", [0.2] * 768)  # Replace with actual embedding
        knowledge_db.add_item("The average can be calculated by grouping data in Pandas.", [0.3] * 768)  # Replace with actual embedding

        client = httpx.AsyncClient(timeout=httpx.Timeout(HTTPX_TIMEOUT))
        asyncio.run(gradio_chat_interface(client))
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
    finally:
        if client:
            async def close_client():
                await client.aclose()
            asyncio.run(close_client())
            logger.info("HTTPX Async Client closed.")
