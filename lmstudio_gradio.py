#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
High-Performance Chat Interface for LM Studio - Enhanced Version

This script creates a robust and efficient chat interface using Gradio,
facilitating seamless interactions with the LM Studio API. It leverages
GPU capabilities for accelerated processing and utilizes asynchronous operations
for improved responsiveness. Token handling is dynamically adjusted for optimal
performance.

Author: Bard (Based on Your Original Script)
Date: 2023-11-22 (Updated 2024-12-08)
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
from typing import Optional, Dict, List
import re  # Added import for regex module

# ===========================
# Logging Configuration
# ===========================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===========================
# Configuration and Constants
# ===========================

BASE_URL = os.getenv("LMSTUDIO_API_BASE_URL", "http://localhost:1234/v1")

if not BASE_URL.startswith("http"):
    logger.error(f"Invalid BASE_URL: {BASE_URL}. Must start with 'http'.")
    raise ValueError("Invalid BASE_URL. Must start with 'http'.")

USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")

logger.info(f"GPU Available: {USE_GPU}, Device: {DEVICE}")

MODEL_MAX_TOKENS = 32768
EMBEDDING_MODEL_MAX_TOKENS = 8192  # Updated to 8192 for Nomic Embed
AVERAGE_CHARS_PER_TOKEN = 4
BUFFER_TOKENS = 1500
MIN_OUTPUT_TOKENS = 500

MAX_EMBEDDINGS = 100
HTTPX_TIMEOUT = 3000

HISTORY_FILE_PATH = "chat_history.json"

client: Optional[httpx.AsyncClient] = None

# ===========================
# Utility Functions
# ===========================

@lru_cache(maxsize=128)
def calculate_max_tokens(message_history_length, model_max_tokens=MODEL_MAX_TOKENS,
                        buffer=BUFFER_TOKENS, avg_chars_per_token=AVERAGE_CHARS_PER_TOKEN,
                        min_tokens=MIN_OUTPUT_TOKENS):
    """
    Calculates the maximum number of tokens for the output.

    Args:
        message_history_length (int): Total length of message history in characters.

    Returns:
        int: Maximum tokens for the output.
    """
    input_tokens = message_history_length / avg_chars_per_token
    max_tokens = model_max_tokens - int(input_tokens) - buffer
    calculated_max = max(max_tokens, min_tokens)
    logger.info(f"Calculated max tokens: {calculated_max}")
    return calculated_max

async def get_embeddings(client, text, embedding_model="nomic-embed-text-v1.5.f32.gguf"):
    """
    Retrieves embeddings from the LM Studio API.

    Args:
        client (httpx.AsyncClient): HTTPX asynchronous client instance.
        text (str): Input text for embedding generation.
        embedding_model (str): Model identifier for embeddings.

    Returns:
        list or None: Embedding vector or None if an error occurs.
    """
    text = text.strip()
    if not text:
        logger.warning("Attempted to get embeddings for empty text.")
        return None

    # Split the text into chunks of up to 8192 tokens
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
            "model": embedding_model,
            "input": [chunk]
        }
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            embedding = np.array(data["data"][0]["embedding"])
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

    # Combine embeddings from multiple chunks if necessary
    combined_embedding = np.concatenate(embeddings)
    
    logger.info(f"Combined embedding length: {len(combined_embedding)}")
    return combined_embedding.tolist()

def calculate_similarity(vec1, vec2):
    """
    Calculates cosine similarity between two vectors.

    Args:
        vec1 (list): First vector.
        vec2 (list): Second vector.

    Returns:
        float: Cosine similarity score.
    """
    if vec1 is None or vec2 is None:
        logger.warning("One or both vectors are None. Returning similarity as 0.0.")
        return 0.0

    vec1_tensor = torch.tensor(vec1, device=DEVICE)
    vec2_tensor = torch.tensor(vec2, device=DEVICE)

    similarity = torch.nn.functional.cosine_similarity(vec1_tensor.unsqueeze(0), vec2_tensor.unsqueeze(0)).item()
    logger.info(f"Calculated similarity: {similarity}")
    return similarity

# ===========================
# Internal Reasoning Mechanism (Enhanced)
# ===========================

async def generate_internal_reasoning_steps(client, message, reasoning_model="Qwen2.5-Coder-32B-Instruct-IQ2_M.gguf"):
    """
    Generates internal reasoning steps for complex problem-solving using embeddings.

    Args:
        client (httpx.AsyncClient): HTTPX asynchronous client instance.
        message (str): User's input message.
        reasoning_model (str): Model identifier for reasoning.

    Returns:
        list: List of reasoning steps.
    """
    try:
        # Strip whitespace from the message
        message = message.strip()
        if not message:
            logger.warning("Attempted to generate reasoning steps for empty message.")
            return []

        # Generate embeddings for the message
        user_embedding = await get_embeddings(client, message)
        if user_embedding is None:
            logger.error("Failed to generate embeddings for the message.")
            return []

        # Step 1: Analyze the user input using embeddings
        step1 = "Step 1: Analyzing the user input based on embeddings."

        # Break down the problem into sub-components
        sub_components = re.split(r'[?.!]', message)
        sub_components = [sc.strip() for sc in sub_components if sc.strip()]

        step2 = "Step 2: Breaking down the problem into sub-components."
        for i, component in enumerate(sub_components):
            step2 += f"\nSub-component {i+1}: {component[:50]}..."

        # Generate embeddings for each sub-component
        sub_component_embeddings = []
        for sc in sub_components:
            embedding = await get_embeddings(client, sc)
            if embedding is not None:
                sub_component_embeddings.append(embedding)

        # Step 3: Evaluate possible solutions for each sub-component using embeddings
        step3 = "Step 3: Evaluating possible solutions for each sub-component."
        for i, component in enumerate(sub_components):
            step3 += f"\nEvaluating solution for sub-component {i+1}: {component[:50]}..."

        # Step 4: Synthesize an overall response using embeddings
        step4 = "Step 4: Synthesizing an overall response."

        reasoning_steps = [
            step1,
            step2,
            step3,
            step4
        ]

        logger.info(f"Generated internal reasoning steps: {reasoning_steps}")
        return reasoning_steps

    except Exception as e:
        logger.error(f"Error generating internal reasoning steps: {e}")
        return []

# ===========================
# API Interaction Handling (Asynchronous Streaming)
# ===========================

async def chat_with_lmstudio(client, messages, model_name="Qwen2.5-Coder-32B-Instruct-IQ2_M.gguf", temperature=1, top_p=0.9, max_tokens=MODEL_MAX_TOKENS):
    """
    Handles chat completions with the LM Studio API using asynchronous streaming.

    Args:
        client (httpx.AsyncClient): HTTPX asynchronous client instance.
        messages (list): List of message dictionaries for the LM Studio API.
        model_name (str): Name of the model to use.
        temperature (float): Temperature parameter for randomness in generation.
        top_p (float): Top-p sampling parameter.
        max_tokens (int): Maximum number of tokens to generate.

    Yields:
        str: Chunks of generated response.
    """
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
# Plugin/Extension System
# ===========================

class PluginManager:
    def __init__(self):
        self.plugins = {}

    def register_plugin(self, name, func):
        self.plugins[name] = func

    async def run_plugins(self, message, context_display):
        results = []
        plugin_tasks = [plugin(message, context_display) for plugin in self.plugins.values()]
        try:
            plugin_results = await asyncio.gather(*plugin_tasks)
        except Exception as e:
            logger.error(f"Error running plugins: {e}")
            return [(name, "Plugin failed") for name in self.plugins.keys()]

        for name, result in zip(self.plugins.keys(), plugin_results):
            results.append((name, result))
        return results

plugin_manager = PluginManager()

# Example plugins
async def database_query_plugin(message, context_display):
    # Placeholder for actual database query functionality
    try:
        import sqlite3

        conn = sqlite3.connect('example.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM example_table WHERE message=?", (message,))
        result = cursor.fetchone()
        conn.close()

        return f"Database Query: {result}"
    except Exception as e:
        logger.error(f"Error during database query: {e}")
        return "Database Query: An error occurred. Ensure the database is accessible."

plugin_manager.register_plugin("Database Query", database_query_plugin)

# ===========================
# Handlers for Chat
# ===========================

async def chat_handler(message, file_obj, state, internal_reasoning, model_name, temperature, top_p, chatbot_history, context_display):
    logger.info("Handling new user message.")

    try:
        # Strip whitespace from the message
        original_message = message.strip()
        if not original_message:
            yield chatbot_history, state, ""
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
            except ValueError as e:
                updated_chat.append([original_message, str(e)])
                yield updated_chat, state, ""
                return
            except UnicodeDecodeError:
                updated_chat.append([original_message, "Error: Could not decode file. Ensure it is a UTF-8 encoded text file."])
                yield updated_chat, state, ""
                return
            except Exception as e:
                logger.error(f"Error reading file: {e}")
                updated_chat.append([original_message, "Error reading file."])
                yield updated_chat, state, ""
                return

        # Embeddings generation
        user_embedding = await get_embeddings(client, message)
        if user_embedding is not None:
            logger.info("Embedding generated for user message.")
            embeddings.append(user_embedding)
            messages_history.append({"role": "user", "content": original_message})
        else:
            updated_chat.append([original_message, "Failed to generate embeddings."])
            yield updated_chat, state, ""
            return

        if len(embeddings) > MAX_EMBEDDINGS:
            logger.info("Trimming embeddings and message history.")
            embeddings = embeddings[-MAX_EMBEDDINGS:]
            messages_history = messages_history[-MAX_EMBEDDINGS:]

        # Improved Context Selection
        max_context_length = min(10, max(3, MODEL_MAX_TOKENS // (AVERAGE_CHARS_PER_TOKEN * 1500)))
        if len(messages_history) > max_context_length:
            logger.info("Trimming message history for context length.")
            messages_history = messages_history[-max_context_length:]
            embeddings = embeddings[-MAX_EMBEDDINGS:]  # Ensure embeddings are trimmed accordingly

        history = [*messages_history]  # Removed appending current message here
        context_text = ""

        if len(embeddings) > 1:
            logger.info("Calculating similarity scores.")
            similarities = sorted(
                [(calculate_similarity(user_embedding, emb), idx) for idx, emb in enumerate(embeddings[:-1])],
                key=lambda x: x[0],
                reverse=True
            )
            top_context = similarities[:3]

            for _, idx in top_context:
                if idx < len(messages_history):
                    context_message = messages_history[idx]
                    if context_message["role"] == "user":
                        history.insert(0, {"role": "system", "content": context_message["content"]})
                        context_text += f"Context: {context_message['content'][:100]}...\n"
            logger.debug(f"Retrieved context: {context_text}")

        total_message_history_length = sum(len(msg["content"]) for msg in messages_history)
        max_tokens = calculate_max_tokens(total_message_history_length)

        response = ""
        updated_chat.append([original_message, None])  # Added current message to chat history

        # Internal reasoning steps
        if internal_reasoning:
            reasoning_steps = await generate_internal_reasoning_steps(client, original_message)
            logger.info(f"Generated internal reasoning steps: {reasoning_steps}")

            # Add reasoning steps to the chat history for display
            updated_chat.extend([[None, step] for step in reasoning_steps])

        # Run plugins
        plugin_results = await plugin_manager.run_plugins(original_message, context_display)
        plugin_output = "\n".join(f"{name}: {result}" for name, result in plugin_results)
        if plugin_output:
            updated_chat.append([None, plugin_output])

        history.append({"role": "user", "content": original_message})  # Appended current message to LM Studio history
        try:
            logger.info("Sending chat request to LM Studio.")
            async for chunk in chat_with_lmstudio(client, history, model_name, temperature, top_p, max_tokens):
                response += chunk
                updated_chat[-1] = [original_message, response]  # Update last message with response
                yield updated_chat, {"embeddings": embeddings, "messages_history": messages_history}, context_text
        except Exception as e:
            logger.error(f"Error during chat response generation: {e}")
            updated_chat.append([original_message, "An error occurred."])  # Update last message with error
            yield updated_chat, state, ""
            return

        messages_history.append({"role": "assistant", "content": response})  # Append assistant response to message history

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
        yield updated_chat, state, ""

# ===========================
# Gradio Interface Implementation
# ===========================

async def gradio_chat_interface(client):
    """
    Creates and launches the Gradio Blocks interface.

    Args:
        client (httpx.AsyncClient): HTTPX asynchronous client instance.
    """
    demo = gr.Blocks(title="LM Studio Chat Interface - Enhanced Version")
    with demo:
        gr.Markdown("# 🚀 High-Performance Chat Interface for LM Studio - Enhanced Version")

        chatbot_history = gr.Chatbot(label="Conversation")  # Removed type="messages"

        with gr.Row():
            user_input = gr.Textbox(
                label="Your Message",
                placeholder="Type your message here...",
                lines=2,
                scale=4
            )
            send_button = gr.Button("Send", variant="primary", scale=1)

        file_input = gr.UploadButton(label="Upload Context File (.txt)", type="binary")
        context_display = gr.Textbox(label="Relevant Context", interactive=False)
        internal_reasoning_checkbox = gr.Checkbox(label="Enable Internal Reasoning", value=False)

        # Multi-Model Support
        model_selector = gr.Dropdown(
            label="Select Model",
            choices=["Qwen2.5-Coder-32B-Instruct-IQ2_M.gguf", "Another_Model"],
            value="Qwen2.5-Coder-32B-Instruct-IQ2_M.gguf"
        )

        # Advanced Parameter Tuning
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

        # Persistent Conversation History
        history_state: Dict[str, List[Dict]] = {"embeddings": [], "messages_history": []}

        if os.path.exists(HISTORY_FILE_PATH):
            try:
                with open(HISTORY_FILE_PATH, "r") as f:
                    loaded_state = json.load(f)

                # Validate the structure of the loaded state
                if not isinstance(loaded_state, dict) or \
                   "embeddings" not in loaded_state or \
                   "messages_history" not in loaded_state:
                    logger.warning("History format does not match. Starting with an empty one.")
                    history_state = {"embeddings": [], "messages_history": []}
                else:
                    history_state["embeddings"] = loaded_state.get("embeddings", [])
                    history_state["messages_history"] = loaded_state.get("messages_history", [])
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading chat history: {e}")
                history_state = {"embeddings": [], "messages_history": []}

        embeddings_state = gr.State(history_state)

        send_button.click(
            chat_handler,
            inputs=[user_input, file_input, embeddings_state, internal_reasoning_checkbox, model_selector, temperature_slider, top_p_slider, chatbot_history],
            outputs=[chatbot_history, embeddings_state, context_display]
        )

        user_input.submit(
            chat_handler,
            inputs=[user_input, file_input, embeddings_state, internal_reasoning_checkbox, model_selector, temperature_slider, top_p_slider, chatbot_history],
            outputs=[chatbot_history, embeddings_state, context_display]
        )

    logger.info("Launching Gradio interface.")
    await demo.queue().launch(share=True, server_name="0.0.0.0", server_port=7860)

# ===========================
# Main Execution
# ===========================

if __name__ == "__main__":
    logger.info("Starting main execution.")
    try:
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
