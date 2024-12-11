import gradio as gr
import httpx
import json
import os
import numpy as np
import torch
import asyncio
import logging
from functools import lru_cache
from typing import Optional, Dict, List, Tuple, AsyncGenerator
import re

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
EMBEDDING_MODEL_MAX_TOKENS = 8192
AVERAGE_CHARS_PER_TOKEN = 4
BUFFER_TOKENS = 1500
MIN_OUTPUT_TOKENS = 8000  # Minimum output tokens

MAX_EMBEDDINGS = 100
HTTPX_TIMEOUT = 300  # Reduced timeout

HISTORY_FILE_PATH = "chat_history.json"

CHAT_MODEL = "bartowski/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-IQ2_M.gguf"  # Replace with your desired chat model
EMBEDDING_MODEL = "nomic-embed-text-v1.5.Q8_0.gguf"  # Replace with your desired embedding model

# ===========================
# In-Memory Knowledge Database
# ===========================

class InMemoryKnowledgeDB:
    def __init__(self):
        self.knowledge: List[Tuple[str, np.ndarray]] = []

    def add_item(self, text: str, embedding: np.ndarray):
        """Adds an item to the knowledge base."""
        if not isinstance(embedding, np.ndarray):
            logger.error("Invalid embedding type. Expected numpy array.")
            return
        self.knowledge.append((text, embedding))

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Searches for similar items in the knowledge database, with diversity enhancement.
        Returns top-k items with their similarity scores.
        """
        if not self.knowledge:
            return []

        similarities = []
        for text, embedding in self.knowledge:
            similarity = calculate_similarity(query_embedding, embedding)
            similarities.append((text, embedding, similarity))  # Include embedding for diversity check

        similarities.sort(key=lambda x: x[2], reverse=True)

        # Select top k items, ensuring diversity
        selected = []
        for text, embedding, score in similarities:
            is_diverse = True
            for selected_text, selected_embedding, _ in selected:
                # Check if the current item is too similar to already selected items
                if calculate_similarity(embedding, selected_embedding) > 0.9:  # Diversity threshold
                    is_diverse = False
                    break
            if is_diverse:
                selected.append((text, embedding, score))
            if len(selected) == k:
                break

        return [(text, score) for text, _, score in selected]

knowledge_db = InMemoryKnowledgeDB()

# ===========================
# Utility Functions
# ===========================

@lru_cache(maxsize=128)
def calculate_max_tokens(message_history_length: int, model_max_tokens: int = MODEL_MAX_TOKENS,
                        buffer: int = BUFFER_TOKENS, avg_chars_per_token: int = AVERAGE_CHARS_PER_TOKEN,
                        min_output_tokens: int = MIN_OUTPUT_TOKENS) -> int:
    """
    Calculates the maximum number of tokens for the output, prioritizing output length.

    This function now prioritizes having at least min_output_tokens available for the
    response. It dynamically adjusts the context window if necessary to accommodate this
    requirement.
    """
    # Estimate the number of tokens used by the input (message history)
    input_tokens = int(message_history_length / avg_chars_per_token)

    # Calculate the remaining tokens after accounting for the minimum output tokens and buffer
    remaining_tokens = model_max_tokens - min_output_tokens - buffer

    # Determine how many tokens to allocate to the context
    context_tokens = min(input_tokens, remaining_tokens)

    # If there's not enough room for the minimum output, log a warning.
    if context_tokens < 0:
        logger.warning(f"Insufficient tokens for minimum output. Output will be {min_output_tokens} tokens.")
        return min_output_tokens

    # Calculate the available space left for output
    available_output_tokens = model_max_tokens - context_tokens - buffer

    # Use the larger value between the minimum output tokens and the available output tokens
    max_tokens = max(available_output_tokens, min_output_tokens)

    logger.info(f"Calculated max tokens: {max_tokens}, Context tokens: {context_tokens}, Input tokens: {input_tokens}")
    return max_tokens

async def get_embeddings(text: str) -> Optional[np.ndarray]:
    """Retrieves embeddings, handling potential empty text and tokenization."""
    text = text.strip()
    if not text:
        logger.warning("Attempted to get embeddings for empty text.")
        return None

    tokenized_text = text.split()
    chunks = [' '.join(tokenized_text[i:i + EMBEDDING_MODEL_MAX_TOKENS]) for i in range(0, len(tokenized_text), EMBEDDING_MODEL_MAX_TOKENS)]

    embeddings = []
    async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
      for chunk in chunks:
          payload = {"model": EMBEDDING_MODEL, "input": [chunk]}
          try:
              response = await client.post(f"{BASE_URL}/embeddings", json=payload)
              response.raise_for_status()
              data = response.json()
              embedding = np.array(data["data"][0]["embedding"], dtype=np.float32)
              embeddings.append(embedding)
          except httpx.HTTPError as e:
              logger.error(f"HTTP error while getting embeddings: {e}")
              return None
          except json.JSONDecodeError as e:
              logger.error(f"JSON decoding error while getting embeddings: {e}")
              return None
    
    # Instead of combining, return the list of embeddings
    return embeddings if embeddings else None

def calculate_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculates cosine similarity between two vectors, handling None values."""
    if vec1 is None or vec2 is None:
        logger.warning("One or both vectors are None. Returning similarity as 0.0.")
        return 0.0

    # Ensure the vectors have the same dimensions
    if vec1.shape != vec2.shape:
        logger.error("Vectors have different dimensions. Returning similarity as 0.0.")
        return 0.0

    # Flatten the vectors to 1D arrays
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()

    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        logger.warning("Norm of one vector is zero. Returning similarity as 0.0.")
        return 0.0

    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity

# ===========================
# Internal Reasoning (Enhanced with more steps)
# ===========================

async def generate_internal_reasoning_steps(message: str) -> List[str]:
    """
    Generates internal reasoning steps for the user input by breaking it down into sub-components
    and retrieving relevant knowledge.
    """
    try:
        message = message.strip()
        if not message:
            logger.warning("Attempted to generate reasoning steps for empty message.")
            return []

        # Generate embeddings for the full message
        full_message_embedding = await get_embeddings(message)
        if full_message_embedding is None:
            logger.error("Failed to generate embeddings for the full message.")
            return []

        steps = [
            "Step 1: Analyzing the user input based on embeddings.",
            "Step 2: Breaking down the problem into sub-components (using regex patterns)."
        ]

        # Use regex to identify sub-components in the message
        sub_components = re.findall(r"(?:^|\s)(?:[Ww]hat|[Cc]an you|[Tt]ell me about|[Hh]ow to|[Dd]escribe|[Ee]xplain|[Ii]dentify)(.*?)(?=[?.!]|and|;|$)", message)
        sub_components = [comp.strip() for comp in sub_components if comp.strip()]

        for i, component in enumerate(sub_components):
            steps.append(f"  Sub-component {i + 1}: {component[:50]}...")

        steps.append("Step 3: Retrieving relevant knowledge for each sub-component.")
        for i, component in enumerate(sub_components):
            component_embeddings = await get_embeddings(component)
            
            if component_embeddings is not None:
              # Treat each embedding as a separate query to the knowledge base
              for embedding in component_embeddings:
                  similar_items = knowledge_db.search(embedding)
                  if similar_items:
                      best_match, score = similar_items[0]
                      steps.append(f"  For sub-component {i + 1}, retrieved knowledge: '{best_match[:50]}...' (similarity: {score:.2f})")
                  else:
                      steps.append(f"  For sub-component {i + 1}, no relevant knowledge found.")
            else:
              logger.error(f"Could not generate embeddings for sub-component {i + 1}")

        steps.append("Step 4: Synthesizing an overall response based on the retrieved knowledge and the original query.")
        
        logger.info(f"Generated internal reasoning steps: {steps}")
        return steps

    except Exception as e:
        logger.error(f"Error generating internal reasoning steps: {e}")
        return []

# ===========================
# API Interaction Handling (Asynchronous Streaming)
# ===========================

async def chat_with_lmstudio(messages: List[Dict], model_name: str = CHAT_MODEL,
                             temperature: float = 1, top_p: float = 0.9, max_tokens: int = MODEL_MAX_TOKENS) -> AsyncGenerator[str, None]:
    """Handles chat completions with the LM Studio API using asynchronous streaming."""
    url = f"{BASE_URL}/chat/completions"
    payload = {"model": model_name, "messages": messages, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, "stream": True}

    async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
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
      except httpx.ReadTimeout:
          logger.error("Timeout error during chat completion streaming.")
          yield "The operation timed out. Please try again."
      except httpx.HTTPError as e:
          logger.error(f"HTTP error during chat completion streaming: {e}")
          yield "An HTTP error occurred. Please check the server status and logs."
      except json.JSONDecodeError as e:
          logger.error(f"JSON decoding error during chat completion streaming: {e}")
          yield "An error occurred while decoding the server's response. Please check the server logs."
      except Exception as e:
          logger.error(f"Unexpected error during chat completion streaming: {e}")
          yield "An unexpected error occurred. Please try again later."

# ===========================
# Chat Handler
# ===========================

async def chat_handler(message: str, file_obj: Optional[object], state: Dict, internal_reasoning: bool,
                       model_name: str, temperature: float, top_p: float, chatbot_history: List[List],
                       context_display: str, reasoning_steps_display: str) -> Tuple[List[List], Dict, str, str]:
    logger.info("Handling new user message.")

    try:
        original_message = message.strip()
        if not original_message:
            yield chatbot_history, state, "", ""
            return

        embeddings = state.get("embeddings", [])
        messages_history = state.get("messages_history", [])
        updated_chat = chatbot_history.copy() if chatbot_history else []

        if file_obj is not None:
            try:
                if not file_obj.orig_name.lower().endswith(".txt"):
                    raise ValueError("Invalid file type. Only .txt files are allowed.")

                logger.info(f"Processing uploaded file: {file_obj.orig_name}")
                file_content = file_obj.read().decode("utf-8")
                message += f"\n[File Content]:\n{file_content}"

                file_content_embedding = await get_embeddings(file_content)
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

        # Generate embeddings for the user message
        user_embeddings = await get_embeddings(message)
        if user_embeddings is not None:
            logger.info("Embeddings generated for user message.")
            # Store each embedding individually
            for embedding in user_embeddings:
                embeddings.append(embedding.tolist())  # Convert to list for JSON serialization
                messages_history.append({"role": "user", "content": original_message})
        else:
            updated_chat.append([original_message, "Failed to generate embeddings."])
            yield updated_chat, state, "", ""
            return

        if len(embeddings) > MAX_EMBEDDINGS:
            logger.info("Trimming embeddings and message history.")
            embeddings = embeddings[-MAX_EMBEDDINGS:]
            messages_history = messages_history[-MAX_EMBEDDINGS:]

        history = [*messages_history]
        context_text = ""

        if len(embeddings) > 1:
            logger.info("Retrieving relevant context from knowledge base.")
            # Use the latest user embedding for context retrieval
            latest_user_embedding = user_embeddings[-1]  # Use the last embedding
            similar_items = knowledge_db.search(latest_user_embedding, k=3)
            for item, similarity in similar_items:
                context_text += f"Context: {item[:200]}... (Similarity: {similarity:.2f})\n"
            if similar_items:
                best_match, _ = similar_items[0]
                history.insert(0, {"role": "system", "content": best_match})

        total_message_history_length = sum(len(msg["content"]) for msg in messages_history)
        max_tokens = calculate_max_tokens(total_message_history_length)

        response = ""
        updated_chat.append([original_message, None])

        reasoning_text = ""
        if internal_reasoning:
            reasoning_steps = await generate_internal_reasoning_steps( original_message)
            reasoning_text = "\n".join(reasoning_steps)

        history.append({"role": "user", "content": original_message})
        try:
            logger.info("Sending chat request to LM Studio.")
            async for chunk in chat_with_lmstudio( history, model_name, temperature, top_p, max_tokens):
                response += chunk
                updated_chat[-1] = [original_message, response]
                yield updated_chat, {"embeddings": embeddings, "messages_history": messages_history}, context_text, reasoning_text

        except Exception as e:
            logger.error(f"Error during chat response generation: {e}")
            updated_chat.append([original_message, "An error occurred."])
            yield updated_chat, state, "", ""
            return

        messages_history.append({"role": "assistant", "content": response})
        new_state = {"embeddings": embeddings, "messages_history": messages_history}

        try:
            with open(HISTORY_FILE_PATH, "w") as f:
                json.dump(new_state, f)
            logger.info("Conversation history saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save conversation history: {e}")

        yield updated_chat, new_state, context_text, reasoning_text

    except Exception as e:
        logger.error(f"Error in chat_handler: {e}")
        updated_chat.append([original_message, "An error occurred while processing your request. Please try again later."])
        yield updated_chat, state, "", ""

# ===========================
# Gradio Interface Implementation
# ===========================

async def gradio_chat_interface():
    """Creates and launches the Gradio Blocks interface."""
    demo = gr.Blocks(title="LM Studio Chat Interface - Enhanced Version")
    async with httpx.AsyncClient(timeout=httpx.Timeout(HTTPX_TIMEOUT)) as client:
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

        history_state = load_history()
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
            """Formats the embeddings for display."""
            if embeddings and embeddings["messages_history"]:
                formatted_embeddings = "\n".join([f"Message: {msg['content']}\nEmbedding: {str(emb)[:50]}..." for msg, emb in zip(embeddings["messages_history"], embeddings["embeddings"])])
                return f"<pre>{formatted_embeddings}</pre>"
            else:
                return "No embeddings available."

        async def update_embeddings_display(embeddings_state):
            """Updates the embeddings display."""
            formatted_embeddings = format_embeddings(embeddings_state)
            return formatted_embeddings

        chatbot_history.change(
            update_embeddings_display,
            inputs=[embeddings_state],
            outputs=embeddings_display
        )

        logger.info("Launching Gradio interface.")
        await demo.queue().launch(share=True, server_name="0.0.0.0", server_port=7860)

def load_history() -> Dict:
    """Loads conversation history from a file or returns an empty history if the file does not exist or is invalid."""
    if not os.path.exists(HISTORY_FILE_PATH):
        logger.warning("History file does not exist. Starting with an empty history.")
        return {"embeddings": [], "messages_history": []}

    try:
        with open(HISTORY_FILE_PATH, "r") as f:
            history = json.load(f)
        
        if not isinstance(history, dict) or "embeddings" not in history or "messages_history" not in history:
            logger.warning("History file format is invalid. Starting with an empty history.")
            return {"embeddings": [], "messages_history": []}

        embeddings = history.get("embeddings", [])
        messages_history = history.get("messages_history", [])

        # Validate embeddings and messages history
        for embedding in embeddings:
            if not isinstance(embedding, list):
                logger.error("Invalid embedding format in history file.")
                return {"embeddings": [], "messages_history": []}

        for message in messages_history:
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                logger.error("Invalid message format in history file.")
                return {"embeddings": [], "messages_history": []}

        return history
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from history file: {e}")
        return {"embeddings": [], "messages_history": []}
    except Exception as e:
        logger.error(f"Error loading history: {e}")
        return {"embeddings": [], "messages_history": []}

# ===========================
# Main Execution
# ===========================

if __name__ == "__main__":
    logger.info("Starting main execution.")
    try:
        knowledge_db.add_item("Pandas can be used to read CSV files.", np.random.rand(768).astype(np.float32))
        knowledge_db.add_item("Matplotlib can create bar charts.", np.random.rand(768).astype(np.float32))
        knowledge_db.add_item("The average can be calculated by grouping data in Pandas.", np.random.rand(768).astype(np.float32))

        asyncio.run(gradio_chat_interface())
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
