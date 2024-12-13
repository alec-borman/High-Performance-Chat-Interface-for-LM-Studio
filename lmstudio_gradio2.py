import os
import json
import asyncio
import logging
import httpx
import numpy as np
import torch
from functools import lru_cache
from typing import Optional, Dict, List, Tuple, AsyncGenerator, Protocol, Any
from transformers import AutoTokenizer, AutoModel
import gradio as gr
import time
import secrets
import hashlib
import torch.nn as nn
from dotenv import load_dotenv
from aiolimiter import AsyncLimiter
import re
from neo4j import GraphDatabase
from chromadb.utils import embedding_functions
import html


load_dotenv()

# ===========================
# Logging Configuration
# ===========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger("HighPerfChatbot")

# ===========================
# Configuration and Constants
# ===========================
class Config:
    """Handles loading and validating configuration parameters."""
    def __init__(self):
        self.BASE_URL = os.getenv("LMSTUDIO_API_BASE_URL", "http://localhost:1234/v1")
        self.CHAT_MODEL = os.getenv("CHAT_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text-v1.5.Q8_0.gguf")
        self.MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", 32768))
        self.EMBEDDING_MODEL_MAX_TOKENS = int(os.getenv("EMBEDDING_MODEL_MAX_TOKENS", 8192))
        self.BUFFER_TOKENS = int(os.getenv("BUFFER_TOKENS", 1500))
        self.MIN_OUTPUT_TOKENS = int(os.getenv("MIN_OUTPUT_TOKENS", 8000))
        self.MAX_EMBEDDINGS = int(os.getenv("MAX_EMBEDDINGS", 100))
        self.HTTPX_TIMEOUT = int(os.getenv("HTTPX_TIMEOUT", 300))
        self.HISTORY_FILE_PATH = os.getenv("HISTORY_FILE_PATH", "chat_history.json")
        self.DB_PATH = os.getenv("DB_PATH", "chroma_db")
        self.EMBEDDING_MODEL_HUGGING_FACE_NAME = os.getenv("EMBEDDING_MODEL_HUGGING_FACE_NAME", "nomic-ai/nomic-embed-text-v1.5")
        self.BIAS_DETECTOR_MODEL = os.getenv("BIAS_DETECTOR_MODEL", "unitary/unbiased-toxic-roberta")
        self.API_KEY = os.getenv("API_KEY", self._generate_api_key())
        self.CONVERSATION_CONTEXT_DEPTH = int(os.getenv("CONVERSATION_CONTEXT_DEPTH", 3))
        self.VECTOR_DATABASE_BATCH_SIZE = int(os.getenv("VECTOR_DATABASE_BATCH_SIZE", 5))
        self.QWEN_TOKENIZER_NAME = os.getenv("QWEN_TOKENIZER_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct")
        self.NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
        self.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
        self.EMBEDDING_DIMS = int(os.getenv("EMBEDDING_DIMS", 768))
        self.NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 3))
        self.LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.0001))

        if not self.BASE_URL.startswith("http"):
            raise ValueError(f"Invalid BASE_URL: {self.BASE_URL}. Must start with 'http'.")

        self.USE_GPU = torch.cuda.is_available()
        self.DEVICE = torch.device("cuda" if self.USE_GPU else "cpu")
        logger.info(f"GPU Available: {self.USE_GPU}, Device: {self.DEVICE}")

        # Setup a simple async rate limiter: 10 requests/minute
        self.rate_limiter = AsyncLimiter(10, 60)

    def _generate_api_key(self):
        """Generate a random API key if not provided in environment"""
        return hashlib.sha256(secrets.token_bytes(32)).hexdigest()

config = Config()

# Create a global rate limiter using aiolimiter
# 10 requests per 60 seconds as an example
limiter = AsyncLimiter(10, 60)

# ===========================
# Interface Protocols
# ===========================
class VectorStore(Protocol):
    def add_item(self, text: str, metadata=None) -> None:
        ...
    def search(self, query_text: str, k: int = 5, diversity_factor: float = 0.5) -> List[Tuple[str, float]]:
        ...

class EmbeddingModel(Protocol):
    async def get_embeddings(self, text: str) -> Optional[np.ndarray]:
        ...

class BiasDetector(Protocol):
    def detect_bias(self, text: str) -> bool:
        ...

# ===========================
# In-Memory GPU Vector Database
# ===========================
class InMemoryGPUVectorDB:
    """
    Vector database using PyTorch tensors on GPU.
    """
    def __init__(self, config: Config):
        self.config = config
        self.embeddings = []
        self.texts = []

    def add_item(self, text: str, metadata=None):
        """Adds a single item to the index"""
        embedding = asyncio.run(embedding_model.get_embeddings(text))
        if embedding is not None:
            self.embeddings.append(torch.tensor(embedding, dtype=torch.float32, device=self.config.DEVICE))
            self.texts.append(text)
            logger.info(f"Added document to in-memory vector DB: {text[:50]}...")

    def search(self, query_text: str, k: int = 5, diversity_factor: float = 0.5) -> List[Tuple[str, float]]:
        """Searches the tensor index for similar items and uses MMR."""
        query_embedding = asyncio.run(embedding_model.get_embeddings(query_text))
        if query_embedding is None:
            logger.warning("Failed to generate embedding for query text.")
            return []
        query_embedding = torch.tensor(query_embedding, dtype=torch.float32, device=self.config.DEVICE)

        if not self.embeddings:
            logger.info("No items in vector DB.")
            return []

        similarities = torch.stack([self._calculate_similarity(query_embedding, emb) for emb in self.embeddings])
        # Sort by highest scores
        scores, indices = torch.sort(similarities, descending=True)
        results = [(self.texts[i], float(scores[idx])) for idx, i in enumerate(indices[:10*k])]
        return self._mmr(query_text, results, k, diversity_factor)

    def _mmr(self, query: str, results: List[Tuple[str, float]], k: int, diversity_factor: float) -> List[Tuple[str, float]]:
        """Maximum Marginal Relevance ranking for more diverse results."""
        if not results:
            return []

        selected_indices = []
        remaining_indices = list(range(len(results)))

        for _ in range(k):
            if not remaining_indices:
                break

            best_index = -1
            best_score = float('-inf')

            for i in remaining_indices:
                text, similarity = results[i]
                mmr_score = (1 - diversity_factor) * similarity
                if selected_indices:
                    max_sim_selected = max([self._calculate_similarity(text, results[j][0]).cpu().item() for j in selected_indices])
                    mmr_score -= diversity_factor * max_sim_selected
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_index = i

            if best_index != -1:
                selected_indices.append(best_index)
                remaining_indices.remove(best_index)

        return [results[index] for index in selected_indices]

    def _calculate_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
        """Calculate cosine similarity between two embeddings."""
        dot_product = torch.dot(embedding1, embedding2)
        norm1 = torch.norm(embedding1)
        norm2 = torch.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return torch.tensor(0.0, device=self.config.DEVICE)
        return dot_product / (norm1 * norm2)

    def add_items_batched(self, texts: List[str], metadatas: Optional[List[Any]] = None) -> None:
        """Adds items in batched mode"""
        for text in texts:
            self.add_item(text)

vector_db = InMemoryGPUVectorDB(config)

# ===========================
# Tokenizer Setup
# ===========================
class Tokenizer:
    """Handles token counting using a Hugging Face tokenizer."""
    def __init__(self, config: Config):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.QWEN_TOKENIZER_NAME)
            logger.info("Tokenizer initialized.")
        except Exception as e:
            logger.error(f"Error initializing tokenizer: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """Returns the number of tokens in the given text."""
        if not text:
            return 0
        tokens = self.tokenizer(text)
        return len(tokens['input_ids'])

tokenizer = Tokenizer(config)

# ===========================
# Utility Functions
# ===========================
def calculate_max_tokens(message_history_length: int, config: Config) -> int:
    """Calculate maximum tokens for output given constraints."""
    input_tokens = message_history_length
    remaining_tokens = config.MODEL_MAX_TOKENS - config.MIN_OUTPUT_TOKENS - config.BUFFER_TOKENS
    context_tokens = min(input_tokens, remaining_tokens)
    if context_tokens < 0:
        logger.warning(f"Not enough tokens for min output. Using {config.MIN_OUTPUT_TOKENS}.")
        return config.MIN_OUTPUT_TOKENS

    available_output_tokens = config.MODEL_MAX_TOKENS - context_tokens - config.BUFFER_TOKENS
    max_tokens = max(available_output_tokens, config.MIN_OUTPUT_TOKENS)

    logger.info(f"Max tokens: {max_tokens}, Context tokens: {context_tokens}, Input tokens: {input_tokens}")
    return max_tokens


def sanitize_input(text: str) -> str:
    """
    Sanitizes user input to reduce the risk of prompt injection and malicious content.
    Strips HTML tags, removes backticks and special characters.
    """
    # Decode any HTML entities & remove HTML tags
    clean = html.unescape(text)
    clean = re.sub(r'<[^>]+>', '', clean)
    clean = clean.replace('`', '')
    # Optional: Further sanitation steps can be added here
    return clean.strip()

# ===========================
# HTTP Embedding Model
# ===========================
class HttpEmbeddingModel:
    """Embedding model that retrieves embeddings from a remote server (LM Studio API)."""
    def __init__(self, config: Config):
        self.config = config
        self.model = None  # Placeholder for future fine-tuning

    @lru_cache(maxsize=256)
    async def get_embeddings(self, text: str) -> Optional[np.ndarray]:
        """Get embeddings for given text from the external embedding service."""
        if not text.strip():
            logger.warning("Attempted to get embeddings for empty text.")
            return None

        async with httpx.AsyncClient(timeout=self.config.HTTPX_TIMEOUT) as client:
            payload = {"model": self.config.EMBEDDING_MODEL, "input": [text]}
            try:
                async with limiter:
                   response = await client.post(f"{self.config.BASE_URL}/embeddings", json=payload)
                response.raise_for_status()
                data = response.json()
                embedding = np.array(data["data"][0]["embedding"], dtype=np.float32)
                return embedding
            except httpx.HTTPError as e:
                logger.error(f"HTTP error getting embeddings: {e}")
                return None
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                return None

    async def fine_tune(self):
        """A dummy placeholder for future fine-tuning logic."""
        pass

embedding_model = HttpEmbeddingModel(config)

# ===========================
# Bias Detection
# ===========================
class HFTransformersBiasDetector:
    """Detects bias using a HuggingFace Transformer model."""
    def __init__(self, config: Config):
        self.config = config
        try:
            self.model = AutoModel.from_pretrained(self.config.BIAS_DETECTOR_MODEL).to(self.config.DEVICE)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.BIAS_DETECTOR_MODEL)
            logger.info("Bias detector initialized.")
        except Exception as e:
            logger.error(f"Error initializing bias detector: {e}")
            raise

    def detect_bias(self, text: str) -> bool:
        """Detects bias in the given text."""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.config.DEVICE)
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Dummy logic: average sigmoid output
            scores = torch.sigmoid(outputs.last_hidden_state).cpu().numpy().flatten()
            bias_score = np.mean(scores)

            logger.debug(f"Bias score detected: {bias_score}")
            return bias_score > 0.65
        except Exception as e:
            logger.error(f"Error detecting bias: {e}")
            return False

bias_detector = HFTransformersBiasDetector(config)

# ===========================
# Knowledge Graph Context Retrieval
# ===========================
class KnowledgeGraphRetriever:
    def __init__(self, config: Config):
        self.config = config
        try:
            self.driver = GraphDatabase.driver(
                self.config.NEO4J_URI,
                auth=(self.config.NEO4J_USER, self.config.NEO4J_PASSWORD)
            )
            logger.info("Knowledge graph (Neo4j) initialized.")
        except Exception as e:
            logger.error(f"Error initializing Neo4j driver: {e}")
            raise

    def get_context(self, query_text: str, k: int = 5) -> str:
        """Retrieves context using graph-based approach."""
        try:
            loop = asyncio.get_event_loop()
            query_embedding = loop.run_until_complete(embedding_model.get_embeddings(query_text))
            if query_embedding is None:
                logger.warning("Could not get embedding of query text.")
                return ""

            with self.driver.session() as session:
                query = """
                    MATCH (n:Document)
                    WITH n, gds.similarity.cosine(n.embedding, $query_embedding) AS similarity
                    WHERE similarity > 0.5
                    ORDER BY similarity DESC
                    LIMIT $k
                    RETURN n.text AS text, similarity
                """
                results = session.run(query, query_embedding=query_embedding.tolist(), k=k)
                records = [record for record in results]

                context_text = ""
                for record in records:
                    text = record["text"]
                    similarity = record["similarity"]
                    context_text += f"Context: {text[:200]}... (Similarity: {similarity:.2f})\n"

                logger.info(f"Retrieved context from KG: {context_text[:200]}...")
                return context_text
        except Exception as e:
            logger.error(f"Error accessing Neo4j graph database: {e}")
            return ""

    def add_item(self, text: str, metadata=None):
        """Adds item to the knowledge graph. Embeddings must be added as well."""
        embedding = asyncio.run(embedding_model.get_embeddings(text))
        if embedding is None:
            logger.warning("Could not get embedding to add item to the graph.")
            return

        try:
            with self.driver.session() as session:
                query = """
                    CREATE (n:Document {text: $text, embedding: $embedding})
                """
                session.run(query, text=text, embedding=embedding.tolist())
                logger.info(f"Added document to knowledge graph: {text[:50]}...")
        except Exception as e:
            logger.error(f"Error adding item to Neo4j graph: {e}")

    def add_items_batched(self, texts: List[str], metadatas: Optional[List[Any]] = None) -> None:
        """Add items in batches."""
        if not texts:
            logger.warning("Attempted to add an empty list of texts to the graph.")
            return
        for text in texts:
            self.add_item(text)

graph_context_retriever = KnowledgeGraphRetriever(config)

# ===========================
# Chain-of-Thought Reasoning
# ===========================
class ChainOfThoughtReasoning:
    """Implements chain-of-thought prompting for enhanced reasoning."""
    def __init__(self, config: Config):
        self.config = config

    async def generate_reasoning(self, message: str, max_iterations=3) -> str:
        """Generates chain of thought reasoning iteratively with verification."""
        if not message.strip():
            logger.warning("Attempted to generate reasoning for empty message.")
            return ""

        reasoning = ""
        for i in range(max_iterations):
            cot_prompt = f"""
                You are an expert at breaking down complex questions using chain-of-thought prompting.
                Given the user query: '{message}', provide a detailed reasoning process, step-by-step,
                before giving the final answer. Start by explicitly stating the key steps to solve this problem.
                Current Reasoning: {reasoning}
            """

            new_reasoning = ""
            async for chunk in chat_with_lmstudio([{"role": "user", "content": cot_prompt}], self.config):
                new_reasoning += chunk

            reasoning = new_reasoning.strip()
            if self._verify_reasoning(reasoning):
                break
            logger.info(f"Reasoning iteration {i+1}: {reasoning[:100]}...")
            if tokenizer.count_tokens(reasoning) > 1000:
              logger.info(f"Reasoning exceeds 1000 token limit. Truncating.");
              reasoning = tokenizer.tokenizer.decode(tokenizer.tokenizer(reasoning, truncation=True, max_length=1000)['input_ids'])
              break;


        logger.info("Chain-of-thought reasoning generated.")
        return reasoning

    def _verify_reasoning(self, reasoning: str) -> bool:
        """Placeholder for verification logic."""
        if not reasoning:
            return False
        return True

cot_reasoning = ChainOfThoughtReasoning(config)


# ===========================
# LLM Streaming
# ===========================
async def chat_with_lmstudio(messages: List[Dict], config: Config, model_name: Optional[str] = None) -> AsyncGenerator[str, None]:
    """Stream responses from LM Studio's chat endpoint."""
    model_name = model_name or config.CHAT_MODEL
    url = f"{config.BASE_URL}/chat/completions"
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 1,
        "top_p": 0.9,
        "max_tokens": config.MODEL_MAX_TOKENS,
        "stream": True
    }

    async with httpx.AsyncClient(timeout=config.HTTPX_TIMEOUT) as client:
        try:
            logger.info("Requesting streamed chat completion.")
            async with limiter:
                async with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip() == "data: [DONE]":
                            logger.info("Received [DONE] from LLM.")
                            break
                        elif line.startswith("data:"):
                            data = json.loads(line[6:])
                            content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                yield content
        except httpx.ReadTimeout:
            logger.error("Timeout during LLM streaming.")
            yield "The operation timed out."
        except httpx.HTTPError as e:
            logger.error(f"HTTP error in LLM streaming: {e}")
            yield "HTTP error occurred."
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error: {e}")
            yield "JSON decode error."
        except Exception as e:
            logger.error(f"Unexpected error in LLM streaming: {e}")
            yield "Unexpected error."


# ===========================
# History Management
# ===========================
def load_history(config: Config) -> Dict:
    """Load conversation history from a file."""
    if not os.path.exists(config.HISTORY_FILE_PATH):
        logger.warning("No history file found.")
        return {"messages_history": []}
    try:
        with open(config.HISTORY_FILE_PATH, "r") as f:
            history = json.load(f)
        if not isinstance(history, dict) or "messages_history" not in history:
            logger.warning("History file format is invalid. Starting fresh.")
            return {"messages_history": []}

        # Validate message formats
        for msg in history["messages_history"]:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                logger.error("Invalid message format in history. Resetting history.")
                return {"messages_history": []}
        return history
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from history file: {e}")
        return {"messages_history": []}
    except Exception as e:
        logger.error(f"Error loading history: {e}")
        return {"messages_history": []}


# ===========================
# Chat Handler
# ===========================
async def chat_handler(message: str, state: Dict, internal_reasoning: bool,
                       model_name: str, temperature: float, top_p: float, min_output_tokens: int,
                       chatbot_history: List[List],
                       context_display: str, reasoning_steps_display: str,
                       config: Config) -> Tuple[List[List], Dict, str, str]:
    """Handles a new chat turn without file upload."""
    logger.info("Received new user message.")
    updated_chat = []
    try:
        original_message = sanitize_input(message)
        if not original_message:
            yield chatbot_history, state, "", ""
            return

        messages_history = state.get("messages_history", [])
        updated_chat = chatbot_history.copy() if chatbot_history else []

        # Get embeddings for the user message
        user_embedding = await embedding_model.get_embeddings(original_message)
        if user_embedding is None:
            updated_chat.append([original_message, "Failed to get embeddings."])
            yield updated_chat, state, "", ""
            return

        # Store message and embeddings
        messages_history.append({"role": "user", "content": original_message, "embedding": user_embedding.tolist()})
        if len(messages_history) > config.MAX_EMBEDDINGS:
            messages_history = messages_history[-config.MAX_EMBEDDINGS:]

        # Fetch context using graph-based method
        context_text = graph_context_retriever.get_context(original_message)

        # Create a system message using the context for the LLM
        system_message = f"You are a helpful assistant. Here is some context for you to use: {context_text}"
        history_for_llm = [{"role": "system", "content": system_message}]

        # Limit to last k messages for conversation context
        history_for_llm += [{"role": msg["role"], "content": msg["content"]} for msg in messages_history[-config.CONVERSATION_CONTEXT_DEPTH:]]

        # Token calculation
        total_length = sum(tokenizer.count_tokens(msg["content"]) for msg in messages_history)
        max_tokens = calculate_max_tokens(total_length, config)

        # Apply Chain-of-Thought reasoning
        reasoning_text = ""
        if internal_reasoning:
            reasoning_text = await cot_reasoning.generate_reasoning(original_message)

        updated_chat.append([original_message, None])
        response = ""

        try:
            async for chunk in chat_with_lmstudio(history_for_llm, config, model_name):
                response += chunk
                updated_chat[-1] = [original_message, response]
                yield updated_chat, {"messages_history": messages_history}, context_text, reasoning_text
        except Exception as e:
            logger.error(f"Error during chat response generation: {e}")
            updated_chat.append([original_message, "An error occurred."])
            yield updated_chat, state, "", ""
            return

        # Detect Bias
        if bias_detector.detect_bias(response):
            logger.warning(f"Bias detected in the response: {response}")
            updated_chat[-1] = [original_message, "Response contains potential bias."]

        messages_history.append({"role": "assistant", "content": response})
        new_state = {"messages_history": messages_history}

        try:
            with open(config.HISTORY_FILE_PATH, "w") as f:
                json.dump(new_state, f)
            logger.info("Conversation history saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save conversation history: {e}")

        yield updated_chat, new_state, context_text, reasoning_text

    except Exception as e:
        logger.error(f"Exception in chat_handler: {e}")
        updated_chat.append([original_message, "An error occurred while processing your request. Please try again later."])
        yield updated_chat, state, "", ""

# ===========================
# Gradio Interface
# ===========================
async def gradio_chat_interface(config: Config):
    """Creates and launches the Gradio interface."""
    demo = gr.Blocks(title="LM Studio Chat Interface")

    with demo:
        gr.Markdown("# 🚀 High-Performance Chat Interface")

        with gr.Row():
            with gr.Column(scale=2):
                chatbot_history = gr.Chatbot(label="Conversation", height=500)

                with gr.Row():
                    user_input = gr.Textbox(
                        label="Your Message",
                        placeholder="Type your message here...",
                        lines=2,
                    )
                    send_button = gr.Button("Send", variant="primary")

            with gr.Column(scale=1):
                context_display = gr.Textbox(label="Relevant Context", interactive=False, lines=5)
                reasoning_steps_display = gr.Textbox(label="Reasoning Steps", interactive=False, lines=10)

                with gr.Accordion("Advanced Settings", open=False):
                    model_selector = gr.Dropdown(
                        label="Select Model",
                        choices=[config.CHAT_MODEL, "Another_Model"],
                        value=config.CHAT_MODEL
                    )
                    temperature_slider = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=2.0,
                        step=0.1,
                        value=1.0
                    )
                    top_p_slider = gr.Slider(
                        label="Top-p",
                        minimum=0.1,
                        maximum=1.0,
                        step=0.1,
                        value=0.9
                    )
                    min_output_tokens_slider = gr.Slider(
                        label="Minimum Output Tokens",
                        minimum=1000,
                        maximum=20000,
                        step=100,
                        value=config.MIN_OUTPUT_TOKENS
                    )
                    internal_reasoning_checkbox = gr.Checkbox(label="Enable Internal Reasoning", value=False)

        history_state = load_history(config)
        embeddings_state = gr.State(history_state)

        # Connect events
        send_button.click(
            chat_handler,
            inputs=[
                user_input,
                embeddings_state,
                internal_reasoning_checkbox,
                model_selector,
                temperature_slider,
                top_p_slider,
                min_output_tokens_slider,
                chatbot_history,
                context_display,
                reasoning_steps_display,
                gr.State(config)
            ],
            outputs=[
                chatbot_history,
                embeddings_state,
                context_display,
                reasoning_steps_display
            ]
        )

        user_input.submit(
            chat_handler,
            inputs=[
                user_input,
                embeddings_state,
                internal_reasoning_checkbox,
                model_selector,
                temperature_slider,
                top_p_slider,
                min_output_tokens_slider,
                chatbot_history,
                context_display,
                reasoning_steps_display,
                gr.State(config)
            ],
            outputs=[
                chatbot_history,
                embeddings_state,
                context_display,
                reasoning_steps_display
            ]
        )

    logger.info("Launching Gradio interface.")
    await demo.queue().launch(share=True, server_name="0.0.0.0", server_port=7860)

# ===========================
# Main Execution
# ===========================
if __name__ == "__main__":
    logger.info("Starting main execution.")
    try:
        vector_db.add_item("Pandas can be used to read CSV files.")
        vector_db.add_item("Matplotlib can create bar charts.")
        vector_db.add_item("The average can be calculated by grouping data in Pandas.")
        vector_db.add_item("FAISS allows efficient similarity search over large datasets.")
        vector_db.add_item("Neo4j is a powerful graph database for storing interconnected data.")
        vector_db.add_item("Transformers can be fine-tuned for specific NLP tasks.")
        vector_db.add_item("Gradio provides an easy way to create web interfaces for ML models.")
        vector_db.add_item("Cosine similarity is commonly used in text similarity tasks.")
        vector_db.add_item("HTTPX is an asynchronous HTTP client for Python.")
        vector_db.add_item("Environment variables can be managed using dotenv.")

        asyncio.run(gradio_chat_interface(config))
    except Exception as e:
        logger.exception(f"An error occurred during startup: {e}")
