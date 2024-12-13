# 🚀 High-Performance Chat Interface for LM Studio

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository provides a **powerful and efficient web-based chat application** designed for seamless interaction with AI language models hosted on **LM Studio**. This application is optimized for **GPU acceleration** (when available) and uses **asynchronous operations** to ensure a highly responsive and performant user experience.

This version incorporates several advanced features, including an **iterative chain-of-thought reasoning system**, **dynamic token management**, and **context handling** using a knowledge graph and in-memory GPU vector database. Notably, this project is implemented as a **self-contained, single-file Python script** for ease of use and deployment.

## ✨ Key Features

-   **LM Studio Integration:** Connects directly to your local LM Studio server to access various large language models (LLMs).
-   **Real-time Interaction:** Built using the Gradio framework for a dynamic and interactive chat experience.
-   **Contextual Conversations:** Maintains conversation history and utilizes a knowledge graph and embeddings to provide contextually relevant responses.
-   **GPU Acceleration:** Automatically detects and utilizes available GPU resources for faster processing with Pytorch tensors.
-   **Asynchronous Operations:** Employs `async` and `await` to ensure non-blocking API calls, leading to a more responsive user interface.
-   **Dynamic Token Handling:** Intelligently calculates the maximum number of tokens for responses, prioritizing a minimum output length while respecting model limitations.
-   **Iterative Chain-of-Thought Reasoning:**
    -   **Detailed Reasoning:** Generates responses based on a step-by-step reasoning process.
    -   **Verification Loop:** Iteratively refines reasoning using self-verification techniques (placeholder for more advanced verification).
-   **Knowledge Graph Context:** Uses a Neo4j database to store interconnected information, enabling richer contextual understanding.
-   **In-Memory GPU Vector Database**: Stores and retrieves text snippets with their embeddings in GPU memory to enhance context awareness.
-   **Persistent Conversation History:** Saves and loads conversation history (including embeddings) to a JSON file (`chat_history.json`), allowing for continuous conversations across sessions.
-   **Bias Detection**: Implements a basic bias detection system using a pre-trained model.
-   **Robust Error Handling:** Includes comprehensive error handling for various scenarios, such as API errors, and JSON parsing errors, providing informative feedback to the user.
-   **Streamlined and Efficient:** Optimized for performance and memory usage.
-   **Real-Time Updates:** The Gradio interface updates dynamically as the response is generated.
-   **Simplified Configuration:** Easily configurable using environment variables.

## 📚 Table of Contents

-   [Installation](#-installation)
    -   [Prerequisites](#-prerequisites)
    -   [Setup Steps](#-setup-steps)
-   [Usage](#-usage)
-   [Configuration](#-configuration)
-   [Iterative Chain-of-Thought Reasoning System](#-iterative-chain-of-thought-reasoning-system)
-   [In-Memory GPU Vector Database](#-in-memory-gpu-vector-database)
-   [Knowledge Graph Context](#-knowledge-graph-context)
-    [Bias Detection](#-bias-detection)
-   [Error Handling](#-error-handling)
-   [Contributing](#-contributing)
-   [License](#-license)
-   [Code Structure](#-code-structure)
-   [Deployment](#-deployment)
-   [FAQ](#-faq)
-   [Contact](#-contact)

## 💻 Installation

### ✅ Prerequisites

-   **Python 3.8 or higher**
-   **LM Studio:** A running instance of LM Studio with the following models loaded:
    -   **Chat Model:** `Qwen/Qwen2.5-Coder-32B-Instruct` (or your preferred Qwen Coder based model).
    -   **Embedding Model:** `nomic-ai/nomic-embed-text-v1.5`

-   **Neo4j Database**: A running instance of the Neo4j graph database accessible from your machine.

### ✅ Setup Steps

1. **Clone this repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

    Replace `<repository_url>` with the URL of this repository and `<repository_directory>` with the directory you want to clone into.

2. **Install the required libraries:**

    ```bash
    pip install httpx numpy torch torchvision torchaudio transformers gradio chromadb scikit-learn neo4j-driver python-dotenv aiolimiter
    ```

3. **(Optional) Set the environment variables:**

   - Configure your API access and Neo4j settings using a `.env` file.
    -   If your LM Studio server is running on a different host or port, adjust the `LMSTUDIO_API_BASE_URL` accordingly.
    -   If your Neo4j database is running on a different host or port, adjust `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD`.
   -   If you don't set these environment variable, the code will default to `http://localhost:1234/v1` for the LM studio API, and `bolt://localhost:7687` with user `neo4j` and password `password` for Neo4j.

## 🕹️ Usage

1. **Run the application:**

    ```bash
    python lmstudio_gradio2.py
    ```

2. **Access the interface:**

    -   Open your web browser and go to the URL provided by Gradio in the console output.

3. **Interact with the chatbot:**

    -   Type your message into the **"Your Message"** text box.
    -   Click **"Send"** or press **Enter** to send the message.
    -   The chatbot will respond in the **"Conversation"** area.
    -   View the **"Relevant Context"** and **"Reasoning Steps"** to see how the model is processing your input.

4.  **Advanced Settings:**

    -   Click on **"Advanced Settings"** to access the following options:
        -   **Select Model:** Choose a different chat model (if available in your LM Studio setup).
        -   **Temperature:** Adjust the randomness of the model's output (higher values = more creative, lower values = more deterministic).
        -   **Top-p:** Control the diversity of the generated tokens (lower values = more focused on the most likely tokens).
        -   **Minimum Output Tokens:** Controls the minimum length of the output from the model.
        -   **Enable Internal Reasoning:** Toggle the internal reasoning system on or off.

## ⚙️ Configuration

The following constants and environment variables can be used to configure the application:

| Constant                     | Description                                                                                                                                                                                                 | Default Value                   |
| :--------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------ |
| `BASE_URL`                   | The base URL of your LM Studio API. (Can also be set via the `LMSTUDIO_API_BASE_URL` environment variable.)                                                                                                   | `"http://localhost:1234/v1"`     |
| `USE_GPU`                    | Automatically set to `True` if a GPU is available, `False` otherwise.                                                                                                                                       | `True` if GPU available        |
| `MODEL_MAX_TOKENS`           | The maximum number of tokens your chat model can handle.                                                                                                                                                         | `32768`                        |
| `EMBEDDING_MODEL_MAX_TOKENS` | The maximum number of tokens for embedding generation.                                                                                                                                                         | `8192`                         |
| `BUFFER_TOKENS`              | A buffer to reserve tokens for internal use.                                                                                                                                                                 | `1500`                          |
| `MIN_OUTPUT_TOKENS`           | The minimum number of tokens for the generated response (set to 8000 in this version).                                                                                                                              | `8000`                          |
| `MAX_EMBEDDINGS`              | The maximum number of embeddings to store in memory.                                                                                                                                                                  | `100`                           |
| `HTTPX_TIMEOUT`              | The timeout (in seconds) for HTTP requests to the LM Studio API.                                                                                                                                                     | `300`                           |
| `HISTORY_FILE_PATH`          | The path to the file where conversation history is saved.                                                                                                                                                        | `"chat_history.json"`           |
| `CHAT_MODEL`                 | The name of the chat model to use (must be loaded in LM Studio).                                                                                                                                                     | Specific to your setup      |
| `EMBEDDING_MODEL`            | The name of the embedding model to use (must be loaded in LM Studio).                                                                                                                                               | Specific to your setup       |
|`NEO4J_URI`             | The URI of your Neo4j database.                                                                                                                                                              |  `bolt://localhost:7687`  |
|`NEO4J_USER`            | The username for your Neo4j database.                                                                                                                                                                      | `"neo4j"`  |
|`NEO4J_PASSWORD`         | The password for your Neo4j database.                                                                                                                                                                       | `"password"`  |
| `EMBEDDING_DIMS`  | The dimensionality of the embeddings. Make sure it matches the embedding model output dimension. | `768`

## 🧠 Iterative Chain-of-Thought Reasoning System

The iterative chain-of-thought reasoning system enhances the chatbot's ability to respond to complex queries by:

1.  **Step-by-Step Reasoning:** The system uses the LLM to generate a step-by-step reasoning process before producing the final answer.
2.  **Iterative Refinement:** The system can iteratively refine the reasoning steps using self-verification techniques (currently a placeholder, but designed for future enhancement).
3.  **Transparency:** The generated reasoning steps are displayed to the user, providing insights into the model's thought process.

## 💾 In-Memory GPU Vector Database

The `InMemoryGPUVectorDB` class provides an in-memory vector database which performs computations on the GPU using PyTorch tensors.

-   **`add_item(text, metadata=None)`:** Adds a text snippet and its embedding to the database, calculated and stored as a PyTorch tensor.
-   **`search(query_text, k=3, diversity_factor=0.5)`:** Searches the database for the `k` most similar items to the given `query_text` using cosine similarity. MMR is used to diversify the results. Returns a list of tuples, where each tuple contains the text snippet and its similarity score.

**Note:** This in-memory knowledge base is suitable for small to medium-sized datasets. For larger datasets, consider using a more scalable database solution.

## 🕸️ Knowledge Graph Context

The application integrates with a Neo4j database to retrieve context using a knowledge graph-based approach. The steps are:
- Get embedding of user query.
- Retrieve relevant nodes from the Neo4j graph database.
- Return results as context for the LLM.
To add items to the graph, use the `add_item` method in the `KnowledgeGraphRetriever` class.

## ⚖️ Bias Detection

The application includes a basic bias detection feature:
-   Uses a Hugging Face Transformer model to analyze the response.
-  If the response has a bias score greater than 0.65 (arbitrary), then it will warn the user.
- This component is a placeholder and would require more thought in a production system.

## ❗ Error Handling

The code includes robust error handling to catch various exceptions that might occur during operation:

-   **`httpx.HTTPError`:** Catches HTTP errors during API requests.
-   **`json.JSONDecodeError`:** Handles errors in decoding JSON responses.
-   **`UnicodeDecodeError`:** Catches errors when decoding uploaded files.
-   **`ValueError`:** Handles invalid file types or other value errors.
-   **`Exception`:** Catches any other unexpected exceptions.

Error messages are logged to the console and displayed in the Gradio interface to inform the user.

## 🤝 Contributing

Contributions to this project are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix:
    ```bash
    git checkout -b feature/your-feature-name
    ```
    or
    ```bash
    git checkout -b bugfix/issue-description
    ```
3.  Make your changes and commit them with clear, descriptive commit messages.
4.  Push your branch to your forked repository.
5.  Create a pull request to the `main` branch of the original repository.

Please ensure your code adheres to the existing code style and includes appropriate comments and documentation.

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🚀 Deployment
You can deploy this Gradio application to Hugging Face Spaces for easy sharing and access. Here's how:

1. **Create a Hugging Face Account:** If you don't have one already, sign up for a free account at [huggingface.co](https://huggingface.co/).
2. **Create a New Space:**
   - Go to your Hugging Face profile.
   - Click on "New Space".
   - Give your Space a name (e.g., `lmstudio-chat-interface`).
   - Choose a license (e.g., MIT).
   - Select "Gradio" as the SDK.
   - Choose "Public" for public accessibility or "Private" if you want to restrict access.
   - Click "Create Space".
3. **Clone the Space's Repository:**
   - In your Space, you'll see instructions to clone the repository. It will be something like:
     ```bash
     git clone https://huggingface.co/spaces/yourusername/your-space-name
     ```
   - Replace `yourusername` and `your-space-name` with your Hugging Face username and the name of your Space.
4. **Copy Files to the Cloned Repository:**
   - Copy the `lmstudio_gradio2.py` file into the cloned repository.
   - Create a `requirements.txt` file listing the required dependencies:
     ```
     gradio
     httpx
     numpy
     torch
     transformers
     aiolimiter
     neo4j-driver
    ```
5. **Commit and Push Changes:**
   - Commit your changes:
     ```bash
     git add lmstudio_gradio2.py requirements.txt
     git commit -m "Add lmstudio chat interface and requirements"
     ```
   - Push the changes to Hugging Face Spaces:
     ```bash
     git push
     ```
6. **Wait for the Build:** Hugging Face Spaces will automatically build your application. This might take a few minutes.
7. **Access Your App:** Once the build is complete, your Gradio application will be live at the URL provided by Hugging Face Spaces (e.g., `https://yourusername-your-space-name.hf.space`).

**Notes:**

-  You might need to adjust your LM Studio setup to allow external connections if you want to use a model running on your local machine. This usually involves port forwarding and configuring your firewall. Consider the security implications of doing this.
- If you have a large knowledge base, consider using a persistent database instead of the in-memory database for better scalability when deploying to Hugging Face Spaces.
- Hugging Face Spaces provides free CPU-based hosting. For GPU support you might need a paid account.

## ❓ FAQ

**Q: What is LM Studio?**

A: LM Studio is a platform for hosting and interacting with various AI language models. It provides an API that allows applications like this one to communicate with the models.

**Q: What models are supported?**

A: This application is optimized for the `Qwen/Qwen2.5-Coder-32B-Instruct` model, but should work with any models supported by LM Studio's chat completion API. You will need to modify the `CHAT_MODEL` and `EMBEDDING_MODEL_HUGGING_FACE_NAME` constants in the code to use other models.

**Q: How do I add knowledge to the in-memory knowledge base?**

A: The example code adds a few sample knowledge items in the `if __name__ == "__main__":` block. You can add more items by calling the `vector_db.add_item(text)` function, where `text` is the text snippet.

**Q: How can I improve the performance of the application?**

A: Here are some tips for improving performance:

-  Make sure you have a compatible GPU installed and that torch is utilizing it for computation.
-   Consider a scalable database instead of the in-memory knowledge base for larger datasets.
-   Tune parameters for prompt generation and model interaction to speed up responses.

## 📞 Contact

If you have any questions or need further assistance, feel free to reach out by opening an issue or submitting a pull request to the repository.
