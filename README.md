# 🚀 High-Performance Chat Interface for LM Studio with Retrieval-Augmented Generation (RAG)

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository provides a **powerful and efficient web-based chat application** designed for seamless interaction with AI language models hosted on **LM Studio**. This application is optimized for **GPU acceleration** (when available) and uses **asynchronous operations** to ensure a highly responsive and performant user experience.

This version incorporates several advanced features, including an **intermediate reasoning step**, **dynamic token management**, and **context handling** using a FAISS based vector database for Retrieval Augmented Generation (RAG). This project is implemented as a **self-contained, single-file Python script** for ease of use and deployment.

**Important Note:** This project is compatible with LM Studio version 0.2.31. **It is NOT compatible with later versions, including LM Studio 0.3.5 and beyond, due to changes in the API and model handling.** Please ensure you are using LM Studio version 0.2.31 for this application to function correctly.

## ✨ Key Features

-   **LM Studio Integration:** Connects directly to your local LM Studio server to access various large language models (LLMs).
-   **Real-time Interaction:** Built using the Gradio framework for a dynamic and interactive chat experience.
-   **Retrieval Augmented Generation (RAG):**  Uses an in memory vector database to enhance responses with relevant context.
-   **GPU Acceleration:** Automatically detects and utilizes available GPU resources for faster processing with FAISS.
-   **Asynchronous Operations:** Employs `async` and `await` to ensure non-blocking API calls, leading to a more responsive user interface.
-   **Dynamic Token Handling:** Intelligently calculates the maximum number of tokens for responses, prioritizing a minimum output length while respecting model limitations.
-   **Intermediate Reasoning:** Generates an intermediate reasoning step to improve response quality.
-   **In-Memory Vector Database**: Stores and retrieves text snippets with their embeddings to enhance context awareness.
-   **Streamlined and Efficient:** Optimized for performance and memory usage.
-   **Real-Time Updates:** The Gradio interface updates dynamically as the response is generated.
-   **Simplified Configuration:** Easily configurable using environment variables.

## 📚 Table of Contents

-   [Installation](#-installation)
    -   [Prerequisites](#-prerequisites)
    -   [Setup Steps (with Virtual Environment)](#-setup-steps-with-virtual-environment)
    - [LM Studio Setup](#-lm-studio-setup)
-   [Usage](#-usage)
-   [Configuration](#-configuration)
-   [Intermediate Reasoning](#-intermediate-reasoning)
-   [Vector Database and Retrieval Augmented Generation (RAG)](#-vector-database-and-retrieval-augmented-generation-rag)
-   [Contributing](#-contributing)
-   [License](#-license)
-   [FAQ](#-faq)
-   [Contact](#-contact)

## 💻 Installation

### ✅ Prerequisites

-   **Python 3.8 or higher**
-   **LM Studio:** Version 0.2.31 (**Important: This project is NOT compatible with LM Studio 0.3.5 or later.**)
-   **Chat Model:** `Qwen/Qwen2.5-Coder-32B-Instruct` (or your preferred Qwen Coder based model).
-   **Embedding Model:** `nomic-ai/nomic-embed-text-v1.5`
- **knowledge_base.jsonl:** A file containing your knowledge base in JSONL format. Each line should be a JSON object with a `text` field containing the text and an `embedding` field containing the corresponding embedding. You can generate this file using the provided `embedding_generator.py` script.

### ✅ Setup Steps (with Virtual Environment)

1. **Create and Activate a Virtual Environment:**

    -   Open your terminal or command prompt.
    -   Navigate to the directory where you cloned this repository.
    -   Create a virtual environment named `venv` (or any name you prefer):

        ```bash
        python3 -m venv venv
        ```

    -   Activate the virtual environment:

        -   **On Windows:**

            ```bash
            venv\Scripts\activate
            ```

        -   **On macOS/Linux:**

            ```bash
            source venv/bin/activate
            ```

2. **Install the required libraries within the virtual environment:**
    ```bash
    pip install aiohttp tiktoken gradio faiss-cpu numpy
    ```
    If you have a CUDA-enabled GPU and want to use FAISS with GPU support, install `faiss-gpu` instead:
    ```bash
    pip install aiohttp tiktoken gradio faiss-gpu numpy
    ```

3. **(Optional) Set the environment variables:**

    -   Configure your API access using a `.env` file or directly in your environment.
    -   If your LM Studio server is running on a different host or port, adjust the `LMSTUDIO_API_BASE_URL` accordingly.
    -   If you don't set these environment variables, the code will default to `http://localhost:1234` for the LM Studio API.

### ✅ LM Studio Setup

1. **Download and Install LM Studio:**
    -   Download LM Studio **version 0.2.31** from the official website or a reliable archive. **Do not use version 0.3.5 or later as they are not compatible.**
    -   Install the application on your system.

2. **Download Models:**
    -   Open LM Studio and navigate to the "Model Library".
    -   Search for and download the following models:
        -   **Chat Model:** `Qwen/Qwen2.5-Coder-32B-Instruct-IQ2_M`
        -   **Embedding Model:** `nomic-ai/nomic-embed-text-v1.5-Q8_0`

3. **Configure the Chat Model:**
    -   In LM Studio, select the `Qwen/Qwen2.5-Coder-32B-Instruct-IQ2_M` model.
    -   Adjust the following settings in the "Model Configuration" panel:

        | Setting                          | Value   |
        | :------------------------------- | :------ |
        | Keep entire model in RAM         | Checked (On) |
        | Prompt eval batch size (`n_batch`) | `512`   |
        | Flash Attention (`flash_attn`)   | Checked (On) |
        | K Cache Quant (`cache_type_k`)   | `q4_0`  |
        | V Cache Quant (`cache_type_v`)   | `q4_0`  |
        | Rotary Position Embedding (RoPE) Frequency Scale (`rope_freq_scale`) | `0`     |
        | Rotary Position Embedding (RoPE) Frequency Base (`rope_freq_base`) | `0`     |

    - These settings will require the model to be reloaded if changed.
    - Apply these settings.

4. **Configure the Embedding Model:**
    -   In LM Studio, select the `nomic-ai/nomic-embed-text-v1.5-Q8_0` model.
    -   You can usually leave the default settings for the embedding model, but ensure it's loaded and ready for use.
    -   Set the **K Cache Quant** and **V Cache Quant** to **`q4_0`**

5. **Verify Server Operation:**
    -   Ensure that the LM Studio server is running and accessible. By default, it should be running on `http://localhost:1234`. Click the **Start Server** button in LM Studio.

## 🕹️ Usage

1. **Start LM Studio and Load Models:**
    -   Open LM Studio.
    -   Load the `Qwen/Qwen2.5-Coder-32B-Instruct-IQ2_M` chat model and the `nomic-ai/nomic-embed-text-v1.5-Q8_0` embedding model.
    -   Ensure the **K Cache Quant** and **V Cache Quant** are set to `q4_0` for both models.
    -   Start the LM Studio server by clicking the **Start Server** button.

2. **Activate the Virtual Environment (if not already active):**

    -   **On Windows:**

        ```bash
        venv\Scripts\activate
        ```

    -   **On macOS/Linux:**

        ```bash
        source venv/bin/activate
        ```

3. **Generate Embeddings for your Knowledge Base (if you haven't already):**

    ```bash
    python embedding_generator.py
    ```

    This will create a `knowledge_base.jsonl` file in the same directory. This file will contain the embeddings for each item in the `YOUR_KNOWLEDGE_BASE` list defined at the bottom of the `embedding_generator.py` script. Modify the `YOUR_KNOWLEDGE_BASE` list with your desired data.

4. **Run the Gradio application:**

    ```bash
    python lmstudio_gradio_rag.py
    ```

5. **Access the interface:**

    -   Open your web browser and go to the URL provided by Gradio in the console output.

6. **Interact with the chatbot:**

    -   Type your message into the **"Your Message"** text box.
    -   Click **"Send"** or press **Enter** to send the message.
    -   The chatbot will respond in the **"Conversation"** area, utilizing enhanced reasoning, intermediate steps, and embeddings retrieved from your knowledge base.

## ⚙️ Configuration

The following constants and environment variables can be used to configure the application:

| Constant/Variable        | Description                                                                                                                 | Default Value                   |
| :----------------------- | :-------------------------------------------------------------------------------------------------------------------------- | :------------------------------ |
| `LM_STUDIO_URL`          | The base URL of your LM Studio API. (Can also be set via the `LMSTUDIO_API_BASE_URL` environment variable.)                 | `"http://localhost:1234"`     |
| `CHAT_MODEL_ID`           | The name of the chat model to use (must be loaded in LM Studio).                                                             | `"Qwen2.5-Coder-32B-Instruct-IQ2_M.gguf"` |
| `EMBEDDING_MODEL_ID`      | The name of the embedding model to use (must be loaded in LM Studio).                                                      | `"nomic-embed-text-v1.5.Q8_0.gguf"` |
| `MAX_TOKENS`             | The maximum number of tokens your chat model can handle.                                                                    | `32768`                        |
| `EMBEDDING_CTX_LENGTH`   | The maximum number of tokens for embedding generation.                                                                     | `2048`                         |
| `EMBEDDING_ENCODING`     | The encoding used for tokenization.                                                                                         | `"cl100k_base"`               |
| `EMBEDDING_DIMENSION`    | The dimensionality of the embeddings.                                                                                       | `768`                          |
| `K_RETRIEVAL`            | The number of similar items to retrieve from the knowledge base for context.                                                | `3`                            |

## 🧠 Intermediate Reasoning

The chat interface now includes an intermediate reasoning step. Before generating the final response, the model first generates a plan or thought process to address the user's query. This intermediate step is shown in the conversation history and can improve the quality and relevance of the final response.

## 💾 Vector Database and Retrieval Augmented Generation (RAG)

The `KnowledgeBase` class provides an in-memory vector database using FAISS for efficient similarity search.

-   **`add_item(text, embedding)`:** Adds a text snippet and its embedding to the database.
-   **`search(query_embedding, k)`:** Searches the database for the `k` most similar items to the given `query_embedding` using L2 distance. Returns a list of the most similar text snippets.
-   **`load_from_file(filepath)`:** Loads knowledge items and their embeddings from a JSONL file.

The chat interface uses this vector database to retrieve relevant context for each user query:

1. **Generate Embedding:** The user's query (or the intermediate reasoning step) is converted into an embedding using the embedding model.
2. **Retrieve Context:** The `KnowledgeBase.search()` function is used to find the most similar items in the knowledge base.
3. **Augment Prompt:** The retrieved context is added to the prompt sent to the chat model, along with the conversation history.

This process allows the model to generate responses that are more informed and relevant to the user's query by leveraging the information stored in the knowledge base.

**Note:** This in-memory knowledge base is suitable for small to medium-sized datasets. For larger datasets, consider using a more scalable database solution.

## 🤝 Contributing

Contributions to this project are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix:

    ```bash
    git checkout -b feature/your-feature-name
    ```

    or

    ```bash
    git checkout -b bugfix/issue-description
    ```

3. Make your changes and commit them with clear, descriptive commit messages.
4. Push your branch to your forked repository.
5. Create a pull request to the `main` branch of the original repository.

Please ensure your code adheres to the existing code style and includes appropriate comments and documentation.

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## ❓ FAQ

**Q: What is LM Studio?**

A: LM Studio is a platform for hosting and interacting with various AI language models. It provides an API that allows applications like this one to communicate with the models.

**Q: What models are supported?**

A: This application is optimized for the `Qwen/Qwen2.5-Coder-32B-Instruct` model, but should work with any models supported by LM Studio's chat completion API. You will need to modify the `CHAT_MODEL_ID` and `EMBEDDING_MODEL_ID` constants in the code to use other models.

**Q: How do I add knowledge to the knowledge base?**

A: Add items to the `YOUR_KNOWLEDGE_BASE` list in `embedding_generator.py` and run the script to generate the `knowledge_base.jsonl` file.

**Q: How can I improve the performance of the application?**

A: Here are some tips for improving performance:

-   Make sure you have a compatible GPU installed and that you are using `faiss-gpu` for FAISS.
-   Consider a scalable database instead of the in-memory knowledge base for larger datasets.
-   Tune parameters for prompt generation and model interaction to speed up responses.

**Q: Why is this not compatible with LM Studio 0.3.5 or later?**

A: Later versions of LM Studio have introduced significant changes to the API and model handling, which are not compatible with the current implementation of this project. Using LM Studio 0.2.31 is required for this application to function correctly. They also removed K and V cache Quant, which really slows down output generation.

## 📞 Contact

If you have any questions or need further assistance, feel free to reach out by opening an issue or submitting a pull request to the repository.
