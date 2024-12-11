# 🚀 High-Performance Chat Interface for LM Studio – Enhanced Version

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository provides a **powerful and efficient web-based chat application** designed for seamless interaction with AI language models hosted on **LM Studio**. This application leverages **GPU acceleration** (where available) and **asynchronous operations** to deliver a highly responsive and performant user experience.

This enhanced version incorporates several advanced features, including a sophisticated **internal reasoning system**, **dynamic token management**, and **context handling** using embeddings. Notably, this project is implemented as a **self-contained, single-file Python script** for ease of use and deployment.

## ✨ Key Features

-   **LM Studio Integration:** Connects directly to your local LM Studio server to access a variety of large language models (LLMs).
-   **Real-time Interaction:** Built on the Gradio framework for a dynamic and interactive chat experience.
-   **Contextual Conversations:** Maintains conversation history and utilizes embeddings to provide contextually relevant responses.
-   **GPU Acceleration:** Automatically detects and utilizes available GPU resources for faster processing.
-   **Asynchronous Operations:** Employs `async` and `await` to ensure non-blocking API calls, leading to a more responsive user interface.
-   **Dynamic Token Handling:** Intelligently calculates the maximum number of tokens for responses, prioritizing a minimum output length of 8000 tokens while respecting model limitations.
-   **Internal Reasoning Mechanism:**
    -   **Problem Decomposition:** Breaks down user queries into smaller sub-components using regular expressions.
    -   **Embedding Generation:** Generates embeddings for each sub-component using a dedicated embedding model.
    -   **Knowledge Retrieval:** Searches an in-memory knowledge base for relevant information using cosine similarity between embeddings.
    -   **Response Synthesis:** Combines retrieved knowledge, the original query, and generated reasoning steps to produce a comprehensive and informed response.
-   **In-Memory Knowledge Base:** Stores and retrieves text snippets with their embeddings to enhance context awareness.
-   **File Upload:** Enables users to upload `.txt` files, providing additional context to the chatbot.
-   **Persistent Conversation History:** Saves and loads conversation history (including embeddings) to a JSON file (`chat_history.json`), allowing for continuous conversations across sessions.
-   **Robust Error Handling:** Includes comprehensive error handling for various scenarios, such as API errors, file handling issues, and JSON parsing errors, providing informative feedback to the user.
-   **Streamlined and Efficient:** Optimized for performance and memory usage.
-   **Real-Time Updates:** The Gradio interface updates dynamically as the response is generated.
-   **Simplified Configuration:** Easily configurable using constants and environment variables.

## 	📚 Table of Contents

-   [Installation](#-installation)
    -   [Prerequisites](#-prerequisites)
    -   [Setup Steps](#-setup-steps)
-   [Usage](#-usage)
-   [Configuration](#-configuration)
-   [Internal Reasoning System](#-internal-reasoning-system)
-   [In-Memory Knowledge Base](#-in-memory-knowledge-base)
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
    -   **Chat Model:**  `bartowski/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-IQ2_M.gguf` (or your preferred model).
    -   **Embedding Model:** `nomic-embed-text-v1.5.Q8_0.gguf`

### 	✅ Setup Steps

1. **Clone this repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

    Replace `<repository_url>` with the URL of this repository and `<repository_directory>` with the directory you want to clone into.

2. **Install the required libraries:**

    ```bash
    pip install gradio httpx numpy torch
    ```

3. **(Optional) Set the environment variable:**

    ```bash
    export LMSTUDIO_API_BASE_URL="http://localhost:1234/v1"
    ```

    -   If your LM Studio server is running on a different host or port, adjust the `LMSTUDIO_API_BASE_URL` accordingly.
    -   If you don't set this environment variable, the code will default to `http://localhost:1234/v1`.

## 🕹️ Usage

1. **Run the application:**

    ```bash
    python lmstudio_gradio2.py
    ```

2. **Access the interface:**

    -   Open your web browser and go to `http://0.0.0.0:7860/` (or the URL indicated in the console output).
    -   If you use Gradio's `share=True` option, a temporary public URL will be provided.

3. **Interact with the chatbot:**

    -   Type your message into the **"Your Message"** text box.
    -   (Optional) Upload a `.txt` file using the **"Upload Context File"** button to provide additional context.
    -   Click **"Send"** or press **Enter** to send the message.
    -   The chatbot will respond in the **"Conversation"** area.
    -   View the **"Relevant Context"** and **"Reasoning Steps"** to see how the model is processing your input.
    -   The **"Embeddings History"** shows a preview of the embeddings generated for recent messages.

4. **Advanced Settings:**

    -   Click on **"Advanced Settings"** to access the following options:
        -   **Select Model:** Choose a different chat model (if available in your LM Studio setup).
        -   **Temperature:** Adjust the randomness of the model's output (higher values = more creative, lower values = more deterministic).
        -   **Top-p:** Control the diversity of the generated tokens (lower values = more focused on the most likely tokens).
        -   **Enable Internal Reasoning:** Toggle the internal reasoning system on or off.

## ⚙️ Configuration

The following constants and environment variables can be used to configure the application:

| Constant                     | Description                                                                                                                                                                                                           | Default Value            |
| :--------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------- |
| `BASE_URL`                   | The base URL of your LM Studio API. (Can also be set via the `LMSTUDIO_API_BASE_URL` environment variable.)                                                                                                         | `"http://localhost:1234/v1"` |
| `USE_GPU`                    | Automatically set to `True` if a GPU is available, `False` otherwise.                                                                                                                                                | `True` if GPU available |
| `MODEL_MAX_TOKENS`           | The maximum number of tokens your chat model can handle.                                                                                                                                                             | `32768`                  |
| `EMBEDDING_MODEL_MAX_TOKENS` | The maximum number of tokens for embedding generation.                                                                                                                                                                 | `8192`                   |
| `AVERAGE_CHARS_PER_TOKEN`    | An estimate of the average number of characters per token.                                                                                                                                                           | `4`                      |
| `BUFFER_TOKENS`             | A buffer to reserve tokens for internal use.                                                                                                                                                                        | `1500`                   |
| `MIN_OUTPUT_TOKENS`          | The minimum number of tokens for the generated response (set to 8000 in this version).                                                                                                                               | `8000`                   |
| `MAX_EMBEDDINGS`             | The maximum number of embeddings to store in memory.                                                                                                                                                                  | `100`                    |
| `HTTPX_TIMEOUT`              | The timeout (in seconds) for HTTP requests to the LM Studio API.                                                                                                                                                      | `300`                    |
| `HISTORY_FILE_PATH`          | The path to the file where conversation history is saved.                                                                                                                                                           | `"chat_history.json"`    |
| `CHAT_MODEL`                 | The name of the chat model to use (must be loaded in LM Studio).                                                                                                                                                        | Specific to your setup  |
| `EMBEDDING_MODEL`           | The name of the embedding model to use (must be loaded in LM Studio).                                                                                                                                                  | Specific to your setup  |

## 🧠 Internal Reasoning System

The internal reasoning system enhances the chatbot's ability to understand and respond to complex queries. It works as follows:

1. **Embedding Generation:** When a user sends a message, the system generates embeddings for the message using the specified embedding model.
2. **Problem Decomposition:** The system attempts to break down the user's message into smaller sub-components using regular expression patterns. This helps identify potential actions or sub-problems within the query.
3. **Knowledge Retrieval:** For each sub-component, the system searches the in-memory knowledge base for relevant information. The search is performed using cosine similarity between the sub-component's embedding and the embeddings of items in the knowledge base.
4. **Context Selection:** The most relevant knowledge items (based on similarity scores) are selected as context for the LM Studio model.
5. **Reasoning Steps Generation:** The system generates a series of internal reasoning steps that describe the analysis, decomposition, and knowledge retrieval process. These steps are displayed in the "Reasoning Steps" section of the Gradio interface, providing transparency into the model's thought process.
6. **Response Synthesis:** Finally, the system combines the original user message, the selected context, and the reasoning steps to generate a comprehensive and informed response using the LM Studio chat completion API.

## 💾 In-Memory Knowledge Base

The `InMemoryKnowledgeDB` class provides a simple in-memory knowledge base for storing and retrieving text snippets and their corresponding embeddings.

-   **`add_item(text, embedding)`:** Adds a text snippet and its embedding to the knowledge base.
-   **`search(query_embedding, k=3)`:** Searches the knowledge base for the `k` most similar items to the given `query_embedding` using cosine similarity. Returns a list of tuples, where each tuple contains the text snippet and its similarity score.

**Note:** This in-memory knowledge base is suitable for small to medium-sized datasets. For larger datasets, consider using a more scalable database solution.

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

A: This application can work with any language model hosted on LM Studio that supports the chat completions endpoint. The provided code specifically mentions the `bartowski/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-IQ2_M.gguf` and `nomic-embed-text-v1.5.Q8_0.gguf` models, but you can easily modify the `CHAT_MODEL` and `EMBEDDING_MODEL` constants to use other models.

**Q: How do I add knowledge to the in-memory knowledge base?**

A: The example code adds a few sample knowledge items in the `if __name__ == "__main__":` block. You can add more items by calling the `knowledge_db.add_item(text, embedding)` function, where `text` is the text snippet and `embedding` is the corresponding embedding (a NumPy array) generated using the `get_embeddings` function.

**Q: How can I improve the performance of the application?**

A: Here are some tips for improving performance:

-   **Use a GPU:** If you have a compatible GPU, make sure `USE_GPU` is set to `True` to enable GPU acceleration.
-   **Optimize Embeddings:** Experiment with different embedding models to find one that provides good performance for your use case.
-   **Reduce Context Size:** If the conversation history becomes very long, consider truncating it or summarizing older parts of the conversation to reduce the number of tokens sent to the model.
-   **Increase `MIN_OUTPUT_TOKENS`:** If the model is consistently generating very short responses, try increasing `MIN_OUTPUT_TOKENS` to encourage longer outputs.
-   **Use a Scalable Database:** For large knowledge bases, consider using a persistent database instead of the in-memory knowledge base.

## 📞 Contact

For questions, suggestions, or contributions, please contact:

-   **Alec Bartowski**
-   **Email:** alecbartowski@gmail.com
-   **GitHub:** [alecb42](https://github.com/alecb42)

Feel free to reach out through GitHub Issues or Pull Requests for any feedback or contributions to the project.
