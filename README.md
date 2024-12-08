# 🚀 High-Performance Chat Interface for LM Studio – Enhanced Version

![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)

## Overview

This repository hosts a **robust and efficient web-based chat application** that integrates seamlessly with various AI models hosted on **LM Studio**, including Mistral, OpenAI, and Llama. Designed for **high performance**, this enhanced version harnesses **GPU acceleration** and **asynchronous operations** to deliver **faster processing** and **improved responsiveness**.

## Key Features

- **Multiple AI Model Integration:** Effortlessly switch between Mistral, OpenAI, Llama, and others hosted on LM Studio.
- **Real-time Interaction:** Enjoy dynamic, interactive chats through a Gradio-based web interface.
- **Contextual Conversations:** Maintain conversation history for coherent, contextually rich responses.
- **GPU Acceleration:** Speed up computations with GPU support where available.
- **Asynchronous Operations:** Enhance responsiveness by employing async operations for non-blocking calls.
- **Dynamic Token Handling:** Optimize token usage on-the-fly.
- **Internal Reasoning Mechanism:** Break down queries into sub-components and evaluate potential solutions.
- **Plugin/Extension System:** Extend capabilities via custom plugins (e.g., database queries).
- **Improved Context Handling:** Use embeddings to select and inject the most relevant context.
- **Persistent Conversation History:** Save and load conversation states for a continuous experience.

## Table of Contents

1. [Installation](#installation)
   - [Prerequisites](#prerequisites)
   - [Setup Steps](#setup-steps)
2. [Usage](#usage)
3. [Contributing](#contributing)
4. [License](#license)
5. [About](#about)
6. [Topics](#topics)
7. [Resources](#resources)
8. [Screenshots/GIFs](#screenshotsgifs)
9. [Code Structure](#code-structure)
10. [Detailed Documentation](#detailed-documentation)
11. [Error Handling](#error-handling)
12. [Deployment](#deployment)
13. [Conclusion](#conclusion)

## Installation

### Prerequisites

- **Python 3.8 or higher**
- **LM Studio:** A running instance of LM Studio with a compatible model (e.g., Mistral, OpenAI, Llama).

### Setup Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/high-performance-chat-interface.git
   cd high-performance-chat-interface
   ```

2. **Install Required Libraries:**

   ```bash
   pip install gradio httpx torch numpy
   ```

3. **Set Environment Variables:**

   ```bash
   export LMSTUDIO_API_BASE_URL=http://localhost:1234/v1
   ```

   Adjust the URL to match your LM Studio setup.

4. **Optional: Example Plugin Setup**
   For the example database query plugin, ensure you have a SQLite database (`example.db`) with a table (`example_table`).

## Usage

**Run the Application:**

```bash
python main.py
```

**Access the Interface:** Open your browser and go to [http://127.0.0.1:7860/](http://127.0.0.1:7860/). If `share=True`, Gradio deploys a publicly accessible link for 72 hours.

**Interact with Models:**

- Type queries into the chat box.
- Select different models from the dropdown.
- Adjust parameters (temperature, top-p) for fine-tuning.
- Upload a `.txt` file for additional contextual information.

## Contributing

1. **Fork the Repository:**

   ```bash
   git clone https://github.com/yourusername/high-performance-chat-interface.git
   cd high-performance-chat-interface
   ```

2. **Create a New Branch:**

   ```bash
   git checkout -b feature-branch-name
   ```

3. **Make Changes:** Implement your features or fixes.

4. **Submit a Pull Request:** Push changes and open a PR with a detailed description.

## License

This project is licensed under the [MIT License](LICENSE).

## About

The High-Performance Chat Interface is designed to provide seamless, real-time interactions with AI models hosted on LM Studio. It incorporates advanced features like internal reasoning, plugins, and optimized context handling to ensure coherent and meaningful conversations.

## Topics

- machine-learning
- chatbot
- gradio
- conversation-ai
- lm-studio
- high-performance
- asynchronous
- gpu-acceleration
- internal-reasoning
- plugins

## Resources

- **LM Studio Documentation:** [LM Studio](https://example.com)
- **Gradio Documentation:** [Gradio](https://gradio.app)

## Screenshots/GIFs

- Main Chat Window
- Model Selection Dropdown
- Parameter Sliders
- File Upload and Context Display
- Internal Reasoning Steps
- Plugin Output Example (Database Query)

## Code Structure

- `main.py`: Core script handling API interactions, chat logic, and Gradio interface.
- `plugins.py` (optional): Example plugins and instructions for creating custom plugins.
- `utils.py` (optional): Helper functions for embeddings, similarity calculations, etc.

## Detailed Documentation

### I. Configuration and Constants

Defines critical constants and configuration parameters like token limits, embedding constraints, model URLs, GPU availability, and file paths.

### II. Utility Functions

Includes helper functions for:

- **Token Calculations:** Dynamically calculate max tokens for responses.
- **Embedding Generation:** Retrieve embeddings from LM Studio to aid in context selection.
- **Similarity Calculation:** Rank context relevance using cosine similarity.

### III. Internal Reasoning Mechanism

Breaks down user messages into sub-components, generates embeddings for each, and uses these to produce structured reasoning steps before generating the final response.

### IV. API Interaction Handling (Asynchronous Streaming)

Implements asynchronous streaming to deliver partial responses in real-time, improving user experience and responsiveness.

### V. Plugin/Extension System

A modular plugin architecture allows integration with external systems. For example, the database query plugin can fetch data from a local SQLite database to enrich responses.

### VI. Handlers for Chat

Orchestrates:

- File processing and context embedding.
- Conversation history management.
- Running internal reasoning steps and plugins.
- Generating and streaming model responses.

### VII. Gradio Interface Implementation

Builds a user-friendly Gradio UI featuring:

- If `share=True`, Gradio deploys a publicly accessible link for 72 hours.
- Text input for messages.
- File upload for supplemental context.
- Model selection and parameter controls.
- Internal reasoning toggle.
- Persistent conversation state management.

### VIII. Main Execution

Initializes the asynchronous HTTPX client, launches the Gradio interface, and cleans up resources on shutdown.

## Error Handling

Robust error handling ensures stable operation:

- Graceful handling of invalid file uploads.
- User-friendly fallbacks on embedding or API request failures.
- Logging of exceptions for debugging and maintenance.

## Deployment

**Hugging Face Spaces:** Deploy easily for hosting and sharing the interface publicly.

**Docker:** Containerize for consistent deployments:

```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "main.py"]
```

## Conclusion

The High-Performance Chat Interface enhances LM Studio interactions with GPU-accelerated, asynchronous performance. Its internal reasoning, plugin architecture, and optimized context handling deliver rich, contextually aware conversations, making it ideal for a wide range of AI-driven applications.

