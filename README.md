
<div>
<img src="https://i0.wp.com/thelettertwo.com/wp-content/uploads/2024/02/ai-economy-rag-explained-1600x914-1.jpg?w=1600&ssl=1" width=1100>
<div>


<div align="center" style="font-family: 'Times New Roman', Times, serif; color: #3498db;">
  <h1 style="font-size: 10em; margin-bottom: 180px;">
    Projet Machine Learning :<br>
    "Prompting(RAG), RHLF"
  </h1>
</div>

<div align="right" style="font-family: 'Times New Roman', Times, serif; 
                          color: #8e44ad;
                          font-style: italic;
                          font-weight: bold;
                          margin-right: 15%;
                          margin-top: 10px;
                          font-size: 20em;">
   Realised by : SABI Houssame & CHADADI Zakaria
</div>

&nbsp;

---

<!-- Le reste de votre contenu ici -->





## Introduction

This project explores three key techniques in modern AI:

- **Prompting**: the art of crafting effective instructions (prompts) to guide language models.
- **RAG (Retrieval-Augmented Generation)**: combines document retrieval with text generation to improve the relevance and accuracy of responses.
- **RHLF (Reinforcement Learning from Human Feedback)**: a training approach that uses human feedback to align models with desired behavior, improving safety and usefulness.

These methods are essential for building advanced and trustworthy AI systems.

---


## 📚 Table of Contents

- [🎯 Project Goals](#-project-goals)
- [⚙️ Package Installation](#%EF%B8%8F-package-installation)
  - [LlamaIndex](#llamaindex-)
  - [OpenAI](#openai-)
  - [Streamlit](#streamlit-)
- [📦 Example Code Explanation](#-example-code-explanation)
  - [Download and Save Educational PDFs from OpenStax](#%EF%B8%8F-download-and-save-educational-pdfs-from-openstax)
  - [Set Up a DeepSeek Chat Model via OpenRouter API with LlamaIndex](#%EF%B8%8F-set-up-a-deepseek-chat-model-via-openrouter-api-with-llamaindex)
  - [Initialize Qwen2-7B Embedding Model via HuggingFace in LlamaIndex](#%EF%B8%8F-initialize-qwen2-7b-embedding-model-via-huggingface-in-llamaindex)
  - [Set Up a Multilingual Embedding Model Qwen2-7B in LlamaIndex](#%EF%B8%8F-set-up-a-multilingual-embedding-model-qwen2-7b-in-llamaindex)
  - [Create a VectorStoreIndex from Files in educational_docs Folder](#%EF%B8%8F-create-a-vectorstoreindex-from-files-in-educational_docs-folder)
  - [Load and Index Local PDF Documents Using LlamaIndex](#%EF%B8%8F-load-and-index-local-pdf-documents-using-llamaindex)
  - [Assign Custom LLM to LlamaIndex Global Configuration](#%EF%B8%8F-assign-custom-llm-to-llamaindex-global-configuration)
  - [Ask a Question Over Indexed Documents with LlamaIndex](#%EF%B8%8F-ask-a-question-over-indexed-documents-with-llamaindex)
  - [Interactive Educational QA Web App Using Streamlit and LlamaIndex](#%EF%B8%8F-interactive-educational-qa-web-app-using-streamlit-and-llamaindex)
  - [Run the Educational QA App with Streamlit](#%EF%B8%8F-run-the-educational-qa-app-with-streamlit)


---

## 🎯 Project Goals

- Understand the role of prompting in interacting with LLMs (Large Language Models).
- Implement a simple Retrieval-Augmented Generation (RAG) pipeline.
- Learn the basics of Reinforcement Learning from Human Feedback (RHLF).

---

## ⚙️ Package Installation

 ### LlamaIndex: <br>
 A data framework for building LLM (Large Language Model) applications by indexing and querying data with ease.
  ```bash
  %pip install llama-index-llms-openrouter
  !pip install llama-index
  %pip install llama-index-embeddings-huggingface
  %pip install llama-index-embeddings-instructor
```
### OpenAI: <br>
An AI research and deployment company offering advanced language models like GPT for natural language understanding and generation.
### Streamlit: <br>
A Python library used to quickly build and share interactive web apps for data science and machine learning projects.
  ```
  pip install requests pymupdf streamlit
  ```

---

 ## 📦 Example Code Explanation
 ### ➡️ Download and Save Educational PDFs from OpenStax
 <div>
 <img src="https://github.com/user-attachments/assets/fcd36e56-928a-4b69-9d5b-d8f266e5ef77" width=1000>
 </div>
 <br>
This Python script imports the os and requests modules, creates a folder named educational_docs (if it doesn’t already exist) to store PDFs, and defines a dictionary called pdf_sources that maps file names to URLs of OpenStax documents (currently commented out, so inactive). For each entry in the dictionary, the script attempts to download the PDF via an HTTP request, checks if the response is valid and contains a proper PDF header (%PDF), then saves the file locally if successful. <br>
If the download fails or the content is not a valid PDF, it prints an error message. Finally, it confirms the outcome of each operation through console messages.
<br>
<br>
⚠️ Key Note: URLs are commented out → uncomment them to enable downloads.

<br>
<br>
<br>

### ➡️ Set Up a DeepSeek Chat Model via OpenRouter API with LlamaIndex
 <div>
 <img src="https://github.com/user-attachments/assets/4aa6ce91-df17-40fa-ae69-45f8b2490929" width=1000>
 </div>
 <br>

In this part code configures a language model (LLM) via OpenRouter (a service providing access to multiple AI models) using the llama_index library. It imports the required classes (OpenRouter for integration, ChatMessage—unused here), initializes an LLM client with a hardcoded API key (⚠️ security risk), and specifies the free deepseek/deepseek-chat:free model, limiting responses to 256 tokens and setting a 4096-token context window. Commented lines suggest using environment variables for secure API key storage (best practice ignored here).<br>
This snippet acts as a foundational setup for interacting with LLMs via OpenRouter but requires adjustments (security fixes, query logic) for real-world use.
<br>
<br>
<br>

### ➡️ Initialize Qwen2-7B Embedding Model via HuggingFace in LlamaIndex
 <div>
 <img src="https://github.com/user-attachments/assets/a8595f88-7ce2-4813-a959-3d5e5ac600ee" width=1000>
 </div>
 <br>
It imports the HuggingFaceEmbedding class to generate vector representations of text. Two options are shown:

1. Commented-out line (`embed_model = HuggingFaceEmbedding()`) :
   - would load the default `BAAI/bge-small-en model`, a lightweight and efficient embedding model optimized for English.

2. Active line :
   - explicitly loads the Alibaba-NLP/gte-Qwen2-7B-instruct model, a far larger and more powerful model (7 billion parameters) designed for advanced multilingual or complex tasks.

#### Key implications:
- Using Alibaba-NLP/gte-Qwen2-7B-instruct implies a need for high-quality embeddings (`deep contextual understanding, multilingual support`) but requires significant computational resources (GPU/RAM).

- The commented option (`bge-small-en`) offers a lightweight alternative for simpler tasks or resource-constrained environments.

- No additional parameters (`device, pooling...`) are specified, so default settings of HuggingFaceEmbedding are used.

<br>
<br>
<br>

### ➡️ Set Up a Multilingual Embedding Model (Qwen2-7B) in LlamaIndex
 <div>
 <img src="https://github.com/user-attachments/assets/e27e33bd-8ae0-4d9f-b9d4-c83f0796b6f5" width=1000>
 </div>
 <br>

- Global Integration: All LlamaIndex components (`indexers, query engines...`) will automatically use this model to generate embeddings.

- Advanced Model: `The gte-Qwen2-7B-instruct` model is designed for complex tasks (e.g., contextual understanding, multilingual Q&A) but requires significant GPU/RAM resources.

- Simple Customization: A single line of code defines the default behavior for the entire library.

Typical Use Cases:

- Building a multilingual semantic search system.

- Generating high-quality embeddings for classification or clustering.

- Experimenting with large-scale models for specialized tasks.

⚠️ Notes:

No additional parameters (`device to force GPU usage`) are specified : relies on HuggingFaceEmbedding default settings.

The chosen model is resource-intensive : verify hardware availability before execution.

<br>
<br>
<br>

### ➡️ Create a VectorStoreIndex from Files in ‘educational_docs’ Folder
 <div>
 <img src="https://github.com/user-attachments/assets/c170c2c3-68d3-45a4-a669-fe670e3c4d4b" width=1000>
 </div>
 <br>

Loads documents from the educational_docs directory using SimpleDirectoryReader.<br>
Constructs a vector index (VectorStoreIndex) from the documents, automatically converting text into embeddings with LlamaIndex's default model.<br>
The minimalist setup leverages predefined configurations, making it ideal for prototyping, but lacks customization (`embedding models, storage settings`) for advanced use cases.
 
<br>
<br>
<br>

### ➡️ Load and Index Local PDF Documents Using LlamaIndex
 <div>
 <img src="https://github.com/user-attachments/assets/9cd18f56-cf95-4212-b67d-79d8d293749b" width=1000>
 </div>
 <br>

Loads documents from the `educational_docs` folder using `SimpleDirectoryReader`.<br>
Builds a vector index (`VectorStoreIndex`) from the documents, leveraging LlamaIndex's default embedding model and storage settings.<br>
The code is a quick-start setup for basic search/retrieval tasks but requires customization (`embedding choice, storage backend`) for production-grade use.<br>

<br>
<br>
<br>
      
### ➡️ Assign Custom LLM to LlamaIndex Global Configuration
 <div>
 <img src="https://github.com/user-attachments/assets/cb7a8447-55cf-40a8-a411-f515b93754c1" width=1000>
 </div>
 <br>

 By assigning `Settings.llm = llm`, you ensure that components like query engines, chatbots, or retrievers automatically use your pre-initialized `llm` (`OpenAI, OpenRouter, or another provider`) for text generation. This centralizes model management, simplifies code, and avoids repetitive parameter passing. Ensure the `llm` object is properly initialized (API keys, model choice) before this assignment.

<br>
<br>
<br>
      
### ➡️ Ask a Question Over Indexed Documents with LlamaIndex
 <div>
 <img src="https://github.com/user-attachments/assets/f4d480de-f4f9-4324-a69e-0298528ecf4b" width=1000>
 </div>
 <br>

The line `query_engine = index.as_query_engine()` initializes a query engine using a vector index (assumed to contain documents like educational PDFs).<br>
The command `response = query_engine.query("who is houssame sabi?")` performs a semantic search across the indexed documents, retrieves relevant information, and uses a language model (LLM) to synthesize a coherent answer.<br>
Finally, `print(response)` displays the result, which could be a concise summary or extracted details about "houssame sabi" based on the indexed content. This transforms raw documents into an interactive question-answering system, assuming the indexed data contains information about the queried subject.

<br>
<br>
<br>
    
### ➡️ Interactive Educational Q&A Web App Using Streamlit and LlamaIndex
 <div>
 <img src="https://github.com/user-attachments/assets/3991ac8e-026a-456e-9586-2ae060ca44d0" width=1000>
 </div>

 
1. **Interface Setup**:  
   - `st.title("Educational Assistant")` : Sets the app title.  
   - `st.text_input("Ask a question:")` : Creates a text box for users to input questions.  

2. **Query Processing**:  
   - When a question is submitted (`if user_question:`), it uses a pre-initialized `query_engine` (not shown here) to generate an AI-powered answer via `query_engine.query(user_question)`.  
   - `st.write(response.response)` : Displays the answer to the user.  

3. **Feedback System**:  
   - `st.radio(...)` : Asks users to rate the answer (👍 Good / 👎 Bad).  
   - Logs the question, answer, and feedback to `feedback_log.csv` for later analysis.  

<br>
<br>
<br>

### ➡️ Run the Q&A App with Streamlit
 <div>
 <img src="https://github.com/user-attachments/assets/4f36966a-e2e7-4f6e-acf2-4869b585a1a1"  width=1000>
 </div>


- **`!`** (if used in Jupyter/Colab): Executes a shell command directly from the notebook environment.  
- **`streamlit run app.py`**: Starts the Streamlit server and opens the app in your default browser.  

**Expected behavior**:  
1. A browser tab opens with the "Educational Assistant" interface.  
2. Users ask questions → AI generates answers using the pre-configured `query_engine`.  
3. Feedback is logged to `feedback_log.csv` for later analysis.  

**Console output example**:  
```  
Local URL: http://localhost:8501  
Network URL: http://192.168.1.X:8501  
```  
**To stop**: Press `Ctrl+C` in the terminal.  

**Requirements**:  
- Streamlit must be installed (`pip install streamlit`).  
- `app.py` must exist and be functional (with `query_engine` properly initialized).

