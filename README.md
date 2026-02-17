## Ragify – RAG & Retriever Chatbot

Ragify is a focused **Streamlit** chatbot that behaves like a **domain expert for RAG and retrievers**. It reads from a curated RAG knowledge base (the bundled PDF) and then answers as if it were its own expertise.

- **Goal**: help you choose the right retriever/RAG pattern for a given use case and explain RAG concepts clearly.
- **Typical questions**:
  - “What retriever should I use for legal PDFs and emails?”
  - “How is naive RAG different from advanced RAG?”
  - “What is a good setup for multi-domain enterprise search?”
- **Answer style**:
  - Clear, professional, and concise (not overly friendly or chatty).
  - Explicit about assumptions (dataset type, query type, accuracy vs latency, etc.).
  - Uses a fixed structure when recommending retrievers (see below).
- **Implementation**:
  - Frontend: `streamlit` chat UI with a sidebar for the **Gemini API key**
  - Model: **Gemini 2.5 Flash** via `langchain-google-genai`
  - Context: `RAG_Retrievers_Knowledge_Base.pdf` is loaded once and passed as hidden context (no vector DB / RAG pipeline)

When recommending retrievers, the assistant uses this **fixed, structured template**:

```text
When recommending a retriever:

Recommended Retriever: <retriever name>
Reason: <why this retriever is best for the project>
Secondary Options: 
- <Option 1> (pros/cons)
- <Option 2> (pros/cons)
Implementation Notes: <pre-processing or configuration tips>
```

The assistant is instructed **not** to say that it is reading from a PDF or to cite sections/pages; it should sound like a knowledgeable engineer drawing on experience.

---

## 1. Requirements

- **Python**: 3.9+ (3.10–3.13 recommended)
- A valid **Gemini / Google AI Studio API key** with access to `gemini-2.5-flash`

Python dependencies are listed in `requirements.txt`:

- `streamlit`
- `langchain`
- `langchain-community`
- `langchain-google-genai`
- `langgraph`
- `python-dotenv`
- `pypdf`

---

## 2. Setup

From the project root (`ragify`):

```bash
cd /Users/app/Desktop/work/ragify

# (optional but recommended) create a virtualenv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -U pip
pip install -r requirements.txt
```

You can provide the Gemini API key in two ways:

- **Option A (sidebar input)** – paste it at runtime (not stored on disk)
- **Option B (`.env` file)** – create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

The app will also accept `GEMINI_API_KEY` if you prefer that name.

Ensure that the knowledge base PDF is present:

- `RAG_Retrievers_Knowledge_Base.pdf` (already in the project root)

---

## 3. Running the App

From the project root:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`).

In the **sidebar**:

1. Paste your **Gemini API key** (unless you set it via `.env`)
2. Use **Clear chat** to reset the conversation if needed

Now you can ask questions like:

- “What retriever is best for a small FAQ chatbot?”
- “I want an enterprise search over legal PDFs and emails. What should I use?”
- “Explain naive vs advanced RAG.”

The assistant will:

- Use the PDF content as hidden context
- Answer in a clear, professional tone
- Use the structured retriever recommendation template when applicable

---

## 4. Code Structure

- `app.py`  
  Main Streamlit application:
  - Loads the PDF using `PyPDFLoader`
  - Builds LangChain messages (system prompt + PDF context + chat history)
  - Calls `ChatGoogleGenerativeAI` with `gemini-2.5-flash`
  - Manages chat history in `st.session_state`
  - Shows a loader while generating and temporarily disables the chat input

- `requirements.txt`  
  Python dependencies.

---

## 5. Customisation Ideas

- Swap `gemini-2.5-flash` with another Gemini model (e.g. a Pro variant).
- Adjust `temperature` in `_generate_assistant_reply`.
- Replace the PDF with another domain-specific knowledge base (e.g. your own RAG guide).
- Add basic analytics (e.g. logging queries and recommendations) behind a feature flag.
