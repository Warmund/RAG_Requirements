# RAG Requirements Generator

A Streamlit application for interactive requirements engineering, powered by Retrieval-Augmented Generation (RAG) with large language models and document embeddings. Designed to help project managers and engineers draft precise specification documents using Q&A and contextual retrieval.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Example Prompts](#example-prompts)
- [File Uploads](#file-uploads)
- [Development Notes](#development-notes)

---

## Features

- Interactive Q&A workflow for gathering project requirements.
- Retrieval-augmented generation based on previous documents and user uploads.
- Selection of requirements sections: general info, functional/non-functional requirements, conditions, and more.
- Optimized for project managers and requirements engineers.

---

## Installation

1. **Clone this repository:**
git clone <your-repo-url>
cd <your-project-directory>


2. **Install dependencies:**
Main dependencies:
- streamlit
- llama-index
- sentence-transformers
- (see `requirements.txt` for full details)

3. **LLM & Embedding Model Access:**
- Ensure a local LLM endpoint (e.g., LM Studio) is running.
- Use embedding model `sentence-transformers/all-MiniLM-L6-v2`.

---

## Usage

1. **Start the application:**
2. **Open the browser link** (usually `http://localhost:8501`).

3. **Proceed through the workflow:**
- Fill out the initial Q&A about your project (concept, goals, target group, constraints, etc.).
- Choose sections to generate: Complete Requirements, General Info, Functional, Non-functional, Conditions.
- Optionally upload past requirements documents (PDF/DOCX/TXT) for improved context.
- Use the chat interface for detailed refinement and generation.

---

## Configuration

- **Local LLM setup:** Change endpoint URLs directly in the script if needed.
- **Embedding Model:** By default, uses `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector storage:** Handled automatically in the local `.storage` directory.

---

## Example Prompts

Typical project questions:

- "Describe the concept of the project."
- "What objectives should the project achieve?"
- "Which target group is addressed?"
- "List must-have requirements."
- "What are the constraints (budget, timeline, resources)?"
- "What are the delivery and acceptance conditions?"

---

## File Uploads

- Supports upload of previous requirements documents (PDF, DOCX, TXT formats).
- Uploaded files are integrated into the context for RAG-based responses.

---

## Development Notes

- Built with: Streamlit, llama-index, HuggingFace embeddings, local LLM interface.
- Modular code enables easy extension (add more questions or requirements sections).
- Designed for clarity, reproducibility, and extensibility.

---

Add more technical, licensing, or contribution sections as your project grows.


