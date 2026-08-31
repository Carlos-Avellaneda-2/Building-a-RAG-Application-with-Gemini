# RAG on Prioritization of Access to Specialist Consultation in Colombia (MGTE)

## Carlos Andres Avellaneda Franco

A RAG (Retrieval-Augmented Generation) application built with **Gemini + LangChain + Chroma**, developed as the homework for the RAG workshop, extending the patterns covered there to a real use case: the semester project *"Intelligent Prioritization of Access to Specialist Consultation — an Enterprise Architecture Problem in the Colombian Health System"* (Carlos Andres Avellaneda Franco).

## 1. Objective and selected use case

The semester project investigates how to prioritize, in a transparent and clinically defensible way, access to specialist medical consultations in Colombia, in a context of growing scarcity of supply relative to demand, and evaluates whether — and under what conditions — artificial intelligence adds value over prioritization based on explicit rules.

This RAG application allows querying, in natural language, the evidence gathered for that project: the new Colombian regulatory framework (Waiting Time Management Model, MGTE), quantitative evidence of the access problem, and the international literature on AI applied to medical-referral prioritization/triage.

Target questions for the application:
- What is the MGTE, who issued it, and what implementation phases does it have?
- Which specialties and maximum waiting times does Circular 038 of 2025 prioritize?
- What quantitative evidence exists about the specialist-consultation access problem in Colombia?
- How well do machine learning models perform at prioritizing medical referrals, according to the international literature?
- Questions outside the scope of the sources (to verify that the system does not hallucinate).

## 2. Description and origin of the document collection

**5 public documents** were used, taken directly from the reference list of the semester project's article (`Articulo_Proyecto_Triage.docx`), downloaded and saved as plain text in `data/` along with their metadata (title, URL, source identifier):

| File | Title | URL | Type |
|---|---|---|---|
| `doc1_circular_038_cups_tiempos.txt` | Circular Externa 038 of 2025 | consultorsalud.com/tiempos-de-espera-para-citas-con-especialistas | Regulatory / operational |
| `doc2_resolucion_2117_mgte.txt` | Resolución 2117 of 2025 (MGTE) | consultorsalud.com/modelo-tiempos-espera-salud-colombia-2025 | Regulatory |
| `doc3_ai_referral_triage_cpc_queensland.txt` | Abdel-Hafez et al. (2023), *Frontiers in Digital Health* (CC-BY) | pmc.ncbi.nlm.nih.gov/articles/PMC10642163 | Academic / international |
| `doc4_radiografia_acceso_salud_colombia.txt` | "Radiografía del acceso a la salud en Colombia" | consultorsalud.com/radiografia-del-acceso-a-la-salud-en-colombia | Problem evidence (2024 report) |
| `doc5_mgte_fases_ambitojuridico.txt` | "Minsalud fija tiempos máximos de espera..." | ambitojuridico.com/.../minsalud-fija-tiempos-maximos-de-espera... | Regulatory (independent legal source) |

> **Note on source language:** these are official Colombian regulations and Colombian news/legal outlets, so their original body text is in **Spanish** — this is intentional, since they are the actual primary sources of the semester project. Only this README, the notebook, and its comments have been translated to English. Gemini's embeddings and chat model handle Spanish content together with English instructions/questions natively, so the pipeline works correctly across languages without needing to translate the source documents.

**Design decision — sources cached locally instead of live scraping:** instead of having the notebook download the web pages every time it runs (a live `WebBaseLoader`), the content was downloaded once and saved as `.txt` with a metadata header. This was done because several of the sites (e.g., Ámbito Jurídico) have partial paywalls and block automated scraping, and because it guarantees the notebook is **reproducible** without depending on the future availability of those pages. The cost of this decision is that the notebook does not re-query the live source; this is explicitly documented as a limitation in Section 9.

All sources are public and openly accessible. No confidential, personal, or restricted enterprise information was used, in line with the security notes in the assignment.

## 3. Architecture

```mermaid
flowchart LR
    A[Public sources\n(5 .txt documents with metadata)] --> B[Load as\nLangChain Document]
    B --> C[Chunking\nRecursiveCharacterTextSplitter\nchunk_size=1000, overlap=150]
    C --> D[Embeddings\nGemini gemini-embedding-001]
    D --> E[(Chroma\nlocal vector store)]
    E --> F{Retrieval\ntop_k=4}
    F --> G[Prompt with context\n+ user question]
    G --> H[Gemini Flash\n(chat / generation)]
    H --> I[Answer + cited sources]
```

Two consumption architectures over the same Chroma index:
- **Two-step RAG chain**: always retrieves `top_k=4` chunks and then generates the answer.
- **RAG agent**: the same retriever exposed as a tool (`create_retriever_tool`); a ReAct agent (`langgraph.prebuilt.create_react_agent`) decides whether it needs to call it before answering.

## 4. Installation and execution

```bash
git clone <your-repository-url>
cd <repository>

python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and paste your free API key from https://aistudio.google.com/apikey

jupyter notebook notebooks/rag_application.ipynb
```

Run the notebook cells in order. The first time you run the embeddings/Chroma section, a `chroma_db/` folder will be created (excluded from the repository via `.gitignore`, since it is regenerated automatically).

## 5. Required environment variables

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Free API key from Google AI Studio (Gemini Developer API, free tier) |

## 6. Gemini models used

- **Chat / generation:** detected **automatically** when running Section 1 of the notebook, since Google retires and renames free-tier Flash models very frequently (for example, `gemini-2.5-flash` stopped being available to new users shortly after this notebook was written, and the API recommended migrating to `gemini-3.6-flash`). The notebook queries `client.models.list()` with your own API key and picks the first available model from a preference list (`gemini-3.6-flash` → `gemini-3.5-flash` → `gemini-3-flash` → `gemini-2.5-flash` → `gemini-2.5-flash-lite` → any other available "flash" model). The exact detected name is printed when running that cell — **write down the one you got here**: `_____________`.
- **Embeddings:** `models/gemini-embedding-001` (fixed by the assignment). If this identifier also becomes unavailable in the future, the same `client.models.list()` mechanism used for chat can be adapted by filtering for "embedding"-type models.

## 7. Key design decisions

- **Chunking (`chunk_size=1000`, `chunk_overlap=150`):** the documents are regulatory/news articles plus a condensed academic summary with paragraphs of roughly 150-400 characters; 1000 characters groups several related paragraphs without mixing distinct sections, and the overlap avoids cutting lists or ideas at the chunk boundary.
- **`top_k=4`:** with only 5 relatively short source documents, 4 chunks cover most questions without diluting the context with low-relevance chunks.
- **"Evidence only, state the gaps" prompt:** the system prompt (in both the chain and the agent) explicitly instructs the model not to answer beyond the retrieved context and to state what information is missing when the context is insufficient — a critical requirement in a health domain.
- **Local caching of sources vs. live scraping:** see Section 2.
- **Source citation:** every chunk keeps `source_id` and `url` in its metadata, and the prompt requires listing the sources used at the end of each answer, so the application is auditable (consistent with one of the semester project's core quality attributes: explainability).

## 8. Evaluation results

Three questions were tested (see the notebook, Section 9, for full detail):

| Question | Retrieved source | Grounded? | Observation |
|---|---|---|---|
| What are the three MGTE phases and how long does each last? | doc1 / doc2 / doc5 | Yes | The fact is reinforced by 3 independent sources → very robust retrieval |
| How accurate is an ML model at prioritizing high-severity cases? | doc3 | Partial | Only evidence from one study (Queensland/ENT, 53.8% agreement); the model must clarify it is not generalizable to Colombia |
| Maximum waiting time for hip surgery in Colombia? | No relevant source | Yes (correct refusal) | The system explicitly states it has no evidence, instead of making up a figure |

*(The full table with the model's complete answers is generated by running Section 9 of the notebook, since the exact answers depend on the live run against the Gemini API.)*

**Case where retrieval worked well:** the question about the MGTE phases, because the same fact appears in 3 different documents with different wording.

**Failure / limitation observed:** with 1000-character chunks, figures that appear in large tables (such as the different agreement levels per method in the Queensland paper) can end up split across more than one chunk and not always be retrieved together with `top_k=4`.

**Possible improvement:** differentiated chunking for tabular content, and a re-ranking step before passing chunks to the LLM.

## 9. Comparison: RAG chain vs. RAG agent

Both architectures were run on the same question (notebook, Section 8). For this use case — a small knowledge base that is almost always relevant to the domain questions — the **two-step chain** is the simpler, more predictable option: a single retrieval call plus a single generation call, without the added cost or latency of an agent's reasoning loop. The **agent** adds value mainly if the system must also handle questions outside the scope of the knowledge base (where it can decide not to retrieve), a scenario that is not the focus of this application.

## 10. Limitations and possible improvements

- The collection of only 5 documents is representative but small; a production system for this project would need to directly integrate the full technical annexes of the MGTE and, eventually, real or synthetic referral data (see the "Next Steps" section of the project's article).
- Sources were cached locally (see Section 2); a production pipeline should include a mechanism for periodically refreshing and version-controlling the source content.
- No re-ranking or automated quantitative evaluation (e.g., RAGAS) was implemented; grounding evaluation in this homework is manual/qualitative, following the requested scope.
- Gemini's free chat model changes name frequently (see Section 6); the notebook isolates that dependency into a single auto-detected variable to make it easier to keep working over time.

## Repository structure

```
repository/
├── notebooks/
│   └── rag_application.ipynb
├── data/
│   ├── doc1_circular_038_cups_tiempos.txt
│   ├── doc2_resolucion_2117_mgte.txt
│   ├── doc3_ai_referral_triage_cpc_queensland.txt
│   ├── doc4_radiografia_acceso_salud_colombia.txt
│   └── doc5_mgte_fases_ambitojuridico.txt
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
```
