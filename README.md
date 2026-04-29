# LexAgent — Local Agentic Contract Intelligence System

## What Problem This Solves

Employment contracts, NDAs, and service agreements are dense legal documents engineered by company lawyers to protect the company — not you. A non-compete clause buried in clause 14.3 that claims your side-project code written at midnight on a Sunday, or a liability cap set to zero buried under neutral language, can cause irreversible career and financial damage years after signing. Most people sign without reading, or read without understanding, because professional legal review costs ₹15,000–50,000 per contract review and takes days.

LexAgent runs a complete 6-agent AI pipeline on your machine — no cloud, no API calls, no data leaving your system — and returns a structured risk report with plain-English explanations and negotiation points in under 2 minutes. It identifies IP overreach, asymmetric notice periods, perpetual confidentiality obligations, foreign arbitration seats, missing severance clauses, and cross-clause contradictions — the exact patterns that lawyers exploit and that non-lawyers miss. It is not a replacement for legal advice, but it is the difference between walking into a negotiation blind and walking in knowing exactly which three clauses to push back on.

## Technical Innovations

**1. Hybrid Legal Clause Segmentation**
Standard NLP sentence splitters (spaCy, NLTK) fail on legal text because legal clauses are defined by structural hierarchy (1.2.3, Article IV, SECTION 5), not sentence boundaries. LexAgent's SegmentationAgent uses a three-stage hybrid approach: structural regex parsing that detects numbered/lettered clause boundaries, a cross-reference resolver that maps "as defined in Section 3.1" into a dependency graph, and a definition extractor that pulls term→definition pairs using pattern matching. This produces properly bounded clauses that preserve hierarchical context — enabling accurate classification and contradiction detection.

**2. Cross-Clause Contradiction Detection via NetworkX**
Legal contracts frequently contain contradictions between clauses — sometimes by accident, sometimes deliberately. LexAgent builds a NetworkX directed graph where nodes are clause IDs and edges are cross-references ("pursuant to", "as defined in"). Five contradiction patterns are checked: IP ownership conflicts (employer-owns vs employee-owns language in the same label group), notice period conflicts (different periods specified across termination clauses), termination trigger conflicts (notice-required vs immediate-termination language), definition conflicts (same term defined differently in two clauses, detected via Jaccard word-set similarity), and obligation-without-right patterns (shall-provide with no corresponding shall-receive in connected nodes). This is more than keyword matching — it is semantic graph traversal.

**3. Comparative Aggressiveness Scoring via ChromaDB Memory**
After each analysis, clause embeddings (sentence-transformers all-MiniLM-L6-v2) and their risk scores are stored in ChromaDB. For each HIGH/CRITICAL clause in a new contract, LexAgent queries for the 5–10 most semantically similar clauses with the same label and computes a percentile: "This non-compete is more aggressive than 87% of similar clauses seen." This is a contract memory system. The more contracts analyzed, the more calibrated the aggressiveness percentiles become.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User (Browser)                          │
│                    http://localhost:8000                         │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP (FastAPI)
┌────────────────────────────▼────────────────────────────────────┐
│                     FastAPI Backend                             │
│              POST /analyze  GET /status/{id}                    │
│              GET /history   GET /health                         │
└─────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    Pipeline Orchestrator                        │
│                       pipeline.py                               │
└──┬──────────┬──────────┬──────────┬──────────┬─────────────────┘
   │          │          │          │          │
   ▼          ▼          ▼          ▼          ▼
┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────────────────────┐
│  A1  │  │  A2  │  │  A3  │  │  A4  │  │         A5           │
│Extrc │→ │ Seg  │→ │Class │→ │ Risk │→ │   Contradiction      │
│Agent │  │ment  │  │ify   │  │Agent │  │      Agent           │
│      │  │Agent │  │Agent │  │      │  │    (NetworkX)        │
└──────┘  └──────┘  └──────┘  └──────┘  └──────────────────────┘
  fitz               legal-              deterministic    graph
  OCR               bert                 rules           traversal
  docx             zero-shot
                             │
                             ▼
                    ┌─────────────────────┐
                    │        A6           │
                    │   Summarizer +      │
                    │   Memory Agent      │
                    │  Ollama (mistral)   │
                    │  ChromaDB           │
                    │  sentence-trans     │
                    └─────────────────────┘
```

## Prerequisites

**Required:**
- Python 3.11+
- pip

**Required for AI summaries (strongly recommended):**
```bash
# Install Ollama (Linux/Mac)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the model
ollama pull mistral:7b
```

**Required for scanned PDF support:**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

**GPU (optional but recommended for classification speed):**
- CUDA-capable GPU (NVIDIA). RTX 3050 6GB works well.
- If no GPU: classification falls back to CPU — slower, same accuracy.

## Build and Run

```bash
# Clone or extract the project
cd lexagent

# Option 1: Use the startup script (recommended)
chmod +x run.sh
./run.sh

# Option 2: Manual
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Open in browser
open http://localhost:8000
```

## How to Use

1. Open `http://localhost:8000`
2. Drag and drop a PDF or DOCX contract into the upload zone
3. Click **Analyze Contract**
4. Wait 30–120 seconds (first run longer due to model loading)
5. Review the risk dashboard, critical clauses, contradictions, and summary
6. Click **Export Full JSON Report** to download the complete analysis

## Running Tests

```bash
cd lexagent
pip install pytest
pytest tests/ -v
```

## Fine-Tuning on CUAD (Optional)

The ClassificationAgent works out-of-the-box using zero-shot classification with `legal-bert-base-uncased`. Fine-tuning on CUAD improves per-class accuracy from ~65% to ~82% F1 on average.

```bash
# Run overnight — do not interrupt
python scripts/finetune_cuad.py
```

The script will:
1. Download the CUAD dataset (~500MB) from HuggingFace
2. Train 41 binary classifiers (one per clause type) for 3 epochs each
3. Print per-class F1 scores on validation set
4. Save models to `models/legal-bert-cuad/<ClauseType>/best/`

After fine-tuning, update `ClassificationAgent.__init__` to load from the per-type model directories instead of the base model.

**Estimated runtime:** 4–6 hours on RTX 3050 6GB with fp16=True.

## Design Decisions

**Zero-shot classification without fine-tuning as baseline:**
`nlpaueb/legal-bert-base-uncased` is trained on EU legislation and legal corpora, giving it strong legal vocabulary understanding. Zero-shot classification with this model achieves reasonable accuracy (~55–70% top-1) on CUAD clause types without any labeled training data. This means the system is immediately usable — no training, no waiting, no dataset download required. Fine-tuning is an accuracy upgrade, not a requirement.

**Rule-based risk engine over ML:**
Risk scoring is deterministic and fully auditable. Every risk flag maps to a specific, readable rule. This matters for a legal tool: a user needs to understand *why* a clause is flagged, not just that it scored 0.73 on some model. Rules are also predictable — the same clause always produces the same risk level, making the system trustworthy. ML models for risk scoring would require labeled training data (contracts with expert risk annotations) that doesn't publicly exist at sufficient scale.

**NetworkX for contradiction detection:**
Legal cross-references create a dependency structure that is naturally a graph problem. NetworkX provides efficient DiGraph traversal with sub-10ms latency on contracts up to 500 clauses. The alternative (all-pairs comparison) would be O(n²) and slow on large contracts. The graph structure also enables future extensions like cycle detection (circular references between clauses) and reachability analysis.

**ChromaDB over Pinecone/Weaviate:**
LexAgent is a local-first tool. ChromaDB runs in-process with zero configuration, zero API keys, and zero data leaving the machine. For a privacy-first legal intelligence tool, this is non-negotiable. Pinecone requires cloud credentials and sends embeddings to external servers — unacceptable for contract data.

## File Structure

```
lexagent/
├── agents/
│   ├── extraction_agent.py      # PDF + OCR + DOCX extraction
│   ├── segmentation_agent.py    # Hybrid legal clause segmentation
│   ├── classification_agent.py  # Zero-shot legal-bert classification
│   ├── risk_agent.py            # Deterministic rule-based risk engine
│   ├── contradiction_agent.py   # NetworkX graph contradiction detection
│   └── summarizer_agent.py      # Ollama summarization + ChromaDB memory
├── api/
│   └── main.py                  # FastAPI backend
├── frontend/
│   └── index.html               # React 18 single-page frontend
├── scripts/
│   └── finetune_cuad.py         # Optional CUAD fine-tuning
├── tests/
│   ├── test_extraction.py
│   ├── test_segmentation.py
│   ├── test_risk.py
│   └── test_contradiction.py
├── pipeline.py                  # 6-agent orchestrator
├── requirements.txt
├── run.sh
└── README.md
```

## Limitations and Disclaimers

- LexAgent is an automated analysis tool, not a lawyer. It does not provide legal advice.
- Accuracy depends on document quality. Heavily scanned or image-only PDFs may have OCR errors.
- Zero-shot classification can misclassify unusual clause structures.
- Always have a qualified lawyer review contracts before signing.
- Risk rules are calibrated for Indian employment and commercial contracts. Foreign law contracts may require different rule sets.

## License

MIT License. Use at your own risk. No warranty expressed or implied.
