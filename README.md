<div align="center">

# 🌐 Neural Query & Proof Matrix
### High-Speed Multilingual RAG Telemetry & Trust Calibration Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blue?style=for-the-badge&logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Deployment Ready](https://img.shields.io/badge/Railway-Ready-000000?style=for-the-badge&logo=railway&logoColor=white)](https://railway.app/)

*A state-of-the-art Retrieval-Augmented Generation (RAG) laboratory console featuring sub-200ms latency distribution tracking, strict script isolation, and grounding verification across English, Hindi, and Urdu.*

</div>

---

## ⚡ Core Features

- **Sub-200ms Telemetry Pipeline:** Optimized vector embedding, indexing, guardrails, and generation phases tracked live via a millisecond distribution oscilloscope.
- **Strict Multilingual Script Isolation:** Employs zero-tolerance Unicode boundary filtering to ensure precise script matching for **English (Latin)**, **Hindi (`hin_Deva`)**, and **Urdu (`urd_Arab`)**.
- **Trust Calibration Array:** Automated verification metrics panel tracking Grounding Pass scores, Hallucination checks, and Safety thresholds in real time.
- **Interactive Control Console:** Full-featured laboratory layout equipped with query presets, voice input toggles, dynamic proof binding, and instant report generation.

---

## 🛠️ System Architecture

```text
 ┌─────────────────────────────────────────────────────────┐
 |                   MOD-01 // QUERY CONSOLE               |
 |       [ Language Selector: EN | HI | UR ]               |
 └────────────────────────────┬────────────────────────────┘
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 |              MOD-03 // TELEMETRY WORKSPACE              |
 |   [EMB: ~20ms] -> [SEARCH: ~0.1ms] -> [GEN: ~0.02ms]    |
 └────────────────────────────┬────────────────────────────┘
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 |            MOD-02A / 02B // PROOF & CALIBRATION         |
 |        FAISS Evidence Nodes & Trust Calibration Array   |
 └─────────────────────────────────────────────────────────┘

 multilingual-rag-matrix/
├── app/
│   ├── server.py             # FastAPI backend & strict script routing
│   ├── retriever.py          # FAISS high-speed vector search engine
│   └── generator.py          # Multilingual generation pipeline
├── data/                     # Vector indices & encoded datasets
├── front.html                # Interactive laboratory telemetry UI
├── .gitignore                # Security & binary exclusion rules
└── requirements.py / pip     # Project dependencies

#1.Clone the repository:
git clone https://github.com/0251cys075/multilingual-rag-matrix.git
cd multilingual-rag-matrix

#2.Create and activate a virtual environment:
python -m venv venv
# Windows:
.\venv\Scripts\Activate
# Mac/Linux:
source venv/bin/activate

#3.Install dependencies:
pip install -r requirements.txt

#4.Run the FastAPI server:
uvicorn app.server:app --reload

#5.Open your browser and navigate to http://127.0.0.1:8000 to launch the console!

