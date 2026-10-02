# System Architecture & Technical Specifications

The Fiqh Co-Pilot Engine is an enterprise-grade Multi-Agent Retrieval-Augmented Generation (RAG) platform tailored for Islamic Jurisprudential research (Hanafi School).

## Workflow Pipeline
1. Linguistic Router Agent: Normalizes vernacular phrasing into Classical Fiqh terms.
2. Dense Semantic Retrieval Engine: Searches Qdrant vector index for top 5-10 classical passages.
3. Juristic Reasoning Agent: Ensures 100% grounding on classical texts and mitigates hallucination.
4. Business Process Automation: Generates formal, archival-quality legal research PDFs.
