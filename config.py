"""
Configuration for the agentic RAG HW.

Lifted directly from the notebook's `defaultConfig` dict (deduplicated).
Settings here drive document processing, the retriever, and tool behavior.
Don't bury settings in code — keep them in this dict so we can tweak without
hunting through files.
"""

CONFIG = {
    # Document processing
    "chunkSize": 500,
    "chunkOverlap": 50,
    # SEC requires a User-Agent header on requests; default urllib UA gets blocked.
    "userAgentHeader": "YourCompany-ResearchBot/1.0 (your@email.com)",

    # Embeddings
    "embeddingModelName": "text-embedding-3-small",

    # Vector store + retrieval
    "numRetrievedDocuments": 12,    # FAISS top-k before reranking

    # Reranker
    "rerankerModel": "ms-marco-TinyBERT-L-2-v2",
    "numRerankedDocuments": 5,      # top-n after reranking

    # Companies to ingest. Each tuple: (display_name, 10-K URL).
    "companyFilingUrls": [
        ("Tesla", "https://www.sec.gov/Archives/edgar/data/1318605/000162828024002390/tsla-20231231.htm"),
        ("General Motors", "https://www.sec.gov/Archives/edgar/data/1467858/000146785824000031/gm-20231231.htm"),
    ],

    # Director-name extraction
    "nameExtractionModel": "gpt-4o-mini",
    "nameExtractionModelTemperature": 0.4,

    # ReAct agent LLM
    "reactModelName": "gpt-4o",
    "reactModelTemperature": 0,

    # Tool result counts
    "numWebToolResults": 3,         # Tavily WebSearch returns top-N
    "numRetrieverToolResults": 3,   # VectorRerankerSearch returns top-N

    # Tool descriptions (used by the agent to decide when to call which tool)
    "directorToolName": "Company Directors Information",
    "directorToolDescription": (
        "Retrieve the names of company directors for a chosen company. "
        "Optionally, their LinkedIn handles can also be included. "
        "Use the format: company_name, true/false."
    ),
    "webToolName": "web_search",
    "webToolDescription": "Performs a web search on the query.",
    "retrieverToolName": "Vector Reranker Search",
    "retrieverToolDescription": (
        "Retrieves information from an embedding based vector DB containing "
        "financial data and company information. Structure query as a sentence."
    ),
}
