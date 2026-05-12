"""
Vector store + reranker.

We build a FAISS index over the chunked filings (OpenAI embeddings), then
wrap it with a Flashrank reranker via LangChain's ContextualCompressionRetriever.

NOTE on the flashrank monkey-patch below: there's a version-skew bug between
`flashrank` and `langchain_community.document_compressors.flashrank_rerank`.
Without the patch, importing FlashrankRerank fails because the module looks
for RerankRequest in the wrong place. This snippet rebinds it.
PRE-FIXED FOR YOU — leave it in place; don't try to debug or simplify it.
"""

from typing import List, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# --- Flashrank monkey-patch (must run before importing FlashrankRerank) ---
from flashrank import Ranker, RerankRequest
import langchain_community.document_compressors.flashrank_rerank as fr_mod
fr_mod.RerankRequest = RerankRequest
from langchain_community.document_compressors import FlashrankRerank
# --- end monkey-patch ---

from langchain_classic.retrievers import ContextualCompressionRetriever


def create_vector_store(chunks: List[Document], embedding_model: str) -> FAISS:
    """
    TODO M4.1: Build a FAISS vector store from the chunks.

    Goal: embed each chunk with OpenAI, store the vectors in FAISS, return
    the FAISS object so downstream code can search it. Two real lines of work
    plus a print to confirm the build size.

    See MILESTONES.md M4 for the conceptual walk-through.
    """
    print(f"[retriever] Building FAISS index with model '{embedding_model}' over {len(chunks)} chunks ...")
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"[retriever] FAISS index built: {vectorstore.index.ntotal} vectors, dim={vectorstore.index.d}")
    return vectorstore


def create_retriever_with_reranking(
    vectorstore: FAISS,
    num_retrieved: int,
    reranker_model: str,
    num_reranked: int,
) -> Any:
    """
    TODO M4.2: Wrap the FAISS retriever with a Flashrank reranker.

    Goal: build a two-stage retriever — FAISS pulls top-`num_retrieved`
    candidates fast, then Flashrank reranks down to the top-`num_reranked`
    most relevant ones. Combine using ContextualCompressionRetriever.

    See MILESTONES.md M4 for the conceptual walk-through. The Flashrank
    monkey-patch above is required — don't remove it.
    """
    # Stage 1: FAISS retrieves top num_retrieved candidates by embedding similarity
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": num_retrieved})

    # Stage 2: Flashrank reranks candidates down to top num_reranked by relevance
    ranker_client = Ranker(model_name=reranker_model)
    compressor = FlashrankRerank(client=ranker_client, model=reranker_model, top_n=num_reranked)

    # Combine into a single retriever interface
    retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor,
    )
    print(f"[retriever] Two-stage retriever ready: FAISS top-{num_retrieved} → Flashrank top-{num_reranked}")
    return retriever


if __name__ == "__main__":
    # Smoke test: build the full retriever pipeline and run a sample query.
    from dotenv import load_dotenv
    load_dotenv()
    from config import CONFIG
    from ingestion import load_and_process_filings

    chunks, _ = load_and_process_filings(
        urls=CONFIG["companyFilingUrls"],
        chunk_size=CONFIG["chunkSize"],
        chunk_overlap=CONFIG["chunkOverlap"],
        user_agent=CONFIG["userAgentHeader"],
    )
    vectorstore = create_vector_store(chunks, CONFIG["embeddingModelName"])
    retriever = create_retriever_with_reranking(
        vectorstore=vectorstore,
        num_retrieved=CONFIG["numRetrievedDocuments"],
        reranker_model=CONFIG["rerankerModel"],
        num_reranked=CONFIG["numRerankedDocuments"],
    )

    sample = "What are Tesla's financial goals for 2024?"
    print(f"\nQuery: {sample}")
    docs = retriever.invoke(sample)
    print(f"\nReturned {len(docs)} reranked docs. First one's metadata:")
    print(docs[0].metadata)
    print(docs[0].page_content[:300])
