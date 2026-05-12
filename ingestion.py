"""
Document ingestion: pull SEC 10-K filings, split into chunks, capture
each filing's last 1000 chars (where the directors section lives).

Returns:
    chunks            — list[Document] with company metadata attached
    director_sections — dict[str, str] mapping company name to last 1000 chars

NOTE: SEC.gov rejects requests with the default urllib User-Agent. We pass a
custom UA via header_template — that's why the loader has the extra param.
This is provided for you in the function signature; don't remove it.
"""

from typing import List, Dict, Tuple
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_process_filings(
    urls: List[Tuple[str, str]],
    chunk_size: int,
    chunk_overlap: int,
    user_agent: str,
) -> Tuple[List[Document], Dict[str, str]]:
    """
    TODO M3: Build the ingestion logic.

    Goal: walk the (company, url) list, load each filing as a list of Documents,
    split them into chunks, and tag each chunk with its company name. Also
    capture the last 1000 chars of each filing into director_sections[company]
    — that's where 10-K director listings live, and the directors tool will
    use it later.

    The function signature already includes `user_agent` because SEC.gov
    blocks default urllib UAs. Make sure WebBaseLoader gets this UA.

    Print progress per company so debug runs are readable.

    See MILESTONES.md M3 for the conceptual walk-through.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks: List[Document] = []
    director_sections: Dict[str, str] = {}

    for company, url in urls:
        print(f"\n[ingestion] Loading {company} from {url} ...")
        loader = WebBaseLoader(url, header_template={"User-Agent": user_agent})
        docs = loader.load()
        print(f"[ingestion] Loaded {len(docs)} document(s) for {company}")

        # Capture last 1000 chars — director listings live at the end of 10-K filings
        director_sections[company] = docs[0].page_content[-1000:]

        # Split into chunks
        chunks = splitter.transform_documents(docs)

        # Tag each chunk with company name so the agent can tell them apart
        for chunk in chunks:
            chunk.metadata["company"] = company

        print(f"[ingestion] Split into {len(chunks)} chunks for {company}")
        all_chunks.extend(chunks)

    print(f"\n[ingestion] Total chunks across all filings: {len(all_chunks)}")
    return all_chunks, director_sections


if __name__ == "__main__":
    # Smoke test: load filings, print stats.
    from dotenv import load_dotenv
    load_dotenv()
    from config import CONFIG

    chunks, director_sections = load_and_process_filings(
        urls=CONFIG["companyFilingUrls"],
        chunk_size=CONFIG["chunkSize"],
        chunk_overlap=CONFIG["chunkOverlap"],
        user_agent=CONFIG["userAgentHeader"],
    )
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Sample chunk metadata: {chunks[0].metadata}")
    print(f"Director sections captured for: {list(director_sections.keys())}")
