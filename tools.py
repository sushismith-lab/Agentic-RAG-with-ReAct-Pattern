"""
Three tools for the ReAct agent:

1. VectorRerankerSearchTool — searches the FAISS+reranker pipeline.
2. WebSearchTool — general web search via Tavily.
3. CompanyDirectorsTool — extracts director names + LinkedIn lookup via SerpAPI.

Why two web-search APIs?
- Tavily returns LLM-optimized summaries: great for "what's the next auto show?"
- SerpAPI is a thin Google proxy: great for URL-precise queries with the
  `site:linkedin.com/in/` operator. Tavily's reranking interferes with that
  precision; SerpAPI passes the operator through to Google verbatim.

NOTE on tool names: OpenAI's tool-calling API validates names against
^[a-zA-Z0-9_-]+$. NO SPACES — use snake_case.
"""

from typing import Any, Dict, List
from langchain_core.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_community.utilities import SerpAPIWrapper

from prompts import NAME_EXTRACTION_PROMPT


# ============================================================================
# Vector Reranker Search Tool
# ============================================================================

class VectorRerankerSearchTool(BaseTool):
    """Searches the FAISS+reranker pipeline. Returns top-N reranked chunks."""

    name: str = "vector_reranker_search"
    description: str = (
        "Retrieves information from an embedding based vector DB containing "
        "financial data and company information. Structure query as a sentence."
    )

    retriever: Any = None
    num_results: int = 3

    def __init__(self, retriever: Any, num_results: int):
        super().__init__(retriever=retriever, num_results=num_results)

    def _run(self, query: str) -> str:
        """
        TODO M5.1: Invoke the reranking retriever and return the top results
        as a single string the agent can read.

        The retriever (`self.retriever`) gives you `Document` objects with
        `metadata` and `page_content`. Limit to `self.num_results` and
        format each result so the agent can tell which company a chunk is
        from. See MILESTONES.md M5.1.
        """
        try:
            docs = self.retriever.invoke(query)
            formatted = []
            for doc in docs[:self.num_results]:
                company = doc.metadata.get("company", "Unknown")
                formatted.append(f"{company}\n{doc.page_content}")
            return "\n\n".join(formatted)
        except Exception as e:
            return f"Error during vector search: {str(e)}"


# ============================================================================
# Web Search Tool — Tavily (general queries)
# ============================================================================

class WebSearchTool(BaseTool):
    """General web search via Tavily. LLM-friendly summaries."""

    name: str = "web_search"
    description: str = "Performs a web search on the query."

    num_results: int = 3
    search: Any = None

    def __init__(self, num_results: int):
        super().__init__(
            num_results=num_results,
            search=TavilySearch(max_results=num_results),
        )

    def _run(self, query: str) -> str:
        """
        TODO M5.2: Run a Tavily search and format the top results.

        Same Tavily shape you used in Reflexion — invoke `self.search` with
        a `{"query": ...}` dict, get back a dict whose `"results"` field is
        a list of dicts (each with title / content / url). Format up to
        `self.num_results` of them as readable text and join. See
        MILESTONES.md M5.2.
        """
        try:
            results = self.search.invoke({"query": query})
            hits = results.get("results", [])
            if not hits:
                return "No web results found."
            formatted = []
            for hit in hits[:self.num_results]:
                formatted.append(
                    f"Title: {hit.get('title', '')}\n"
                    f"Snippet: {hit.get('content', '')}\n"
                    f"Link: {hit.get('url', '')}"
                )
            return "\n\n".join(formatted)
        except Exception as e:
            return f"Error during web search: {str(e)}"


# ============================================================================
# Company Directors Tool — name extraction + SerpAPI LinkedIn lookup
# ============================================================================

# Module-level cache so repeated runs of the same agent (in one process)
# don't re-query SerpAPI for the same person.
_linkedin_cache: Dict[str, str] = {}


class CompanyDirectorsTool(BaseTool):
    """
    Two-stage tool:
      1. Extract director names from the cached 10-K end-section using gpt-4o-mini.
      2. (Optional) Look up each director's LinkedIn profile via SerpAPI with
         a `site:linkedin.com/in/` query.

    Input format: "company_name, true|false"
    """

    name: str = "company_directors_information"
    description: str = (
        "Retrieve the names of company directors for a chosen company. "
        "Optionally, their LinkedIn handles can also be included. "
        "Use the format: company_name, true/false."
    )

    director_sections: Dict[str, str] = {}
    extraction_model: str = "gpt-4o-mini"
    extraction_temperature: float = 0.4

    def __init__(
        self,
        director_sections: Dict[str, str],
        extraction_model: str,
        extraction_temperature: float,
    ):
        # Inject the available companies into the description so the agent
        # knows what's queryable.
        available = ", ".join(director_sections.keys())
        desc = (
            f"Retrieve the names of company directors for a chosen company. "
            f"Optionally, their LinkedIn handles can also be included. "
            f"Use the format: company_name, true/false. "
            f"Available companies: {available}"
        )
        super().__init__(
            director_sections=director_sections,
            extraction_model=extraction_model,
            extraction_temperature=extraction_temperature,
            description=desc,
        )

    def _run(self, query: str) -> str:
        """
        TODO M5.3a: Orchestrate the two-stage tool.

        Input format from the agent: "company_name, true|false" — parse it.
        Look up the cached director section for that company, extract the
        names with `_extract_director_names`, optionally look up each name's
        LinkedIn URL with `_get_linkedin_handle`. Format and return.

        Failure modes to handle: unknown company, empty extraction, exception
        during any step. See MILESTONES.md M5.3.
        """
        try:
            # Parse "company_name, true|false"
            parts = query.split(",", 1)
            company = parts[0].strip()
            include_linkedin = True
            if len(parts) > 1:
                include_linkedin = parts[1].strip().lower() != "false"

            # Look up cached 10-K director section for this company
            section = self.director_sections.get(company)
            if section is None:
                available = ", ".join(self.director_sections.keys())
                return f"Unknown company '{company}'. Available: {available}"

            # Extract director names via LLM
            names = self._extract_director_names(section)
            if not names:
                return f"Could not extract director names for {company}."

            if include_linkedin:
                results = []
                for name in names:
                    handle = self._get_linkedin_handle(name, company)
                    results.append(f"{name} (LinkedIn: {handle})")
                return "; ".join(results)
            else:
                return ", ".join(names)
        except Exception as e:
            return f"Error in CompanyDirectorsTool: {str(e)}"

    def _extract_director_names(self, text: str) -> List[str]:
        """
        TODO M5.3b: Use a small LLM + a parser to extract director names
        from the text snippet.

        This is the same `prompt | llm | parser` chain pattern from Reflexion,
        but the parser here is a CommaSeparatedListOutputParser (you want a
        plain Python list of strings, not a Pydantic object). The LLM model
        and temperature live on `self.extraction_*` attributes. The prompt
        is `NAME_EXTRACTION_PROMPT` (already imported).

        Try/except wrap — return [] on parse failure so the tool degrades
        gracefully. See MILESTONES.md M5.3.
        """
        try:
            llm = ChatOpenAI(
                model=self.extraction_model,
                temperature=self.extraction_temperature,
            )
            parser = CommaSeparatedListOutputParser()
            prompt = PromptTemplate.from_template(NAME_EXTRACTION_PROMPT)
            chain = prompt | llm | parser
            return chain.invoke({"text": text})
        except Exception:
            return []

    def _get_linkedin_handle(self, name: str, company: str) -> str:
        """
        TODO M5.3c: Look up a LinkedIn profile URL via SerpAPI.

        Three steps in order: check `_linkedin_cache` first (saves SerpAPI
        quota — this is a 100/mo free tier), if not cached run a SerpAPI
        query scoped to LinkedIn profiles using the `site:linkedin.com/in/`
        operator with the person's name and company, pluck the first
        organic result, cache it, return.

        Wrap in try/except — return an error string on failure. See
        MILESTONES.md M5.3.
        """
        cache_key = f"{name}::{company}"
        if cache_key in _linkedin_cache:
            return _linkedin_cache[cache_key]

        try:
            serpapi = SerpAPIWrapper()
            query = f'"{name}" {company} site:linkedin.com/in/'
            result = serpapi.results(query)
            organic = result.get("organic_results", [])
            link = organic[0].get("link", "Profile not found") if organic else "Profile not found"
            _linkedin_cache[cache_key] = link
            return link
        except Exception as e:
            return f"LinkedIn lookup failed: {str(e)}"


if __name__ == "__main__":
    # Smoke test each tool independently.
    from dotenv import load_dotenv
    load_dotenv()
    from config import CONFIG
    from ingestion import load_and_process_filings
    from retriever import create_vector_store, create_retriever_with_reranking

    print("=== Building pipeline ===")
    chunks, director_sections = load_and_process_filings(
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

    print("\n=== Test 1: Vector retriever tool ===")
    rt = VectorRerankerSearchTool(retriever=retriever, num_results=CONFIG["numRetrieverToolResults"])
    print(rt._run("What are Tesla's financial goals for 2024?")[:600])

    print("\n=== Test 2: Web search tool (Tavily) ===")
    wt = WebSearchTool(num_results=CONFIG["numWebToolResults"])
    print(wt._run("Next Tesla auto show participation 2026")[:600])

    print("\n=== Test 3: Directors tool (SerpAPI for LinkedIn) ===")
    dt = CompanyDirectorsTool(
        director_sections=director_sections,
        extraction_model=CONFIG["nameExtractionModel"],
        extraction_temperature=CONFIG["nameExtractionModelTemperature"],
    )
    print(dt._run("Tesla, true")[:1500])
