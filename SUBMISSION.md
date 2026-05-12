# Submission — HW Agentic RAG
**Name:** Smitha Bala
**Student ID:** sxb240026

## LangSmith trace URL
https://smith.langchain.com/o/4d383219-07d4-4a6e-9647-0a06c9fc49a0/projects/p/0766aa27-3504-431a-9a29-d83ad1bc4954/r/019e1a38-ed01-7fb2-8a4e-0528b437106d

## Final answer text
### Directors of Tesla and Their LinkedIn Handles:
1. **Elon Musk** - [LinkedIn](https://www.linkedin.com/in/ceomrmusk-96ba32324)
2. **Robyn Denholm** - [LinkedIn](https://au.linkedin.com/in/robyn-denholm-a807795)
3. **Ira Ehrenpreis** - [LinkedIn](https://www.linkedin.com/in/iraehrenpreis)
4. **Joseph Gebbia** - [LinkedIn](https://www.linkedin.com/in/jgebbia)
5. **James Murdoch** - [LinkedIn](https://www.linkedin.com/in/jamesrmurdoch)
6. **Kimbal Musk** - [LinkedIn](https://www.linkedin.com/in/kimbalmusk)
7. **JB Straubel** - [LinkedIn](https://www.linkedin.com/in/jb-straubel-b694981)
8. **Kathleen Wilson-Thompson** - [LinkedIn](https://www.linkedin.com/in/kathleen-wilson-thompson-275654201)

### Financial Goals of Tesla in 2023:
Tesla aims to meet several financial goals in 2023, including:
- Achieving RMB 14.08 billion in capital expenditures by the end of 2023.
- Generating RMB 2.23 billion of annual tax revenues starting at the end of 2023.
- Strengthening its workforce, with a global employee headcount of 140,473 as of December 31, 2023.

### Next Auto Show Tesla Will Participate In:
Tesla will participate in the **2023 Detroit Auto Show**, which will take place from September 13 to 24 at Huntington Place in Detroit. Tesla will showcase its vehicles at the Powering Michigan EV Experience and offer visitors a test drive. Additionally, Tesla will also participate in the **2023 Munich Auto Show**.

## What the agent did (1 paragraph)
The agent handled the multi-part demo question by invoking all three tools in sequence. First, it called `company_directors_information` with `"Tesla, true"` to extract director names from the end-section of Tesla's 10-K filing using `gpt-4o-mini` and look up each director's LinkedIn profile via SerpAPI — returning all 8 directors with URLs. Next, it called `vector_reranker_search` to retrieve Tesla's financial goals from the FAISS+Flashrank pipeline, pulling the most relevant chunks from the 2023 10-K. Finally, it called `web_search` via Tavily to find the next auto show Tesla would participate in. The question says "this year" and "the next auto show," but the 10-K data is from 2023, so the agent answered with 2023 figures and past auto shows — which is reasonable behavior given the data available. To make the agent more temporally aware, one improvement would be to inject the current date into the system prompt and instruct the agent to explicitly flag when its retrieval data may be outdated, or to prefer web search results over the vector store when the question contains time-sensitive language like "this year" or "next."

## Notes
- The `TEST_TO_RUN = "tools"` run (M5 test) happened before the full agent run in the same Python session, so the SerpAPI LinkedIn lookups were already cached in `_linkedin_cache`. The full agent run therefore reused those cached results and did not consume additional SerpAPI quota.
- A deprecation warning appeared during the full run: `create_react_agent has been moved to langchain.agents`. This does not affect functionality in the current version of LangGraph and can be resolved in a future refactor by updating the import.
- The `agent_graph.png` file was generated and saved to the `starter/` directory as required.
