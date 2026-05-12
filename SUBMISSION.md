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
The agent had to answer the 4 part question and it decided it needed 3 tools to answer them. It called the directors tool, which grabbed the last 1000 characters of Tesla's 10-K filing to find the name of the director, then looked each person up on LinkedIn with the SerpAPI. It got back their profile URLs. Then, it called the vector search tool. The query it sent was about Tesla's financial goals to the VectorRerankerSearchTool. That tool searched the FAISS index, pulled the top 12 by embedding similarity, then reranked them to the 5 most relevant and gave them to the agent. Then, the agent called the web search tool. It sent a query about Tesla's auto shows to the tool, which used Tavily and returned the top 3 results. Then it combined all the outputs from the tools and wrote the response given above in Final Answer Text. What is interesting to note is that the question says "this year," meaning 2026, but the 10-K data is from 2023, so even though the agent answered with what information it had from the 10-K data, it is not actually totally accurate.

## Notes
- The `TEST_TO_RUN = "tools"` run (M5 test) happened before the full agent run in the same Python session, so the SerpAPI LinkedIn lookups were already cached in `_linkedin_cache`. The full agent run therefore reused those cached results and did not consume additional SerpAPI quota.
- A deprecation warning appeared during the full run: `create_react_agent has been moved to langchain.agents`. This does not affect functionality in the current version of LangGraph and can be resolved in a future refactor by updating the import.
- The `agent_graph.png` file was generated and saved to the `starter/` directory as required.
