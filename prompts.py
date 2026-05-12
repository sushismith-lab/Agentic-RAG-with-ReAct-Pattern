"""
Prompts for the agentic RAG HW.

Two prompts:
- REACT_PROMPT: drives the ReAct agent's reasoning loop. Standard LangChain ReAct format.
- NAME_EXTRACTION_PROMPT: extracts director names from a snippet of 10-K text.
"""

REACT_PROMPT = """Your task is to gather relevant information to build context for the question. Focus on collecting details related to the question.
Gather as much context as possible before formulating your answer.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer

Thought: you should always think about what to do

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action

Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know the final answer

Final Answer: the final answer to the question.

Follow these steps:

Begin!

Question: {input}

Thought:{agent_scratchpad}
"""


NAME_EXTRACTION_PROMPT = """
Extract and list the names of all individuals with the title 'Director' from the following text, excluding any additional information such as dates or signatures.
Present the names as a simple, comma-separated list.

{text}
"""


# The complex multi-tool demo question from the notebook.
DEMO_QUESTION = (
    "Who are the directors of Tesla. What are their linkedin handles? "
    "What are the financial goals of tesla this year. "
    "What is the next auto show that Tesla will participate in."
)
