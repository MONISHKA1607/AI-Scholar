from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import GEMINI_KEY, GEMINI_MODEL
from . import tools

model = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GEMINI_KEY,
    temperature=0.1,
)


def rag(user_query: str):
    """Classify the user query and return retrieved context.

    Returns either a structured context dict (for downstream citation handling)
    or a plain string for short-circuit cases (non-research, retrieval failure).
    """

    classifier_prompt = f"""
      You are a Research Question Classification Agent.
      Your task is to analyze a user's question and classify it into one of three categories based on intent and context.

      Classification categories:

      - Specific-to-paper — The question references or implies a specific research paper, author, DOI, or title (e.g., "What does the 2021 paper by Smith et al. conclude about transformers?").

      - Generic-research — The question is about a research topic or domain in general, not a single paper (e.g., "How do attention mechanisms work in NLP?").

      - Non-research — The question is unrelated to academic or scientific research (e.g., "What's the weather today?" or "Write a poem.").

      Instructions:

      Always output only one label: specific-to-paper, generic-research, or non-research.

      If uncertain, choose the closest category by analyzing research intent.

      Do not include explanations or reasoning in the final output.

      Question: {user_query}
    """

    response = model.invoke(classifier_prompt)
    query_type = response.content.strip().lower()

    if query_type == "non-research":
        return "Irrelevant question."

    if query_type == "generic-research":
        agent_prompt = """
          You are a **Context Extractor Agent**. Your sole purpose is to gather and return **only the relevant context** needed to answer the user's query — not the final answer itself.

          You have access to two tools: `kb_retrieval` (internal knowledge base) and `web_search` (public search engine).

          **PROCEDURE:**

          1. **PRIMARY SOURCE:** First, attempt to retrieve information using the `kb_retrieval` tool.
          - **Action:** Call `kb_retrieval` with the user's exact query.
          - **Observation Analysis:** Evaluate the content returned by `kb_retrieval`.
          - **IF** the content is sufficient or partially relevant → **return only the relevant context** (verbatim; no rewriting or summarizing).
          - **IF** the content is empty or clearly irrelevant → proceed to Step 2.

          2. **SECONDARY SOURCE (Fallback):** If content returned by `kb_retrieval` is empty or clearly irrelevant, call the `web_search` tool.
          - **Action:** Call `web_search`, following instructions mentioned in tool definition.
          - **Observation Analysis:** Evaluate the search results.
          - **IF** relevant → **return only the relevant parts** (original wording).
          - **IF** still unhelpful → return:
            ```
            No relevant context could be retrieved.
            ```
              and add reason as to why you felt the content is not enough to answer the user's query.

          **RULES:**
          - Do **not** generate or formulate an answer to the query.
          - Do **not** add explanations, reasoning, or commentary.
          - Return **only the extracted text** or the "no relevant context" message.
        """

        agent = create_agent(model, tools=[tools.kb_retrieval, tools.web_search])
        return _run_agent(agent, agent_prompt, user_query)

    if query_type == "specific-to-paper":
        agent_prompt = """
          You are a **Context Extractor Agent**. Your sole purpose is to gather and return **only the relevant context** needed to answer the user's query — not the final answer itself.

          You have access to two tools: `kb_retrieval` and `specific_web_search`.

          **PROCEDURE:**

          1. **PRIMARY SOURCE:** First, attempt to retrieve information using the `kb_retrieval` tool.
          - **Action:** Call `kb_retrieval` with the user's exact query.
          - **Observation Analysis:** Evaluate the content returned by `kb_retrieval`.
          - **IF** the content is sufficient or partially relevant → **return only the relevant context** (verbatim; no rewriting or summarizing).
          - **IF** the content is empty or clearly irrelevant → proceed to Step 2.

          2. **SECONDARY SOURCE (Fallback):** If content returned by `kb_retrieval` is empty or clearly irrelevant, call the `specific_web_search` tool.
          - **Action:** Call `specific_web_search`, following instructions mentioned in tool definition.
          - **Observation Analysis:** Evaluate the search results.
          - **IF** relevant → **return only the relevant parts** (original wording).
          - **IF** still unhelpful → return exactly:
            ```
            No relevant context could be retrieved.
            ```

          **RULES:**
          - Do **not** generate or formulate an answer to the query.
          - Do **not** add explanations, reasoning, or commentary.
          - Return **only the extracted text** or the "no relevant context" message.
        """

        agent = create_agent(
            model, tools=[tools.kb_retrieval, tools.specific_web_search]
        )
        return _run_agent(agent, agent_prompt, user_query)

    return "Irrelevant question."


def _run_agent(agent, agent_prompt: str, user_query: str) -> str:
    conversation = agent.invoke(
        {
            "messages": [
                SystemMessage(content=agent_prompt),
                HumanMessage(content=f"User's Question: {user_query}"),
            ]
        }
    )

    messages = conversation["messages"]
    final_message = next(
        (
            m
            for m in reversed(messages)
            if getattr(m, "content", None)
            and str(m.content).strip() not in ["", None]
            and not isinstance(m, SystemMessage)
        ),
        None,
    )

    if final_message:
        return final_message.content
    return "No relevant context could be retrieved."
