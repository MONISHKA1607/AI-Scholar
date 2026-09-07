from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
from asyncio import sleep
from google.genai import types
from ..rag.main import rag
from ..config import ai, GEMINI_MODEL

class Body(BaseModel):
    user_query: str

router = APIRouter(prefix="/fastapi/chat", tags=["chat"])

async def genRAG(user_query: str):
    try:
        model = GEMINI_MODEL
        if (model == None):
            model = "gemini-2.5-flash"

        rag_context = rag(user_query)
        for chunk in ai.models.generate_content_stream(
            model=model,
            contents=f"""
            Context: {rag_context}\n\n
            Question: {user_query}\n\n

            ---

            **Strict Rules (MUST follow exactly):**

            1. **Source-of-Truth Rule**:
            - **ONLY** use information explicitly present in the **Provided Context** above.
            - **DO NOT** invent, infer, recall from training data, or hallucinate any facts, numbers, names, or details not directly stated in the context.

            2. **Insufficient Context Rule**:
            - If the context does **not** contain enough information to answer the question **accurately and completely**, respond **only** with:
                > **"Sorry but cannot answer your question at the moment"**
                and add a reason as to why you felt the context is insufficient to answer the user's query.
            - Do **not** guess, partially answer, or suggest external sources.

            3. **Citation Rule** (critical):
            - The Context above is composed of chunks. Each chunk begins with a header of the form
              `[^n] (Title; Authors; Year) — URL`. Use exactly those `[^n]` markers when citing.
            - Every factual claim or quoted excerpt must end with the matching `[^n]` marker (e.g., "Transformers use multi-head attention[^2].").
            - Do **not** invent citation numbers. Do **not** cite chunks that you did not actually use.
            - End the answer with a `## References` section listing every cited `[^n]`, one per line, in the format:
              `[^n]: Title — Authors (Year). URL` (omit fields that are absent in the chunk header).
            - If the Context contains no `[^n]` headers, omit the References section but still follow the Source-of-Truth Rule.

            4. **Answer Formatting Rules**:
            - Use **clear, concise, and structured** Markdown formatting:
                - **Bold** for key terms or emphasis
                - *Italics* for definitions or subtle points
                - `Code blocks` for technical terms, file paths, or snippets
                - Bullet points or numbered lists for multiple items
                - Tables when comparing data
                - LaTeX equations via `$$...$$` if needed (e.g., $$  E = mc^2  $$)
            - Quote direct excerpts from context using `> blockquotes` when referencing evidence.
            - Keep the main answer **under 300 words** (the References section does not count toward the limit).

            **Now, answer the question using only the context and rules above.**
            """,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            ),
        ):
            yield f"data: {json.dumps({'chunk': chunk.text, 'finished': False})}\n\n"
            await sleep(0)

        yield f"data: {json.dumps({'chunk': '', 'finished': True})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e), 'finished': True})}\n\n"


@router.post("/generate", status_code=200)
async def gen_answer(body: Body):
    return StreamingResponse(
        genRAG(body.user_query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        },
    )
