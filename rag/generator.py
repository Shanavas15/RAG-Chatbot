import logging
import os
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()
logger = logging.getLogger(__name__)

NO_ANSWER = "NO_ANSWER_FOUND"


class LLMError(RuntimeError):
    """Raised when the LLM provider call fails. Callers must handle this explicitly."""


class RAGGenerator:
    def __init__(
        self,
        model_name: str | None = None,
        temperature: float = 0.2,
    ):
        
        model_name = model_name or os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b"

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file or environment."
            )

        print(f"Connecting to GroqCloud with model '{model_name}'...")
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            max_retries=2,
            timeout=60,
        )

        self.doc_prompt_template = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template="""You are an expert, professional enterprise AI assistant.
Answer the question strictly using the provided context below.
If the answer is not contained in the context, respond with exactly: NO_ANSWER_FOUND
Do not guess or use outside knowledge. Do not add any other text if the answer is not found.

PREVIOUS CONVERSATION HISTORY:
{chat_history}

CONTEXT:
{context}

QUESTION:
{question}

HELPFUL ANSWER:""",
        )

        self.web_prompt_template = PromptTemplate(
            input_variables=["web_context", "question"],
            template="""You are a helpful AI assistant. You couldn't find the answer in the user's local documents, so you searched the web.
Use the provided Web Search Results below to answer the user's question accurately.

WEB SEARCH RESULTS:
{web_context}

QUESTION:
{question}

HELPFUL ANSWER:""",
        )

        self.rewrite_prompt_template = PromptTemplate(
            input_variables=["chat_history", "question"],
            template="""You are an expert search query generator.
Given the conversation history and a follow-up question, rephrase the follow-up question into a standalone search query for a search engine.
If the question is already standalone, return the original question.
Output ONLY the search query. Do not answer the question. No explanations.

HISTORY:
{chat_history}

QUESTION:
{question}

STANDALONE QUERY:""",
        )

    def _invoke(self, prompt: str) -> str:
        """Single place where LLM calls happen. Wraps all provider errors in LLMError."""
        try:
            response = self.llm.invoke(prompt)
        except Exception as e:
            logger.exception("Groq request failed")
            raise LLMError(str(e)) from e

        content = response.content
        if not content or not content.strip():
            raise LLMError("Model returned an empty response.")
        return content.strip()

    def generate_answer(
        self,
        question: str,
        documents: List[Document],
        chat_history: str = "",
    ) -> str:
        """Answer from local documents. Returns NO_ANSWER when context lacks the answer.
        Raises LLMError if the provider call fails."""
        if not documents:
            return NO_ANSWER

        context_parts = []
        for doc in documents:
            page = doc.metadata.get("page", "Unknown")
            context_parts.append(f"--- SOURCE PAGE: {page} ---\n{doc.page_content}")

        prompt = self.doc_prompt_template.format(
            context="\n\n".join(context_parts),
            chat_history=chat_history,
            question=question,
        )
        return self._invoke(prompt)

    def generate_web_answer(self, question: str, web_context: str) -> str:
        """Answer from web search results. Raises LLMError if the provider call fails."""
        prompt = self.web_prompt_template.format(
            web_context=web_context,
            question=question,
        )
        return self._invoke(prompt)

    def rewrite_query(self, question: str, chat_history: str) -> str:
        """Turn a follow-up question into a standalone search query.
        Falls back to the original question on failure, since that is a safe degradation."""
        if not chat_history.strip():
            return question

        prompt = self.rewrite_prompt_template.format(
            chat_history=chat_history,
            question=question,
        )
        try:
            return self._invoke(prompt) or question
        except LLMError as e:
            logger.warning("Query rewrite failed, using original question: %s", e)
            return question