"""Retrieval evaluation dataset using Pydantic AI."""

from typing import Any
from second_brain.agents.ingestor import RAGManager
from second_brain.eval.pydantic_eval_compat import Dataset, Case, LLMJudge


def create_retrieval_dataset() -> Dataset[str, str, Any]:
    """Create evaluation dataset for RAG retrieval."""
    
    retrieval_dataset = Dataset[str, str, Any](
        cases=[
            Case(
                name="exact_match_query",
                inputs="learning goals",
                metadata={"category": "exact_match", "difficulty": "easy"},
                evaluators=(
                    LLMJudge(
                        rubric="The retrieved context should contain information about learning goals, including topics like LangChain, vector databases, or RAG pipeline.",
                        model="google-gla:gemini-2.5-pro"
                    ),
                ),
            ),
            Case(
                name="semantic_query",
                inputs="places to visit for vacation",
                metadata={"category": "semantic_search", "difficulty": "medium"},
                evaluators=(
                    LLMJudge(
                        rubric="The retrieved context should contain travel-related information such as destinations, travel plans, or travel ideas.",
                        model="google-gla:gemini-2.5-pro"
                    ),
                ),
            ),
            Case(
                name="project_ideas_query",
                inputs="AI projects and ideas",
                metadata={"category": "topical", "difficulty": "easy"},
                evaluators=(
                    LLMJudge(
                        rubric="The retrieved context should mention project ideas, especially AI-related ones like Second Brain or other tech projects.",
                        model="google-gla:gemini-2.5-pro"
                    ),
                ),
            ),
            Case(
                name="finance_query",
                inputs="money management tips",
                metadata={"category": "topical", "difficulty": "medium"},
                evaluators=(
                    LLMJudge(
                        rubric="The retrieved context should contain financial information, tips, or records related to finance.",
                        model="google-gla:gemini-2.5-pro"
                    ),
                ),
            ),
            Case(
                name="vague_query",
                inputs="stuff",
                metadata={"category": "vague", "difficulty": "hard"},
                evaluators=(
                    LLMJudge(
                        rubric="The retrieval should handle vague queries gracefully, either returning relevant general content or acknowledging limited context.",
                        model="google-gla:gemini-2.5-pro"
                    ),
                ),
            ),
        ],
        evaluators=[],
    )
    
    return retrieval_dataset


def main(query: str) -> str:
    """Main function to evaluate - runs RAG retrieval."""
    rag_manager = RAGManager()
    return rag_manager.rag_retrieve(query, n_results=3)

