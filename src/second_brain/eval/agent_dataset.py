"""Agent evaluation dataset using Pydantic AI."""

from typing import Any
from second_brain.agents.thought_agent import ThoughtAgent
from second_brain.eval.pydantic_eval_compat import Dataset, Case, LLMJudge


def create_agent_dataset() -> Dataset[str, str, Any]:
    """Create evaluation dataset for agent responses."""
    
    agent_dataset = Dataset[str, str, Any](
        cases=[
            Case(
                name="learning_goals_query",
                inputs="What are my learning goals?",
                metadata={"category": "knowledge_retrieval", "difficulty": "easy"},
                evaluators=(
                    LLMJudge(
                        rubric="The response should mention learning goals from the knowledge base, such as LangChain, vector databases, RAG pipeline, OpenTelemetry, or hackathons.",
                        model="google-gla:gemini-2.5-pro"
                    ),
                ),
            ),
            Case(
                name="travel_ideas_query",
                inputs="Suggest me some travel ideas",
                metadata={"category": "knowledge_retrieval", "difficulty": "easy"},
                evaluators=(
                    LLMJudge(
                        rubric="The response should mention travel destinations like Japan, Italy, Iceland, Vietnam, or Himachal from the knowledge base.",
                        model="google-gla:gemini-2.5-pro"
                    ),
                ),
            ),
            Case(
                name="project_ideas_query",
                inputs="What project ideas do I have?",
                metadata={"category": "knowledge_retrieval", "difficulty": "easy"},
                evaluators=(
                    LLMJudge(
                        rubric="The response should mention project ideas such as AI-powered Second Brain, DevOps dashboard, Chess tutor app, Recipe recommendation system, or Daily journal summarizer.",
                        model="google-gla:gemini-2.5-pro"
                    ),
                ),
            ),
            Case(
                name="finance_tips_query",
                inputs="What are some finance tips?",
                metadata={"category": "knowledge_retrieval", "difficulty": "medium"},
                evaluators=(
                    LLMJudge(
                        rubric="The response should provide finance tips or mention financial information from the knowledge base.",
                        model="google-gla:gemini-2.5-pro"
                    ),
                ),
            ),
            Case(
                name="conversational_memory",
                inputs="What did we discuss about travel earlier?",
                metadata={"category": "memory_recall", "difficulty": "hard"},
                evaluators=(
                    LLMJudge(
                        rubric="The response should demonstrate memory of previous conversations about travel if any exist, or acknowledge lack of prior conversation.",
                        model="google-gla:gemini-2.5-pro"
                    ),
                ),
            ),
            Case(
                name="synthesis_query",
                inputs="Based on my notes, what should I focus on this quarter?",
                metadata={"category": "synthesis", "difficulty": "hard"},
                evaluators=(
                    LLMJudge(
                        rubric="The response should synthesize information from multiple notes (learning goals, career goals, project ideas) to provide a coherent recommendation.",
                        model="google-gla:gemini-2.5-pro"
                    ),
                ),
            ),
            Case(
                name="unclear_query",
                inputs="tell me something interesting",
                metadata={"category": "general", "difficulty": "medium"},
                evaluators=(
                    LLMJudge(
                        rubric="The response should attempt to retrieve relevant information from the knowledge base or acknowledge the vague nature of the query appropriately.",
                        model="google-gla:gemini-2.5-pro"
                    ),
                ),
            ),
        ],
        evaluators=[],
    )
    
    return agent_dataset


def main(query: str) -> str:
    """Main function to evaluate - runs the agent."""
    agent = ThoughtAgent()
    return agent.run(query)

