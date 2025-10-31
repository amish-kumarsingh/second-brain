"""Run all evaluations using Pydantic AI datasets."""

from second_brain.eval.agent_dataset import create_agent_dataset, main as agent_main
from second_brain.eval.retrieval_dataset import create_retrieval_dataset, main as retrieval_main


def run_agent_evaluation():
    """Run agent evaluation."""
    print("\n🤖 Agent Evaluation")
    print("=" * 60)
    dataset = create_agent_dataset()
    report = dataset.evaluate_sync(agent_main)
    report.print(include_input=True, include_output=True, include_reasons=True)


def run_retrieval_evaluation():
    """Run retrieval evaluation."""
    print("\n🔍 Retrieval Evaluation")
    print("=" * 60)
    dataset = create_retrieval_dataset()
    report = dataset.evaluate_sync(retrieval_main)
    report.print(include_input=True, include_output=True, include_reasons=True)


def run_all_evaluations():
    """Run all evaluation datasets."""
    print("\n🧪 Running All Evaluations")
    print("=" * 60)
    
    run_retrieval_evaluation()
    print("\n")
    run_agent_evaluation()
    
    print("\n" + "=" * 60)
    print("✅ All evaluations complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_evaluations()
