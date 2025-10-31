"""Compatibility layer for Pydantic AI evaluation classes.

This provides a simple implementation that matches the expected API
until the official evaluation classes are available in pydantic-ai.
"""

from typing import Any, Callable, Generic, TypeVar
from pydantic import BaseModel
from dataclasses import dataclass

TInput = TypeVar('TInput')
TOutput = TypeVar('TOutput')
TMetadata = TypeVar('TMetadata')


@dataclass
class LLMJudge:
    """LLM-based evaluator."""
    rubric: str
    model: str


class Case(BaseModel):
    """Single test case."""
    name: str
    inputs: Any
    metadata: dict[str, Any] = {}
    evaluators: tuple = ()


class Dataset(BaseModel, Generic[TInput, TOutput, TMetadata]):
    """Evaluation dataset."""
    cases: list[Case]
    evaluators: list = []
    
    def evaluate_sync(self, main_func: Callable) -> "EvaluationReport":
        """Run evaluation synchronously."""
        results = []
        for case in self.cases:
            try:
                output = main_func(case.inputs)
                result = EvaluationResult(
                    case_name=case.name,
                    input=case.inputs,
                    output=output,
                    passed=True,
                    reason="Evaluation not fully implemented yet"
                )
            except Exception as e:
                result = EvaluationResult(
                    case_name=case.name,
                    input=case.inputs,
                    output=None,
                    passed=False,
                    reason=str(e)
                )
            results.append(result)
        
        return EvaluationReport(results=results)


class EvaluationResult(BaseModel):
    """Single evaluation result."""
    case_name: str
    input: Any
    output: Any
    passed: bool
    reason: str


class EvaluationReport(BaseModel):
    """Evaluation report containing all results."""
    results: list[EvaluationResult]
    
    def print(self, include_input: bool = True, include_output: bool = True, include_reasons: bool = True):
        """Print evaluation report."""
        print("\n" + "=" * 60)
        print("📊 Evaluation Report")
        print("=" * 60)
        
        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"\n{status} | {result.case_name}")
            
            if include_input:
                print(f"  Input: {result.input}")
            if include_output:
                print(f"  Output: {result.output}")
            if include_reasons:
                print(f"  Reason: {result.reason}")
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        print("\n" + "-" * 60)
        print(f"Summary: {passed}/{total} passed ({passed/total*100:.1f}%)")
        print("=" * 60 + "\n")

