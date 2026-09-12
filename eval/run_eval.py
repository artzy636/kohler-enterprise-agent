"""Evaluation harness: runs eval/test_set.json against the pipeline and reports pass/fail."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import answer_query

TEST_SET_PATH = Path(__file__).resolve().parent / "test_set.json"
DELAY_SECONDS = 2


def main() -> None:
    test_cases = json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))
    total = len(test_cases)
    results = []

    for i, case in enumerate(test_cases, start=1):
        query = case["query"]
        expected_action = case["expected_action"]

        answer = answer_query(query, [])
        actual_action = answer.action

        passed = actual_action == expected_action
        status = "PASS" if passed else "FAIL"
        results.append(
            {"query": query, "expected_action": expected_action, "actual_action": actual_action, "passed": passed}
        )

        print(f"[{i}/{total}] {status} | expected={expected_action:<9} actual={actual_action:<9} | {query}")

        if i < total:
            time.sleep(DELAY_SECONDS)

    passed_count = sum(1 for r in results if r["passed"])
    failed_count = total - passed_count
    pass_rate = (passed_count / total * 100) if total else 0.0

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total tests: {total}")
    print(f"Passed:      {passed_count}")
    print(f"Failed:      {failed_count}")
    print(f"Pass rate:   {pass_rate:.1f}%")

    failed_cases = [r for r in results if not r["passed"]]
    if failed_cases:
        print()
        print("Failed cases:")
        for r in failed_cases:
            print(f"  - expected={r['expected_action']} actual={r['actual_action']} | {r['query']}")


if __name__ == "__main__":
    main()
