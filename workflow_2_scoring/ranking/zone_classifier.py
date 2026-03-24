"""
Zone Classification Module
Divides candidates into 3 zones based on score thresholds

Zones:
- SELECTED: Score >= 75 (move to next round)
- BORDERLINE: Score 40-74 (send feedback email)
- POOR_MATCH: Score < 40 (no action)
"""

from typing import Dict, List, Any
from loguru import logger


class ZoneClassifier:
    """
    Classifies candidates into zones based on their ATS scores.

    Zones:
        - SELECTED: High scores, move to interview
        - BORDERLINE: Medium scores, send constructive feedback
        - POOR_MATCH: Low scores, no actionable feedback
    """

    def __init__(self, selected_threshold: float = 75.0, borderline_threshold: float = 40.0):
        self.selected_threshold = selected_threshold
        self.borderline_threshold = borderline_threshold
        logger.info(f"ZoneClassifier initialized: Selected >= {selected_threshold}, Borderline >= {borderline_threshold}")

    def classify(self, final_score: float) -> Dict[str, Any]:
        if final_score >= self.selected_threshold:
            return {
                "zone": "SELECTED",
                "action": "MOVE_TO_NEXT_ROUND",
                "send_feedback": False,
                "explanation": "Score meets threshold for interview"
            }
        elif final_score >= self.borderline_threshold:
            return {
                "zone": "BORDERLINE",
                "action": "SEND_FEEDBACK_EMAIL",
                "send_feedback": True,
                "explanation": "Close to requirements - feedback will help improve"
            }
        else:
            return {
                "zone": "POOR_MATCH",
                "action": "NO_ACTION",
                "send_feedback": False,
                "explanation": "Score too far from requirements for actionable feedback"
            }

    def batch_classify(self, candidates_with_scores: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        results = {"selected": [], "borderline": [], "poor_match": []}

        for candidate in candidates_with_scores:
            score = candidate.get("score") or candidate.get("ats_score", 0)
            classification = self.classify(score)

            candidate_with_zone = candidate.copy()
            candidate_with_zone["zone"] = classification["zone"]
            candidate_with_zone["action"] = classification["action"]
            candidate_with_zone["send_feedback"] = classification["send_feedback"]
            candidate_with_zone["zone_explanation"] = classification["explanation"]

            if classification["zone"] == "SELECTED":
                results["selected"].append(candidate_with_zone)
            elif classification["zone"] == "BORDERLINE":
                results["borderline"].append(candidate_with_zone)
            else:
                results["poor_match"].append(candidate_with_zone)

        results["summary"] = {
            "total": len(candidates_with_scores),
            "selected_count": len(results["selected"]),
            "borderline_count": len(results["borderline"]),
            "poor_match_count": len(results["poor_match"]),
            "selected_threshold": self.selected_threshold,
            "borderline_threshold": self.borderline_threshold
        }

        logger.info(f"Classification: {results['summary']['selected_count']} selected, {results['summary']['borderline_count']} borderline")
        return results

    def get_feedback_candidates(self, classified_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        return classified_results.get("borderline", [])

    def get_summary_stats(self, classified_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate summary statistics from classification results."""
        total = (
            len(classified_results.get("selected", [])) +
            len(classified_results.get("borderline", [])) +
            len(classified_results.get("poor_match", []))
        )

        if total == 0:
            return {
                "total_candidates": 0,
                "selected_count": 0,
                "borderline_count": 0,
                "poor_match_count": 0,
                "selected_percentage": 0.0,
                "borderline_percentage": 0.0,
                "poor_match_percentage": 0.0,
                "feedback_emails_to_send": 0
            }

        return {
            "total_candidates": total,
            "selected_count": len(classified_results.get("selected", [])),
            "borderline_count": len(classified_results.get("borderline", [])),
            "poor_match_count": len(classified_results.get("poor_match", [])),
            "selected_percentage": round(len(classified_results.get("selected", [])) / total * 100, 1),
            "borderline_percentage": round(len(classified_results.get("borderline", [])) / total * 100, 1),
            "poor_match_percentage": round(len(classified_results.get("poor_match", [])) / total * 100, 1),
            "feedback_emails_to_send": len(classified_results.get("borderline", []))
        }

    def print_classification_report(self, classified_results: Dict[str, List[Dict[str, Any]]]) -> None:
        summary = classified_results.get("summary", {})
        print("\n" + "=" * 70)
        print("[ZONE] CANDIDATE CLASSIFICATION RESULTS")
        print("=" * 70)
        print(f"Thresholds: Selected >= {summary.get('selected_threshold', 75)}, Borderline >= {summary.get('borderline_threshold', 40)}")
        print("-" * 70)

        selected = classified_results.get("selected", [])
        print(f"\n[OK] SELECTED ({len(selected)} candidates) - Moving to next round:")
        for c in selected:
            name = c.get("filename") or c.get("name", "Unknown")
            score = c.get("ats_score") or c.get("score", 0)
            print(f"     - {name}: {score:.1f}/100")

        borderline = classified_results.get("borderline", [])
        print(f"\n[!] BORDERLINE ({len(borderline)} candidates) - Will receive feedback:")
        for c in borderline:
            name = c.get("filename") or c.get("name", "Unknown")
            score = c.get("ats_score") or c.get("score", 0)
            print(f"     - {name}: {score:.1f}/100")

        poor_match = classified_results.get("poor_match", [])
        print(f"\n[X] POOR MATCH ({len(poor_match)} candidates) - No action:")
        for c in poor_match:
            name = c.get("filename") or c.get("name", "Unknown")
            score = c.get("ats_score") or c.get("score", 0)
            print(f"     - {name}: {score:.1f}/100")
        print("=" * 70)


if __name__ == "__main__":
    classifier = ZoneClassifier(selected_threshold=75, borderline_threshold=40)
    candidates = [
        {"id": 1, "name": "Alice", "score": 85},
        {"id": 2, "name": "Bob", "score": 62},
        {"id": 3, "name": "Charlie", "score": 28}
    ]
    results = classifier.batch_classify(candidates)
    classifier.print_classification_report(results)
