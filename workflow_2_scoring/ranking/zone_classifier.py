"""
Zone Classification Module
Divides candidates into 3 zones based on PERCENTILE ranking

Zones:
- SELECTED: Top 10% of candidates (move to next round)
- BORDERLINE: Next 40% of candidates (send feedback email)
- REJECTED: Bottom 50% of candidates (no action)
"""

from typing import Dict, List, Any
from loguru import logger


class ZoneClassifier:
    """
    Classifies candidates into zones based on PERCENTILE ranking.

    Zones (percentile-based):
        - SELECTED: Top 10% of candidates → move to interview
        - BORDERLINE: Next 40% of candidates → send constructive feedback
        - REJECTED: Bottom 50% of candidates → no actionable feedback
    """

    def __init__(
        self,
        selected_percentile: float = 10.0,
        borderline_percentile: float = 40.0
    ):
        """
        Initialize the percentile-based classifier.

        Args:
            selected_percentile: Top X% of candidates are SELECTED (default: 10%)
            borderline_percentile: Next Y% of candidates are BORDERLINE (default: 40%)
            (Remaining candidates are REJECTED: 100 - 10 - 40 = 50%)
        """
        self.selected_percentile = selected_percentile
        self.borderline_percentile = borderline_percentile
        self.rejected_percentile = 100.0 - selected_percentile - borderline_percentile
        logger.info(
            f"ZoneClassifier initialized (percentile-based): "
            f"Top {selected_percentile}% SELECTED, "
            f"Next {borderline_percentile}% BORDERLINE, "
            f"Bottom {self.rejected_percentile}% REJECTED"
        )

    def batch_classify(
        self,
        candidates_with_scores: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Classify candidates using percentile-based ranking.

        Args:
            candidates_with_scores: List of candidates with 'score' or 'final_score' key

        Returns:
            Dict with 'selected', 'borderline', 'rejected' lists
        """
        results = {"selected": [], "borderline": [], "rejected": []}

        if not candidates_with_scores:
            results["summary"] = self._create_summary(results, 0)
            return results

        # Sort candidates by score (descending)
        sorted_candidates = sorted(
            candidates_with_scores,
            key=lambda x: x.get("score") or x.get("final_score") or x.get("ats_score", 0),
            reverse=True
        )

        total = len(sorted_candidates)

        # Calculate cutoff indices based on percentiles
        selected_cutoff = max(1, int(total * (self.selected_percentile / 100)))
        borderline_cutoff = selected_cutoff + max(1, int(total * (self.borderline_percentile / 100)))

        # Handle edge cases for small candidate pools
        if total <= 2:
            # If 1-2 candidates: top one is selected, rest are borderline
            selected_cutoff = 1
            borderline_cutoff = total
        elif total <= 5:
            # If 3-5 candidates: at least 1 selected, at least 1 borderline
            selected_cutoff = max(1, int(total * 0.1))
            borderline_cutoff = max(selected_cutoff + 1, int(total * 0.5))

        logger.info(
            f"Total: {total} candidates | "
            f"Selected: top {selected_cutoff} | "
            f"Borderline: next {borderline_cutoff - selected_cutoff} | "
            f"Rejected: bottom {total - borderline_cutoff}"
        )

        # Classify each candidate based on their rank
        for rank, candidate in enumerate(sorted_candidates):
            score = candidate.get("score") or candidate.get("final_score") or candidate.get("ats_score", 0)

            candidate_with_zone = candidate.copy()

            if rank < selected_cutoff:
                # Top 10%
                candidate_with_zone["zone"] = "SELECTED"
                candidate_with_zone["action"] = "MOVE_TO_NEXT_ROUND"
                candidate_with_zone["send_feedback"] = False
                candidate_with_zone["zone_explanation"] = f"Top {self.selected_percentile}% (Rank {rank + 1}/{total})"
                results["selected"].append(candidate_with_zone)
            elif rank < borderline_cutoff:
                # Next 40%
                candidate_with_zone["zone"] = "BORDERLINE"
                candidate_with_zone["action"] = "SEND_FEEDBACK_EMAIL"
                candidate_with_zone["send_feedback"] = True
                candidate_with_zone["zone_explanation"] = f"Next {self.borderline_percentile}% (Rank {rank + 1}/{total})"
                results["borderline"].append(candidate_with_zone)
            else:
                # Bottom 50%
                candidate_with_zone["zone"] = "REJECTED"
                candidate_with_zone["action"] = "NO_ACTION"
                candidate_with_zone["send_feedback"] = False
                candidate_with_zone["zone_explanation"] = f"Bottom {self.rejected_percentile}% (Rank {rank + 1}/{total})"
                results["rejected"].append(candidate_with_zone)

        results["summary"] = self._create_summary(results, total)

        logger.info(
            f"Classification complete: "
            f"{len(results['selected'])} selected, "
            f"{len(results['borderline'])} borderline, "
            f"{len(results['rejected'])} rejected"
        )

        return results

    def _create_summary(self, results: Dict, total: int) -> Dict[str, Any]:
        """Create summary statistics."""
        return {
            "total": total,
            "selected_count": len(results.get("selected", [])),
            "borderline_count": len(results.get("borderline", [])),
            "rejected_count": len(results.get("rejected", [])),
            "selected_percentile": self.selected_percentile,
            "borderline_percentile": self.borderline_percentile,
            "rejected_percentile": self.rejected_percentile,
            "classification_method": "percentile"
        }

    def classify(self, final_score: float) -> Dict[str, Any]:
        """
        Single candidate classification (legacy method).
        Note: For accurate percentile-based classification, use batch_classify().
        This method is kept for backwards compatibility but doesn't use percentiles.
        """
        logger.warning(
            "classify() called for single candidate - "
            "percentile classification requires batch_classify() with all candidates"
        )
        return {
            "zone": "BORDERLINE",
            "action": "SEND_FEEDBACK_EMAIL",
            "send_feedback": True,
            "explanation": "Single candidate - use batch_classify for percentile ranking"
        }

    def get_feedback_candidates(
        self,
        classified_results: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Get borderline candidates who should receive feedback emails."""
        return classified_results.get("borderline", [])

    def get_summary_stats(
        self,
        classified_results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Generate summary statistics from classification results."""
        total = (
            len(classified_results.get("selected", [])) +
            len(classified_results.get("borderline", [])) +
            len(classified_results.get("rejected", []))
        )

        if total == 0:
            return {
                "total_candidates": 0,
                "selected_count": 0,
                "borderline_count": 0,
                "rejected_count": 0,
                "selected_percentage": 0.0,
                "borderline_percentage": 0.0,
                "rejected_percentage": 0.0,
                "feedback_emails_to_send": 0
            }

        return {
            "total_candidates": total,
            "selected_count": len(classified_results.get("selected", [])),
            "borderline_count": len(classified_results.get("borderline", [])),
            "rejected_count": len(classified_results.get("rejected", [])),
            "selected_percentage": round(len(classified_results.get("selected", [])) / total * 100, 1),
            "borderline_percentage": round(len(classified_results.get("borderline", [])) / total * 100, 1),
            "rejected_percentage": round(len(classified_results.get("rejected", [])) / total * 100, 1),
            "feedback_emails_to_send": len(classified_results.get("borderline", []))
        }

    def print_classification_report(
        self,
        classified_results: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """Print formatted zone classification report."""
        summary = classified_results.get("summary", {})
        total = summary.get("total", 0)

        print("\n" + "=" * 70)
        print("[ZONE] CANDIDATE CLASSIFICATION RESULTS (Percentile-Based)")
        print("=" * 70)
        print(
            f"Distribution: Top {self.selected_percentile}% SELECTED | "
            f"Next {self.borderline_percentile}% BORDERLINE | "
            f"Bottom {self.rejected_percentile}% REJECTED"
        )
        print("-" * 70)

        # SELECTED
        selected = classified_results.get("selected", [])
        print(f"\n[OK] SELECTED ({len(selected)} candidates) - Top {self.selected_percentile}% → Moving to next round:")
        for c in selected:
            name = c.get("filename") or c.get("name", "Unknown")
            score = c.get("final_score") or c.get("score") or c.get("ats_score", 0)
            print(f"     - {name}: {score:.1f}/100")

        # BORDERLINE
        borderline = classified_results.get("borderline", [])
        print(f"\n[!] BORDERLINE ({len(borderline)} candidates) - Next {self.borderline_percentile}% → Will receive feedback:")
        for c in borderline:
            name = c.get("filename") or c.get("name", "Unknown")
            score = c.get("final_score") or c.get("score") or c.get("ats_score", 0)
            print(f"     - {name}: {score:.1f}/100")

        # REJECTED
        rejected = classified_results.get("rejected", [])
        print(f"\n[X] REJECTED ({len(rejected)} candidates) - Bottom {self.rejected_percentile}% → No action:")
        for c in rejected:
            name = c.get("filename") or c.get("name", "Unknown")
            score = c.get("final_score") or c.get("score") or c.get("ats_score", 0)
            print(f"     - {name}: {score:.1f}/100")

        print("=" * 70)


# Backwards compatibility alias
def classify_by_threshold(score: float, selected_threshold: float = 75.0, borderline_threshold: float = 40.0) -> str:
    """
    Legacy threshold-based classification (for single scores).
    DEPRECATED: Use ZoneClassifier.batch_classify() for percentile-based classification.
    """
    if score >= selected_threshold:
        return "SELECTED"
    elif score >= borderline_threshold:
        return "BORDERLINE"
    return "REJECTED"


if __name__ == "__main__":
    # Test with 10 candidates
    classifier = ZoneClassifier(selected_percentile=10.0, borderline_percentile=40.0)

    candidates = [
        {"id": 1, "name": "Alice", "score": 92},
        {"id": 2, "name": "Bob", "score": 88},
        {"id": 3, "name": "Charlie", "score": 75},
        {"id": 4, "name": "Diana", "score": 70},
        {"id": 5, "name": "Eve", "score": 65},
        {"id": 6, "name": "Frank", "score": 58},
        {"id": 7, "name": "Grace", "score": 52},
        {"id": 8, "name": "Henry", "score": 45},
        {"id": 9, "name": "Ivy", "score": 38},
        {"id": 10, "name": "Jack", "score": 25},
    ]

    print(f"\nTesting with {len(candidates)} candidates:")
    print("Expected: 1 selected (10%), 4 borderline (40%), 5 rejected (50%)")

    results = classifier.batch_classify(candidates)
    classifier.print_classification_report(results)
