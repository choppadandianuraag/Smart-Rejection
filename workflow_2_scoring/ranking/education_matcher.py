"""
Education Hierarchy Matcher
Handles education equivalence (MTech > BTech, etc.)
"""

from typing import Tuple, Optional
from loguru import logger


EDUCATION_HIERARCHY = {
    "phd": {
        "level": 5,
        "aliases": ["phd", "ph.d", "ph d", "doctorate", "doctoral", "doctor of philosophy"],
        "satisfies": ["mtech", "msc", "masters", "btech", "bsc", "bachelors"]
    },
    "mtech": {
        "level": 4,
        "aliases": ["mtech", "m.tech", "m tech", "msc", "m.sc", "m sc",
                   "me", "m.e", "m e", "masters", "master's", "master of technology",
                   "master of science"],
        "satisfies": ["btech", "bsc", "bachelors"]
    },
    "btech": {
        "level": 3,
        "aliases": ["btech", "b.tech", "b tech", "be", "b.e", "b e",
                   "bsc", "b.sc", "b sc", "bachelors", "bachelor's",
                   "bachelor of technology", "bachelor of science", "bachelor of engineering"],
        "satisfies": []
    },
    "diploma": {
        "level": 2,
        "aliases": ["diploma", "polytechnic"],
        "satisfies": []
    }
}


def normalize_education(degree_text: str) -> Tuple[Optional[str], int]:
    """
    Convert degree text to standardized level

    Args:
        degree_text: str (e.g., "M.Tech in CSE", "Bachelor of Technology")

    Returns:
        tuple: (standard_name, level) or (None, 0) if not found
    """
    if not degree_text:
        return None, 0

    degree_lower = degree_text.lower().strip()

    for standard_name, info in EDUCATION_HIERARCHY.items():
        if any(alias in degree_lower for alias in info["aliases"]):
            return standard_name, info["level"]

    return None, 0


def check_education_match(candidate_degree: str, required_degree: str) -> Tuple[bool, float]:
    """
    Check if candidate education satisfies requirement

    Args:
        candidate_degree: str (candidate's degree)
        required_degree: str (required degree from JD)

    Returns:
        tuple: (matches: bool, score: float)
    """
    cand_std, cand_level = normalize_education(candidate_degree)
    req_std, req_level = normalize_education(required_degree)

    if not req_std:
        return True, 100  # No requirement specified

    if not cand_std:
        return False, 0  # Candidate has invalid/unknown degree

    # Exact match
    if cand_std == req_std:
        return True, 100

    # Higher degree satisfies lower requirement
    if req_std in EDUCATION_HIERARCHY[cand_std]["satisfies"]:
        bonus = (cand_level - req_level) * 10
        return True, min(100 + bonus, 120)  # Bonus for overqualification, cap at 120

    # Lower degree doesn't satisfy higher requirement
    return False, (cand_level / req_level) * 70 if req_level > 0 else 0


def get_education_level(degree_text: str) -> int:
    """
    Get numeric level for a degree.

    Args:
        degree_text: str (e.g., "BTech", "Masters")

    Returns:
        int: Level (0-5)
    """
    _, level = normalize_education(degree_text)
    return level


def get_standard_degree_name(degree_text: str) -> Optional[str]:
    """
    Get standardized degree name.

    Args:
        degree_text: str (e.g., "B.Tech in CSE")

    Returns:
        str: Standard name (e.g., "btech") or None
    """
    standard_name, _ = normalize_education(degree_text)
    return standard_name


class EducationMatcher:
    """
    Class-based wrapper for education matching functionality.
    Provides additional features like logging and batch processing.
    """

    def __init__(self):
        logger.info("EducationMatcher initialized")

    def match(self, candidate_degree: str, required_degree: str) -> Tuple[bool, float]:
        """
        Check if candidate education satisfies requirement.

        Args:
            candidate_degree: Candidate's degree
            required_degree: Required degree from JD

        Returns:
            tuple: (matches, score)
        """
        matches, score = check_education_match(candidate_degree, required_degree)
        logger.debug(f"Education match: {candidate_degree} vs {required_degree} -> Match={matches}, Score={score}")
        return matches, score

    def normalize(self, degree_text: str) -> Tuple[Optional[str], int]:
        """Normalize degree text to standard name and level."""
        return normalize_education(degree_text)

    def score_education(self, candidate_edu: str, required_edu: str) -> float:
        """
        Score education match (for integration with existing scoring system).

        Args:
            candidate_edu: Candidate's education level
            required_edu: Required education from JD

        Returns:
            float: Score (0-120)
        """
        _, score = check_education_match(candidate_edu, required_edu)
        return score


# Usage
if __name__ == "__main__":
    # Test cases
    tests = [
        ("MTech in CSE", "BTech required"),
        ("BTech", "MTech required"),
        ("PhD", "BTech required"),
        ("Diploma", "BTech required"),
        ("Bachelor of Technology", "bachelors"),
        ("M.Sc in Computer Science", "BTech"),
        ("B.E. in Electronics", "Bachelor's degree"),
    ]

    print("=" * 60)
    print("EDUCATION HIERARCHY MATCHER - TEST RESULTS")
    print("=" * 60)

    for cand, req in tests:
        matches, score = check_education_match(cand, req)
        status = "PASS" if matches else "FAIL"
        print(f"[{status}] {cand} vs {req}: Score={score:.1f}")

    print("=" * 60)
