from typing import List, Any
from Levenshtein import distance as levenshtein_distance


def jaccard_similarity(set1: List[Any], set2: List[Any]) -> float:
    """Calculate Jaccard similarity between two lists.

    Args:
        set1: First list of items
        set2: Second list of items

    Returns:
        Jaccard similarity score between 0 and 1
    """
    if not set1 and not set2:
        return 1.0
    set1 = set(set1) if set1 else set()
    set2 = set(set2) if set2 else set()
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def normalized_levenshtein(str1: str, str2: str) -> float:
    """Calculate normalized Levenshtein similarity between two strings.

    Args:
        str1: First string
        str2: Second string

    Returns:
        Normalized similarity score between 0 and 1
    """
    if not str1 and not str2:
        return 1.0
    str1 = str1 or ""
    str2 = str2 or ""
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 1.0
    return 1 - (levenshtein_distance(str1, str2) / max_len)


def boolean_accuracy(val1: Any, val2: Any) -> int:
    """Compare two boolean values for exact match.

    Args:
        val1: First boolean value
        val2: Second boolean value

    Returns:
        1 if values match, 0 otherwise
    """
    val1 = val1 if val1 is not None else False
    val2 = val2 if val2 is not None else False
    return 1 if val1 == val2 else 0


def calculate_recall(ground_truth_list: List[Any], parsed_list: List[Any]) -> float:
    """Calculate recall: what percentage of ground truth labels were correctly identified.

    Recall = True Positives / (True Positives + False Negatives)

    Args:
        ground_truth_list: Ground truth labels
        parsed_list: Parsed labels

    Returns:
        Recall score between 0 and 1, where 1 means all ground truth labels were found
    """
    if not ground_truth_list:
        return 1.0  # Perfect recall if no ground truth to find

    ground_truth_set = set(ground_truth_list) if ground_truth_list else set()
    parsed_set = set(parsed_list) if parsed_list else set()

    # True Positives: ground truth labels that were found
    true_positives = len(ground_truth_set.intersection(parsed_set))
    total_ground_truth = len(ground_truth_set)

    return true_positives / total_ground_truth if total_ground_truth > 0 else 1.0


def calculate_precision(ground_truth_list: List[Any], parsed_list: List[Any]) -> float:
    """Calculate precision: what percentage of parsed labels were correct.

    Precision = True Positives / (True Positives + False Positives)

    Args:
        ground_truth_list: Ground truth labels
        parsed_list: Parsed labels

    Returns:
        Precision score between 0 and 1, where 1 means all parsed labels were correct
    """
    if not parsed_list:
        return 1.0  # Perfect precision if nothing was parsed (no false positives)

    ground_truth_set = set(ground_truth_list) if ground_truth_list else set()
    parsed_set = set(parsed_list) if parsed_list else set()

    # True Positives: parsed labels that were in ground truth
    true_positives = len(ground_truth_set.intersection(parsed_set))
    total_parsed = len(parsed_set)

    return true_positives / total_parsed if total_parsed > 0 else 1.0
