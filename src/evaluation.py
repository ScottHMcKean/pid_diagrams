import re
from typing import Dict, List, Any, Tuple
import pandas as pd
from Levenshtein import distance as levenshtein_distance


def clean_pid_tags(tags_dict: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Clean P&ID tags dictionary by applying formatting rules.

    Args:
        tags_dict: Dictionary with tag categories as keys and lists of tags as values

    Returns:
        Cleaned dictionary with filtered tags

    Rules:
        - Equipment/line tags must contain dashes (e.g., "P-101", "V-201")
        - Stream tags must contain dots (e.g., "1.01", "2.15")
        - Remove any tags containing brackets (), (1), etc.
    """

    cleaned_dict = {}

    for category, tag_list in tags_dict.items():
        cleaned_tags = []

        # Ensure tag_list is actually a list and handle None cases
        if tag_list is None:
            cleaned_tags = []
        elif not isinstance(tag_list, (list, tuple, set)):
            # Handle single values by converting to list
            tag_list = [tag_list]

        if category in ["equipment_tags", "line_tags"]:
            cleaned_tags = [
                re.sub(r"\([^)]*\)", "", str(tag)).strip()
                for tag in tag_list
                if isinstance(tag, (str, int, float)) and "-" in str(tag)
            ]
        elif category in ["incoming_streams", "outgoing_streams"]:
            cleaned_tags = [
                re.sub(r"\([^)]*\)", "", str(tag)).strip()
                for tag in tag_list
                if isinstance(tag, (str, int, float)) and "." in str(tag)
            ]
        else:
            # For other categories (like locations, legacy_numbers, moc_numbers),
            # ensure we have a clean list of strings
            cleaned_tags = [
                str(tag).strip()
                for tag in tag_list
                if tag is not None and not isinstance(tag, (list, dict))
            ]

        cleaned_dict[category] = cleaned_tags

    return cleaned_dict


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


def load_ground_truth_data(examples_path: str) -> pd.DataFrame:
    """Load and process ground truth data from examples directory.

    Args:
        examples_path: Path to examples directory containing JSON files

    Returns:
        DataFrame with ground truth data
    """
    from pathlib import Path
    import json
    import numpy as np

    examples_path = Path(examples_path)
    all_files = list(examples_path.glob("*.json"))
    all_tag_files = list(examples_path.glob("*_t*.json"))
    metadata_files = list(set(all_files) - set(all_tag_files))

    ground_truth_series = []
    for metadata_file in metadata_files:
        with open(metadata_file, "r") as f:
            metadata = pd.Series(json.load(f))

        metadata["unique_key"] = metadata_file.stem

        # Find all tag json files matching the unique key
        tag_files = [
            file
            for file in all_tag_files
            if file.stem.startswith(metadata["unique_key"])
        ]

        # Initialize a dictionary to store combined unique values
        combined_tags = {}

        # Iterate through all tag files and combine dictionaries
        for tag_file in tag_files:
            with open(tag_file, "r") as f:
                tag_dict = json.load(f)

            # Combine dictionaries using set operations
            for key, value in tag_dict.items():
                if isinstance(value, np.ndarray):
                    value = list(value)

                if key not in combined_tags:
                    combined_tags[key] = set()

                if isinstance(value, (list, tuple, set)):
                    combined_tags[key].update(value)
                else:
                    combined_tags[key].add(value)

        # Convert sets back to lists for easier handling
        combined_tags = {
            key: list(value_set) for key, value_set in combined_tags.items()
        }

        # Apply cleaning to the combined tags
        cleaned_combined_tags = clean_pid_tags(combined_tags)
        tags = pd.Series(cleaned_combined_tags)

        # Combine moc_numbers from metadata and tags
        metadata["moc_numbers"] = list(
            set(metadata["moc_numbers"] + tags["moc_numbers"])
        )

        # Create combined columns for high-level evaluation
        equipment_tags = tags.get("equipment_tags", []) or []
        line_tags = tags.get("line_tags", []) or []
        tags["combined_tags"] = list(set(equipment_tags + line_tags))

        incoming_streams = tags.get("incoming_streams", []) or []
        outgoing_streams = tags.get("outgoing_streams", []) or []
        tags["combined_streams"] = list(set(incoming_streams + outgoing_streams))

        ground_truth_series.append(pd.concat([metadata, tags]).to_dict())

    return pd.DataFrame(ground_truth_series)


def load_parsed_data(local_tables_path: str) -> pd.DataFrame:
    """Load and process parsed data from parquet files.

    Args:
        local_tables_path: Path to directory containing parquet files

    Returns:
        DataFrame with parsed data
    """
    from pathlib import Path
    import numpy as np

    local_tables_path = Path(local_tables_path)

    # Load metadata
    output_metadata = pd.read_parquet(local_tables_path / "metadata_results.parquet")
    output_metadata_df = pd.json_normalize(output_metadata.parsed_metadata)
    output_metadata_df["unique_key"] = output_metadata.unique_key.str[0:-3].values

    # Load tags
    output_tags = pd.read_parquet(local_tables_path / "tag_results.parquet")
    output_tags["unique_key"] = output_tags.unique_key.str[0:-3].values

    # Group by unique_key and combine tag dictionaries
    combined_output_tags = {}

    for unique_key in output_tags["unique_key"].unique():
        key_rows = output_tags[output_tags["unique_key"] == unique_key]
        combined_tags_for_key = {}

        for _, row in key_rows.iterrows():
            tag_dict = row["parsed_tag"]

            for key, value in tag_dict.items():
                if key not in combined_tags_for_key:
                    combined_tags_for_key[key] = set()

                if isinstance(value, np.ndarray):
                    value = list(value)

                if isinstance(value, (list, tuple, set)):
                    combined_tags_for_key[key].update(value)
                else:
                    combined_tags_for_key[key].add(value)

        # Convert sets back to lists and clean
        combined_tags_for_key = {
            key: list(value_set) for key, value_set in combined_tags_for_key.items()
        }
        cleaned_combined_tags = clean_pid_tags(combined_tags_for_key)

        # Create combined columns for high-level evaluation
        equipment_tags = cleaned_combined_tags.get("equipment_tags", []) or []
        line_tags = cleaned_combined_tags.get("line_tags", []) or []
        cleaned_combined_tags["combined_tags"] = list(set(equipment_tags + line_tags))

        incoming_streams = cleaned_combined_tags.get("incoming_streams", []) or []
        outgoing_streams = cleaned_combined_tags.get("outgoing_streams", []) or []
        cleaned_combined_tags["combined_streams"] = list(
            set(incoming_streams + outgoing_streams)
        )

        combined_output_tags[unique_key] = cleaned_combined_tags

    # Convert to DataFrame
    output_tags_df = pd.DataFrame(combined_output_tags).T
    output_tags_df.index.name = "unique_key"
    output_tags_df = output_tags_df.reset_index()

    # Merge metadata and tags
    merged_df = output_metadata_df.merge(
        output_tags_df, on="unique_key", how="left", suffixes=("_metadata", "_tags")
    )

    # Combine moc_numbers arrays
    def combine_moc_arrays(row):
        meta_moc = list(row["moc_numbers_metadata"])
        tags_moc = list(row["moc_numbers_tags"])
        return list(set(list(meta_moc) + list(tags_moc)))

    merged_df["moc_numbers"] = merged_df.apply(combine_moc_arrays, axis=1)
    parsed_df = merged_df.drop(columns=["moc_numbers_metadata", "moc_numbers_tags"])

    return parsed_df


def evaluate_parsed_vs_ground_truth(
    ground_truth_df: pd.DataFrame,
    parsed_df: pd.DataFrame,
    string_columns: List[str],
    boolean_columns: List[str],
) -> pd.DataFrame:
    """Evaluate similarity between ground truth and parsed dataframes with high-level metrics.

    This function uses pre-combined tags and streams columns for high-level reporting.

    Args:
        ground_truth_df: DataFrame with ground truth values (must include combined_tags, combined_streams)
        parsed_df: DataFrame with parsed values (must include combined_tags, combined_streams)
        string_columns: List of column names containing strings
        boolean_columns: List of column names containing booleans

    Returns:
        DataFrame with high-level similarity scores for each unique_key
    """
    # Merge dataframes on unique_key for comparison
    evaluation_df = ground_truth_df.merge(
        parsed_df, on="unique_key", suffixes=("_gt", "_parsed")
    )

    similarities = []

    for _, row in evaluation_df.iterrows():
        result = {"unique_key": row["unique_key"]}

        # Combined tags Jaccard similarity (using pre-combined columns)
        combined_tags_gt = row.get("combined_tags_gt", []) or []
        combined_tags_parsed = row.get("combined_tags_parsed", []) or []
        result["tags_jaccard"] = jaccard_similarity(
            combined_tags_gt, combined_tags_parsed
        )
        result["tags_recall"] = calculate_recall(combined_tags_gt, combined_tags_parsed)
        result["tags_precision"] = calculate_precision(
            combined_tags_gt, combined_tags_parsed
        )

        # Combined streams Jaccard similarity (using pre-combined columns)
        combined_streams_gt = row.get("combined_streams_gt", []) or []
        combined_streams_parsed = row.get("combined_streams_parsed", []) or []
        result["streams_jaccard"] = jaccard_similarity(
            combined_streams_gt, combined_streams_parsed
        )
        result["streams_recall"] = calculate_recall(
            combined_streams_gt, combined_streams_parsed
        )
        result["streams_precision"] = calculate_precision(
            combined_streams_gt, combined_streams_parsed
        )

        # Normalized Levenshtein for string columns
        for col in string_columns:
            if col + "_gt" in row.index and col + "_parsed" in row.index:
                gt_val = str(row[col + "_gt"]) if row[col + "_gt"] is not None else ""
                parsed_val = (
                    str(row[col + "_parsed"])
                    if row[col + "_parsed"] is not None
                    else ""
                )
                result[f"{col}_levenshtein"] = normalized_levenshtein(
                    gt_val, parsed_val
                )

        # Boolean columns
        for col in boolean_columns:
            if col + "_gt" in row.index and col + "_parsed" in row.index:
                gt_val = row[col + "_gt"]
                parsed_val = row[col + "_parsed"]
                result[f"{col}_boolean"] = boolean_accuracy(gt_val, parsed_val)

        similarities.append(result)

    return pd.DataFrame(similarities)
