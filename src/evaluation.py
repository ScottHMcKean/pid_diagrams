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

        if category in ["equipment_tags", "line_tags"]:
            cleaned_tags = [
                re.sub(r"\([^)]*\)", "", tag).strip() for tag in tag_list if "-" in tag
            ]
        elif category in ["incoming_streams", "outgoing_streams"]:
            cleaned_tags = [
                re.sub(r"\([^)]*\)", "", tag).strip() for tag in tag_list if "." in tag
            ]
        else:
            cleaned_tags = tag_list

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


def boolean_accuracy(val1: Any, val2: Any) -> bool:
    """Compare two boolean values for exact match.

    Args:
        val1: First boolean value
        val2: Second boolean value

    Returns:
        True if values match, False otherwise
    """
    val1 = val1 if val1 is not None else False
    val2 = val2 if val2 is not None else False
    return val1 == val2


def evaluate_dataframes(
    ground_truth_df: pd.DataFrame,
    parsed_df: pd.DataFrame,
    array_columns: List[str],
    string_columns: List[str],
    boolean_columns: List[str],
) -> pd.DataFrame:
    """Evaluate similarity between ground truth and parsed dataframes.

    Args:
        ground_truth_df: DataFrame with ground truth values
        parsed_df: DataFrame with parsed values
        array_columns: List of column names containing arrays
        string_columns: List of column names containing strings
        boolean_columns: List of column names containing booleans

    Returns:
        DataFrame with similarity scores for each unique_key
    """
    # Merge dataframes on unique_key for comparison
    evaluation_df = ground_truth_df.merge(
        parsed_df, on="unique_key", suffixes=("_gt", "_parsed")
    )

    similarities = []

    for _, row in evaluation_df.iterrows():
        result = {"unique_key": row["unique_key"]}

        # Jaccard similarity for array columns
        for col in array_columns:
            if col + "_gt" in row.index and col + "_parsed" in row.index:
                gt_val = row[col + "_gt"] if row[col + "_gt"] is not None else []
                parsed_val = (
                    row[col + "_parsed"] if row[col + "_parsed"] is not None else []
                )
                result[f"{col}_jaccard"] = jaccard_similarity(gt_val, parsed_val)

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


def get_evaluation_summary(similarity_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Generate summary statistics for evaluation results.

    Args:
        similarity_df: DataFrame with similarity scores

    Returns:
        Dictionary with summary statistics for each metric type
    """
    summary = {}

    # Jaccard similarity summary
    jaccard_cols = [col for col in similarity_df.columns if col.endswith("_jaccard")]
    if jaccard_cols:
        summary["jaccard"] = similarity_df[jaccard_cols].describe()

    # Levenshtein similarity summary
    levenshtein_cols = [
        col for col in similarity_df.columns if col.endswith("_levenshtein")
    ]
    if levenshtein_cols:
        summary["levenshtein"] = similarity_df[levenshtein_cols].describe()

    # Boolean accuracy summary
    boolean_cols = [col for col in similarity_df.columns if col.endswith("_boolean")]
    if boolean_cols:
        summary["boolean"] = similarity_df[boolean_cols].mean().to_frame("accuracy")

    return summary


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
