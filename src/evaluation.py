import re
import json
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
from pathlib import Path
import numpy as np
from databricks.connect import DatabricksSession

from src.config import PIDConfig
from src.metrics import (
    jaccard_similarity,
    calculate_recall,
    calculate_precision,
    normalized_levenshtein,
    boolean_accuracy,
)
from src.utils import get_spark, get_page_and_tag_files


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


def load_ground_truth_load_sheet(
    spark: DatabricksSession, table_name: str
) -> pd.DataFrame:
    """Load and process ground truth data from spark table of a load sheet.
    Args:
        table_name: Name of the spark table

    Returns:
        DataFrame with ground truth data
    """
    spark = get_spark()
    ground_truth_df = spark.table(table_name).toPandas()
    return ground_truth_df


def load_ground_truth_json(config: PIDConfig) -> pd.DataFrame:
    """Load and process ground truth data from examples directory or volume.

    Args:
        examples_path: Path to examples directory containing JSON files

    Returns:
        DataFrame with ground truth data
    """
    page_files, tag_files = get_page_and_tag_files(config.parse.example_path)

    ground_truth_series = []
    for page_file in page_files:
        with open(page_file, "r") as f:
            metadata = pd.Series(json.load(f))

        metadata["unique_key"] = page_file.stem

        page_tag_files = [
            file for file in tag_files if file.stem.startswith(metadata["unique_key"])
        ]

        combined_tags = {}
        for tag_file in page_tag_files:
            with open(tag_file, "r") as f:
                tag_dict = json.load(f)
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

        # Create combined columns for high-level evaluation
        equipment_tags = tags.get("equipment_tags", []) or []
        line_tags = tags.get("line_tags", []) or []
        tags["combined_tags"] = list(set(equipment_tags + line_tags))

        incoming_streams = tags.get("incoming_streams", []) or []
        outgoing_streams = tags.get("outgoing_streams", []) or []
        tags["combined_streams"] = list(set(incoming_streams + outgoing_streams))

        ground_truth_series.append(pd.concat([metadata, tags]).to_dict())

    return pd.DataFrame(ground_truth_series)


def load_ground_truth(
    config: PIDConfig, spark: DatabricksSession = None
) -> pd.DataFrame:
    """
    Load ground truth data based on configuration.

    Args:
        config: PIDConfig object with evaluation settings
        spark: Optional DatabricksSession for load_sheet mode

    Returns:
        DataFrame with ground truth data

    Raises:
        ValueError: If configuration is invalid or required parameters missing
    """
    if config.evaluate.ground_truth_source == "json":
        if not config.evaluate.ground_truth_json_path:
            raise ValueError("ground_truth_json_path is required when source is 'json'")

        # Create a temporary config with the JSON path set as example_path
        # This maintains backward compatibility with load_ground_truth_json
        temp_config = config.model_copy(deep=True)
        temp_config.parse.example_path = config.evaluate.ground_truth_json_path

        return load_ground_truth_json(temp_config)

    elif config.evaluate.ground_truth_source == "load_sheet":
        if not config.evaluate.ground_truth_table:
            raise ValueError(
                "ground_truth_table is required when source is 'load_sheet'"
            )
        if spark is None:
            spark = get_spark()

        return load_ground_truth_load_sheet(spark, config.evaluate.ground_truth_table)

    else:
        raise ValueError(
            f"Invalid ground_truth_source: {config.evaluate.ground_truth_source}. "
            "Must be 'json' or 'load_sheet'"
        )


def load_parsed_metadata(
    spark: Optional[DatabricksSession], config: PIDConfig
) -> pd.DataFrame:
    """Load and process parsed metadata based on config source.

    Args:
        spark: Spark session (optional, required only for spark mode)
        config: PIDConfig object

    Returns:
        DataFrame with parsed metadata
    """
    if config.evaluate.metadata_tag_source == "spark":
        if spark is None:
            raise ValueError(
                "Spark session is required when metadata_tag_source is 'spark'"
            )
        return load_parsed_metadata_spark(spark, config)
    else:
        return load_parsed_metadata_local(config)


def load_parsed_metadata_spark(
    spark: DatabricksSession, config: PIDConfig
) -> pd.DataFrame:
    """Load and process parsed metadata from spark table.

    Args:
        config: PIDConfig object

    Returns:
        DataFrame with parsed metadata
    """
    output_metadata_raw = spark.table(
        f"{config.catalog}.{config.schema}.{config.parse.metadata_table_name}"
    ).toPandas()
    output_metadata_raw["parsed_metadata"] = output_metadata_raw.parsed_metadata.apply(
        lambda x: json.loads(x)
    )
    output_metadata = pd.json_normalize(output_metadata_raw.parsed_metadata)
    return output_metadata


def load_parsed_tags_spark(spark: DatabricksSession, config: PIDConfig) -> pd.DataFrame:
    """Load and process parsed tags fromSspark.

    Args:
        config: PIDConfig object

    Returns:
        DataFrame with parsed tags
    """
    output_tags = spark.table(
        f"{config.catalog}.{config.schema}.{config.parse.tags_table_name}"
    ).toPandas()
    return output_tags


def load_parsed_metadata_local(config: PIDConfig) -> pd.DataFrame:
    """Load and process parsed data from parquet files.

    Args:
        local_tables_path: Path to directory containing parquet files

    Returns:
        DataFrame with parsed data
    """
    output = pd.read_parquet(
        Path(config.parse.local_tables_path)
        / f"{config.parse.metadata_table_name}.parquet"
    )
    try:
        output["parsed_metadata"] = output.parsed_metadata.apply(
            lambda x: json.loads(x)
        )
    except:
        pass

    metadata = pd.json_normalize(output.parsed_metadata)
    return pd.concat([output, metadata], axis=1)


def load_parsed_tags_local(config: PIDConfig) -> pd.DataFrame:
    """Load and process parsed tags from local parquet files.

    Args:
        config: PIDConfig object

    Returns:
        DataFrame with parsed tags
    """
    output_tags = pd.read_parquet(
        Path(config.parse.local_tables_path) / f"{config.parse.tags_table_name}.parquet"
    )
    return output_tags


def load_parsed_tags(
    spark: Optional[DatabricksSession], config: PIDConfig
) -> pd.DataFrame:
    """Load and process parsed tags based on config source.

    Args:
        spark: Spark session (optional, required only for spark mode)
        config: PIDConfig object

    Returns:
        DataFrame with parsed tags
    """
    if config.evaluate.metadata_tag_source == "spark":
        if spark is None:
            raise ValueError(
                "Spark session is required when metadata_tag_source is 'spark'"
            )
        return load_parsed_tags_spark(spark, config)
    else:
        return load_parsed_tags_local(config)


def combine_metadata_and_tags(
    metadata_df: pd.DataFrame, tags_df: pd.DataFrame
) -> pd.DataFrame:
    """Combine metadata and tags into a single dataframe.

    Args:
        metadata_df: DataFrame with parsed metadata
        tags_df: DataFrame with parsed tags

    Returns:
        DataFrame with combined metadata and tags
    """
    tags_df = tags_df.copy()
    combined_tags = {}

    tags_df["page_unique_key"] = tags_df["unique_key"].str.extract(r"(.+_p\d+)")[0]

    for unique_key in tags_df["page_unique_key"].unique():
        key_rows = tags_df[tags_df["page_unique_key"] == unique_key]
        combined_tags_for_key = {}
        for _, row in key_rows.iterrows():
            tag_dict = json.loads(row["parsed_tag"])
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

        combined_tags[unique_key] = cleaned_combined_tags

    # Convert to DataFrame
    combined_tags_df = pd.DataFrame(combined_tags).T
    combined_tags_df.index.name = "page_unique_key"
    combined_tags_df = combined_tags_df.reset_index()

    # Merge metadata with combined tags
    parsed_df = metadata_df.merge(
        combined_tags_df, left_on="unique_key", right_on="page_unique_key", how="left"
    ).drop(columns=["page_unique_key"])

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
