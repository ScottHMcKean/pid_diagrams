# Databricks notebook source
# MAGIC %md
# MAGIC # 3 - Evaluation
# MAGIC
# MAGIC This notebook uses the parsing outputs to compare against the ground truth labels.

# COMMAND ----------

# MAGIC %pip install -U --quiet -r requirements.txt
# MAGIC %restart_python

# COMMAND ----------

# 1. Pull grount truth labels
# 2. Write function to combine tags
# 3. Load parsing outputs
# 4. Compare parsing outputs to ground truth labels

# COMMAND ----------

import pandas as pd
from pathlib import Path

from src.config import load_config
from src.utils import get_spark
from src.evaluation import (
    load_ground_truth_json,
    load_parsed_metadata,
    load_parsed_tags,
    combine_metadata_and_tags,
    evaluate_parsed_vs_ground_truth,
)

# COMMAND ----------

spark = get_spark()
config = load_config("config.yaml")

# Load ground truth and parsed data using abstracted functions
ground_truth_df = load_ground_truth_json(config)
metadata_df = load_parsed_metadata(spark, config)
tags_df = load_parsed_tags(spark, config)
parsed_df = combine_metadata_and_tags(metadata_df, tags_df)

# Assert column alignment
assert sorted(ground_truth_df.columns) == sorted(
    parsed_df.columns
), "Column alignment mismatch between ground truth and parsed data"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation Metrics

# COMMAND ----------

# Define columns to evaluate
string_columns = ["drawing_name", "title", "revision", "date", "organization"]
boolean_columns = ["has_stamp"]

# Calculate metrics
metrics_df = evaluate_parsed_vs_ground_truth(
    ground_truth_df, parsed_df, string_columns, boolean_columns
)

# Summary statistics
metrics_df.drop("unique_key", axis=1).agg(
    ["mean", "max", "min", "count", "std"]
).T.round(2)
