# Databricks notebook source
# MAGIC %md
# MAGIC # 3 - Evaluation
# MAGIC
# MAGIC This notebook uses the parsing outputs to compare against the ground truth labels.

# COMMAND ----------

# MAGIC %pip install uv

# COMMAND ----------

# MAGIC %sh uv pip install .

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# 1. Pull grount truth labels
# 2. Write function to combine tags
# 3. Load parsing outputs
# 4. Compare parsing outputs to ground truth labels

# COMMAND ----------

import sys
sys.path.append('.')

# COMMAND ----------

import pandas as pd
from pathlib import Path

from src.config import load_config
from src.utils import get_spark
from src.evaluation import (
    load_ground_truth_json,
    load_ground_truth_load_sheet,
    load_parsed_metadata,
    load_parsed_tags,
    combine_metadata_and_tags,
    evaluate_parsed_vs_ground_truth,
)

# COMMAND ----------

spark = get_spark()
config = load_config("config.yaml")

# Load ground truth and parsed data using abstracted functions
# ground_truth_df = load_ground_truth_json(config)
# TODO: add filename to ground truth
ground_truth_df = load_ground_truth_load_sheet(spark, 'shm.pid.alb_load_sheet')
ground_truth_df['filename'] = ground_truth_df.closest_filename.str.replace(".pdf","")

metadata_df = load_parsed_metadata(spark, config)
tags_df = load_parsed_tags(spark, config)

# TODO: Clean up col names
parsed_df = combine_metadata_and_tags(metadata_df, tags_df).rename(columns={'legacy_number':'legacy_numbers'})

# ground_truth_df = ground_truth_df[parsed_df.columns.tolist()]

# Assert column alignment
assert sorted(ground_truth_df.columns) == sorted(
    parsed_df.columns
), "Column alignment mismatch between ground truth and parsed data"

# COMMAND ----------

metadata_df

# COMMAND ----------

parsed_df.merge(tags_df, on='unique_key', how='left')

# COMMAND ----------

ground_truth_df.closest_filename.str.replace(".pdf","")

# COMMAND ----------

ground_truth_df

# COMMAND ----------

tags_df

# COMMAND ----------

ground_truth_df

# COMMAND ----------

ground_truth_df.drop('unique_key', axis=1).merge(tags_df[['filename','unique_key']], on='filename')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation Metrics

# COMMAND ----------

parsed_df

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
