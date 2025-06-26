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
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.evaluation import clean_pid_tags

# COMMAND ----------

from src.evaluation import load_ground_truth_data, load_parsed_data

# Load ground truth and parsed data using abstracted functions
ground_truth_df = load_ground_truth_data("examples")
parsed_df = load_parsed_data("local_tables")

# Assert column alignment
assert sorted(ground_truth_df.columns) == sorted(
    parsed_df.columns
), "Column alignment mismatch between ground truth and parsed data"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation Metrics

# COMMAND ----------

from src.evaluation import evaluate_parsed_vs_ground_truth

# Define columns to evaluate
string_columns = ["drawing_name", "title", "revision", "date", "organization"]
boolean_columns = ["has_stamp"]

# Calculate high-level similarities for reporting
similarity_df = evaluate_parsed_vs_ground_truth(
    ground_truth_df, parsed_df, string_columns, boolean_columns
)

# Summary statistics
similarity_df.drop("unique_key", axis=1).agg(
    ["mean", "max", "min", "count", "std"]
).T.round(2)
