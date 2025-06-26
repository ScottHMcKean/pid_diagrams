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

# Verify column alignment
print(
    f"Column alignment: {sorted(ground_truth_df.columns) == sorted(parsed_df.columns)}"
)
print(f"Ground truth shape: {ground_truth_df.shape}")
print(f"Parsed data shape: {parsed_df.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation Metrics

# COMMAND ----------

from src.evaluation import evaluate_dataframes, get_evaluation_summary

# Define columns to evaluate
array_columns = [
    "legacy_numbers",
    "moc_numbers",
    "equipment_tags",
    "line_tags",
    "incoming_streams",
    "outgoing_streams",
]
string_columns = ["drawing_name", "title", "revision", "date", "organization"]
boolean_columns = ["has_stamp"]

# Calculate similarities using abstracted function
similarity_df = evaluate_dataframes(
    ground_truth_df, parsed_df, array_columns, string_columns, boolean_columns
)

# Get summary statistics
summary_stats = get_evaluation_summary(similarity_df)

# Display summary statistics
for metric_type, stats in summary_stats.items():
    print(f"\n{metric_type.title()} Summary:")
    print(stats)

# COMMAND ----------

# Display the detailed results
display(similarity_df)
