# Databricks notebook source
# MAGIC %md
# MAGIC # Load Sheets
# MAGIC Most document vendors provide load sheets that can serve as examples for documents, few shot prompts, and evaluation. We load an example sheet made for a couple of the P&IDs in the example.
# MAGIC

# COMMAND ----------

# MAGIC %pip install openpyxl
# MAGIC %restart_python

# COMMAND ----------

import pandas as pd

load_sheet = pd.read_excel(
    "/Volumes/shm/pid/load_sheets/ALB PID Examples.xlsx", skiprows=2
)

# COMMAND ----------

load_sheet.shape

# COMMAND ----------

load_sheet.columns = [col.lower().replace("_", "") for col in load_sheet.columns]
target_cols = [
    "existingname",
    "title",
    "discipline",
    "documenttype",
    "revisionstatus",
    "primarylocation",
    "locations",
    "organization",
    "legacynumber",
    "physicallocation",
    "mocnumbers",
    "equipmenttags",
]
filt_load_sheet = load_sheet[target_cols]

# COMMAND ----------

from src.utils import get_spark

spark = get_spark()

from pyspark.sql.types import StructType, StructField, StringType
import pyspark.sql.functions as F

# Convert the 'mocnumbers' column to string
load_sheet["mocnumbers"] = load_sheet["mocnumbers"].astype(str)

schema = StructType(
    [
        StructField("existingname", StringType(), True),
        StructField("title", StringType(), True),
        StructField("discipline", StringType(), True),
        StructField("documenttype", StringType(), True),
        StructField("revisionstatus", StringType(), True),
        StructField("primarylocation", StringType(), True),
        StructField("locations", StringType(), True),
        StructField("organization", StringType(), True),
        StructField("legacynumber", StringType(), True),
        StructField("physicallocation", StringType(), True),
        StructField("mocnumbers", StringType(), True),
        StructField("equipmenttags", StringType(), True),
    ]
)

load_sheet_sp = spark.createDataFrame(load_sheet[target_cols], schema)
load_sheet_sp = load_sheet_sp.withColumn(
    "tags_array", F.split(F.col("equipmenttags"), ",")
).withColumnRenamed("existingname", "drawingname")
display(load_sheet_sp)

# COMMAND ----------

import json
import pyspark.sql.functions as F
from pyspark.sql.types import StringType


@F.udf(StringType())
def to_json_udf(row):
    columns_to_select = [
        "drawingname",
        "title",
        "primarylocation",
        "organization",
        "mocnumbers",
        "equipmenttags",
    ]
    row_dict = {col: row[col] for col in columns_to_select}
    return json.dumps(row_dict)


# COMMAND ----------

load_sheet_sp = load_sheet_sp.withColumn(
    "json_output",
    to_json_udf(F.struct([load_sheet_sp[x] for x in load_sheet_sp.columns])),
)
display(load_sheet_sp)

# COMMAND ----------

import os
from difflib import get_close_matches
import pyspark.sql.functions as F
from pyspark.sql.types import StringType

# List all filenames in the specified folder
folder_path = "/Volumes/shm/pid/raw_pdfs/ALB/with_load_sheet"
filenames = os.listdir(folder_path)
stems = [x.split("_")[0] for x in filenames]


@F.udf(StringType())
def find_closest_filename(drawingname):
    closest_match = get_close_matches(drawingname, filenames, n=1)
    matched_stem = closest_match[0] if closest_match else None
    return (
        filenames[filenames.index(matched_stem)] if matched_stem in filenames else None
    )


@F.udf(StringType())
def compare_drawingname_with_closest(drawingname, closest_filename):
    if closest_filename:
        closest_stem = closest_filename.split("_")[0]
        return drawingname == closest_stem
    return False


load_sheet_sp_closest_file = (
    load_sheet_sp.withColumn(
        "closest_filename", find_closest_filename(F.col("drawingname"))
    )
    .withColumn(
        "is_drawingname_match",
        compare_drawingname_with_closest(
            F.col("drawingname"), F.col("closest_filename")
        ),
    )
    .filter("is_drawingname_match == True")
    .select("drawingname", "closest_filename", "is_drawingname_match")
)

# COMMAND ----------

from pyspark.sql.functions import lit
from pyspark.sql import DataFrame


def add_for_examples_flag(
    df: DataFrame, seed: int = 42, sample_size: int = 23
) -> DataFrame:
    sampled_df = df.sample(
        withReplacement=False, fraction=sample_size / df.count(), seed=seed
    )
    flagged_df = sampled_df.withColumn("for_examples", lit(True))
    return df.join(
        flagged_df.select("drawingname").withColumn("for_examples", lit(True)),
        on="drawingname",
        how="left",
    ).fillna(False, subset=["for_examples"])


load_sheet_sp_closest_file = add_for_examples_flag(load_sheet_sp_closest_file)
display(load_sheet_sp_closest_file)

# COMMAND ----------

from pyspark.sql.functions import lit, col

load_sheet_sp_closest_file = load_sheet_sp_closest_file.alias("closest")
load_sheet_sp = load_sheet_sp.alias("original")

joined_df = (
    load_sheet_sp.join(
        load_sheet_sp_closest_file,
        load_sheet_sp["drawingname"] == load_sheet_sp_closest_file["drawingname"],
        "right",
    )
    .drop(load_sheet_sp["drawingname"])
    .select(
        col("closest_filename"),
        lit("ALB").alias("facility"),
        col("for_examples"),
        *load_sheet_sp.columns,
    )
)

display(joined_df)

# COMMAND ----------

# Save the DataFrame to a table under shm.pid with schema evolution enabled
joined_df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(
    "shm.pid.load_sheet_alb"
)
