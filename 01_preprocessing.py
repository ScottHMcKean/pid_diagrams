# Databricks notebook source
# MAGIC %md
# MAGIC # 1 - Preprocessing
# MAGIC
# MAGIC This notebook preprocesses the PDFs, including:
# MAGIC - converting single or multi page pdfs to image files
# MAGIC - hashing the input and gathering metadata
# MAGIC - making a driver table for the workflow
# MAGIC
# MAGIC This notebook works with serverless.

# COMMAND ----------

# MAGIC %pip install -U --quiet pdfplumber pydantic pyyaml
# MAGIC %restart_python

# COMMAND ----------

import hashlib
from pathlib import Path
import pandas as pd

from src.utils import _is_spark_available
from src.preprocess import process_pdf_to_tiles
from src.config import load_config

# COMMAND ----------

config = load_config("config.yaml")
ppconfig = config.preprocess

# COMMAND ----------

# MAGIC %md
# MAGIC # Page Tiling
# MAGIC We use [pdfplumber](https://github.com/jsvine/pdfplumber) to deal with multipage pdfs, and the python image library ([PIL](https://pillow.readthedocs.io/en/stable/)) to crop them into tiles. This has the advantage of keeping a relatively consistent tile size (edges excluded) and maintaining a consistent resolution among different pdf pages.

# COMMAND ----------

# Single file (multipage) workflow
pdf_file_path = Path(ppconfig.raw_path)
file_path_hash = hashlib.md5(str(pdf_file_path).encode()).hexdigest()
pdf_name = pdf_file_path.stem

# Hashed output dir
output_dir = Path(ppconfig.processed_path)

# Process PDF to tiles using the refactored function
metadata = process_pdf_to_tiles(
    pdf_path=str(pdf_file_path),
    output_dir=str(output_dir),
    config=ppconfig,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Metadata
# MAGIC When dealing with a huge number of files, it is important to keep track of metadata. We will be doing inference on both the whole pages and individual tiles, so need both logged and ready to go. We write this file into spark for future use and driving our parsing workflow.

# COMMAND ----------

if _is_spark_available():
    (
        spark.createDataFrame(pd.DataFrame(metadata))
        .write.mode("overwrite")
        .saveAsTable(f"{config.catalog}.{config.schema}.tile_info")
    )
else:
    pd.DataFrame(metadata).to_parquet(Path("local_tables") / "tile_info.parquet")

# COMMAND ----------

if _is_spark_available():
    spark.sql(f"SELECT * FROM {config.catalog}.{config.schema}.tile_info").display()
else:
    pd.read_parquet(Path("local_tables") / "tile_info.parquet")

# MAGIC %md
# MAGIC ## Load Sheets
# MAGIC Most document vendors provide load sheets that can serve as examples for documents, few shot prompts, and evaluation. We load an example sheet made for a couple of the P&IDs in the example.
