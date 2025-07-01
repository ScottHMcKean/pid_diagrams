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

# MAGIC %pip install uv

# COMMAND ----------

# MAGIC %sh uv pip install .

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import hashlib
from pathlib import Path
import pandas as pd

from src.utils import get_spark
from src.preprocess import process_pdf_to_tiles
from src.config import load_config

# COMMAND ----------

spark = get_spark()
config = load_config("config_local.yaml")
ppconfig = config.preprocess

# COMMAND ----------

# MAGIC %md
# MAGIC # Page Tiling
# MAGIC We use [pdfplumber](https://github.com/jsvine/pdfplumber) to deal with multipage pdfs, and the python image library ([PIL](https://pillow.readthedocs.io/en/stable/)) to crop them into tiles. This has the advantage of keeping a relatively consistent tile size (edges excluded) and maintaining a consistent resolution among different pdf pages.

# COMMAND ----------

all_metadata = []
for pdf_file_path in Path(ppconfig.raw_path).glob("*.pdf"):
    print(pdf_file_path)
    # Single file (multipage) workflow
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
    all_metadata.extend(metadata)

# COMMAND ----------

len(all_metadata)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Metadata
# MAGIC When dealing with a huge number of files, it is important to keep track of metadata. We will be doing inference on both the whole pages and individual tiles, so need both logged and ready to go. We write this file into spark for future use and driving our parsing workflow.

# COMMAND ----------

spark = get_spark()
if spark:
    (
        spark.createDataFrame(pd.DataFrame(all_metadata))
        .write.mode("overwrite")
        .option("overwriteSchema", True)
        .saveAsTable(
            f"{config.catalog}.{config.schema}.{config.preprocess.tile_table_name}"
        )
    )
else:
    pd.DataFrame(metadata).to_parquet(
        Path("local_tables") / f"{config.preprocess.tile_table_name}.parquet"
    )
