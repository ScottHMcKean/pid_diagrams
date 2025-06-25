# Databricks notebook source
# MAGIC %md
# MAGIC # 2 - Parsing
# MAGIC
# MAGIC This notebook uses the preprocessed images to do zero and few shot parsing of the P&ID diagrams.

# COMMAND ----------

# MAGIC %pip install -U --quiet pdfplumber pydantic pyyaml
# MAGIC %restart_python

# COMMAND ----------

import hashlib
from pathlib import Path
import pandas as pd

from openai import OpenAI
from databricks.sdk import WorkspaceClient

from src.config import load_config
from src.parser import OpenAIRequestHandler, ImageProcessor
from src.utils import _is_spark_available

# COMMAND ----------
config = load_config("config.yaml")
pconfig = config.parse

# COMMAND ----------
# Setup LLM Client
w = WorkspaceClient()

workspace_client = WorkspaceClient()

llm_client = OpenAI(
    api_key=workspace_client.config.token,
    base_url=f"{workspace_client.config.host}/serving-endpoints",
)

# COMMAND ----------
# Setup request handler and image processor
request_handler = OpenAIRequestHandler(llm_client, pconfig)
image_processor = ImageProcessor(request_handler, pconfig)

# COMMAND ----------
if _is_spark_available():
    driver_table = spark.table(f"{config.catalog}.{config.schema}.tile_info")
else:
    driver_table = pd.read_parquet(Path("local_tables") / "tile_info.parquet")

# COMMAND ----------
# Get examples for few shot
row = driver_table.sample(1).iloc[0]

tag_out = image_processor._parse_row(row, "tag")
metadata_out = image_processor._parse_row(row, "metadata")
