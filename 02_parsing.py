# Databricks notebook source
# MAGIC %md
# MAGIC # 2 - Parsing
# MAGIC
# MAGIC This notebook uses the preprocessed images to do zero and few shot parsing of the P&ID diagrams. This notebook has been tested on serverless.

# COMMAND ----------

# MAGIC %pip install -U --quiet -r requirements.txt
# MAGIC %restart_python


# COMMAND ----------

from pathlib import Path
import pandas as pd
import json

from openai import OpenAI
from databricks.sdk import WorkspaceClient

from src.config import load_config
from src.parser import OpenAIRequestHandler, ImageProcessor
from src.utils import get_spark, get_token

# COMMAND ----------

spark = get_spark()
config = load_config("config.yaml")
pconfig = config.parse

# COMMAND ----------

# Setup LLM Client
w = WorkspaceClient()
token = get_token(w)

llm_client = OpenAI(
    api_key=token,
    base_url=f"{w.config.host}/serving-endpoints",
)

# COMMAND ----------

# Setup request handler and image processor
request_handler = OpenAIRequestHandler(llm_client, pconfig)
image_processor = ImageProcessor(request_handler, pconfig)

# COMMAND ----------

# Metadata parsing (per page)
# This query pulls the last tile from each example page
# This section runs the metadata prompt using the entire image from each example and the last tile (which is always the lower right). The last tile should contain most title blocks due to the dimensions of the tiles and resolution.

if spark:
    tile_info_df = spark.table(f"{config.catalog}.{config.schema}.tile_info").toPandas()
else:
    tile_info_df = pd.read_parquet(Path("local_tables") / "tile_info.parquet")

pages_to_parse = (
    tile_info_df[tile_info_df["page_number"].isin([12])]
    .sort_values(["page_number", "tile_number"], ascending=[True, False])
    .groupby("page_number")
    .first()
    .reset_index()
)

# COMMAND ----------
# We are going to use a naive loop to query the examples, but will move to Ray or Spark for parallelization for the larger set of queries. The code below sends the excerpt and drawing into our model for a zero shot extraction.
metadata_results = []
metadata_raw_responses = []
for idx, row in pages_to_parse.iterrows():
    metadata_row, raw_response = image_processor._parse_row(row, "metadata")
    metadata_results.append(metadata_row)
    metadata_raw_responses.append(raw_response)

# COMMAND ----------

# Metadata results
# We write the results to a table for future use.
# We cast the json to a string to avoid issues with mixes lists and strings, or null types.
metadata_df = pd.DataFrame(metadata_results)
if spark:
    metadata_df["parsed_metadata"] = metadata_df["parsed_metadata"].apply(json.dumps)
    (
        spark.createDataFrame(metadata_df)
        .write.mode("overwrite")
        .option("overwriteSchema", True)
        .saveAsTable(f"{config.catalog}.{config.schema}.metadata_results")
    )
else:
    metadata_df.to_parquet(Path("local_tables") / "metadata_results.parquet")

# COMMAND ----------

# Tag parsing (per tile)
# This section runs the tag prompt using the entire image from each example and the last tile (which is always the lower right). The last tile should contain most title blocks due to the dimensions of the tiles and resolution.
tag_results = []
tag_raw_responses = []
for idx, row in tile_info_df[tile_info_df["page_number"].isin([12])].iterrows():
    tag_row, raw_response = image_processor._parse_row(row, "tag")
    tag_results.append(tag_row)
    tag_raw_responses.append(raw_response)


# COMMAND ----------

# refresh spark connection
spark = get_spark()
tag_df = pd.DataFrame(tag_results)
if spark:
    tag_df["parsed_tag"] = tag_df["parsed_tag"].apply(json.dumps)
    (
        spark.createDataFrame(tag_df)
        .write.mode("overwrite")
        .option("overwriteSchema", True)
        .saveAsTable(f"{config.catalog}.{config.schema}.tag_results")
    )
else:
    pd.DataFrame(tag_df).to_parquet(Path("local_tables") / "tag_results.parquet")
