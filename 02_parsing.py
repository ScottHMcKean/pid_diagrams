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

if spark:
    pages_to_parse = spark.sql(
        f"""
        SELECT *
        FROM (
        SELECT 
            *,
            ROW_NUMBER() OVER (PARTITION BY page_number ORDER BY tile_number DESC) as rn
        FROM {config.catalog}.{config.schema}.tile_info
        )
        WHERE rn = 1
        AND page_number in (12,32)
        """
    ).toPandas()
else:
    tile_info_df = pd.read_parquet(Path("local_tables") / "tile_info.parquet")
    pages_to_parse = (
        tile_info_df[tile_info_df["page_number"].isin([12, 32])]
        .sort_values(["page_number", "tile_number"], ascending=[True, False])
        .groupby("page_number")
        .first()
        .reset_index()
    )

# COMMAND ----------

# We are going to use a naive loop to query the examples, but will move to Ray or Spark for parallelization for the larger set of queries. The code below sends the excerpt and drawing into our model for a zero shot extraction.
metadata_results = []
for idx, row in pages_to_parse.iterrows():
    metadata_results.append(image_processor._parse_row(row, "metadata"))

# COMMAND ----------

# Metadata results
# We write the results to a table for future use.
metadata_df = pd.DataFrame(metadata_results)
if spark:
    # Use JSON strings because of mixed list and strings
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
for idx, row in tile_info_df[tile_info_df["page_number"].isin([12])].iterrows():
    tag_results.append(image_processor._parse_row(row, "tag"))

# COMMAND ----------

if spark:
    (
        spark.createDataFrame(pd.DataFrame(tag_results))
        .write.mode("overwrite")
        .option("overwriteSchema", True)
        .saveAsTable(f"{config.catalog}.{config.schema}.tag_results")
    )
else:
    pd.DataFrame(tag_results).to_parquet(Path("local_tables") / "tag_results.parquet")
