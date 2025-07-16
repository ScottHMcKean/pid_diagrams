# Databricks notebook source
# MAGIC %md
# MAGIC # 2 - Parsing
# MAGIC
# MAGIC This notebook uses the preprocessed images to do zero and few shot parsing of the P&ID diagrams. This notebook has been tested on serverless v3.

# COMMAND ----------

# MAGIC %pip install uv

# COMMAND ----------

# MAGIC %sh uv pip install .

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import sys
sys.path.append('.')

# COMMAND ----------

from pathlib import Path
import pandas as pd
import json

from openai import OpenAI
from databricks.sdk import WorkspaceClient
from src.config import load_config
from src.parser import OpenAIRequestHandler, ImageProcessor
from src.utils import get_token, test_spark

# COMMAND ----------

spark = test_spark()
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
    tile_info_df = spark.table(
        f"{config.catalog}.{config.schema}.{config.preprocess.tile_table_name}"
    ).toPandas()
else:
    tile_info_df = pd.read_parquet(
        Path("local_tables") / f"{config.preprocess.tile_table_name}.parquet"
    )

pages_to_parse = (
    tile_info_df.sort_values(
        ["filename", "page_number", "tile_number"], ascending=[False, True, False]
    )
    .groupby(["filename", "page_number"])
    .first()
    .reset_index()
)

# COMMAND ----------

sample_unique_keys = [
    # "05010ca2e0d35676718f1cc15862b8fc_p3_t6", #for local testing
    "7203369372d032999062c2d0156e776a_p1_t6",
    "bb0c134e870dcb5618dd8fcb594bc16a_p1_t6",
    "ddd29e2a334e61750b34a978f06c3643_p1_t6",
    "579036a2c6cbb4f74243a84961cfdfd8_p1_t6",
    "8011066889c0502abb02a4828e0c6653_p1_t6",
]

# COMMAND ----------

pages_to_parse = pages_to_parse[pages_to_parse.unique_key.isin(sample_unique_keys)]

# COMMAND ----------

import mlflow
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

mlflow.set_experiment("/Users/scott.mckean@databricks.com/experiments/pid_diagram")

# COMMAND ----------

# register the prompt
register_prompt = False
if register_prompt:
    mlflow.genai.register_prompt(
        name="shm.pid.metadata_prompt",
        template=pconfig.metadata_prompt,
    )
    mlflow.genai.register_prompt(
        name="shm.pid.tag_prompt",
        template=pconfig.metadata_prompt,
    )

# Searching prompt example
# TODO: Set prompt version for table outputs
metadata_prompt = [
    x
    for x in mlflow.genai.search_prompts("catalog = 'shm' AND schema = 'pid'")
    if "metadata" in x.name
][0]
version = metadata_prompt.tags["PromptVersionCount"]
old_prompt = mlflow.genai.load_prompt(
    f"prompts:/{metadata_prompt.name}/{version}"
).template

# COMMAND ----------

# MAGIC %md
# MAGIC We are going to use a naive loop to query the examples, but will move to Ray or Spark for parallelization for the larger set of queries. The code below sends the excerpt and drawing into our model for a zero shot extraction.

# COMMAND ----------

metadata_results = []
for idx, row in pages_to_parse.iterrows():
    metadata_row = image_processor._parse_row(row, "metadata")
    metadata_results.append(metadata_row)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Metadata Results

# COMMAND ----------

spark = test_spark()
metadata_df = pd.DataFrame(metadata_results)
if spark:
    metadata_df["parsed_metadata"] = metadata_df["parsed_metadata"].apply(json.dumps)
    (
        spark.createDataFrame(metadata_df)
        .write.mode("overwrite")
        .option("overwriteSchema", True)
        .saveAsTable(f"{config.catalog}.{config.schema}.{pconfig.metadata_table_name}")
    )
else:
    metadata_df.to_parquet(Path("local_tables") / "metadata_results.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tag Parsing
# MAGIC This section runs the tag prompt using the entire image from each example and the last tile (which is always the lower right). The last tile should contain most title blocks due to the dimensions of the tiles and resolution.

# COMMAND ----------

tiles_to_parse = tile_info_df[
    tile_info_df.unique_key.str.contains("|".join(metadata_df.unique_key.astype(str)))
]

tag_results = []
for idx, row in tiles_to_parse.iterrows():
    tag_row = image_processor._parse_row(row, "tag")
    tag_results.append(tag_row)

# COMMAND ----------

spark = test_spark()
tag_df = pd.DataFrame(tag_results)
if spark:
    tag_df["parsed_tag"] = tag_df["parsed_tag"].apply(json.dumps)
    (
        spark.createDataFrame(tag_df)
        .write.mode("overwrite")
        .option("overwriteSchema", True)
        .saveAsTable(f"{config.catalog}.{config.schema}.{pconfig.tags_table_name}")
    )
else:
    pd.DataFrame(tag_df).to_parquet(Path("local_tables") / "tag_results.parquet")
