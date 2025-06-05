# Databricks notebook source
# MAGIC %md
# MAGIC # Preprocessing
# MAGIC
# MAGIC This notebook does preprocessing of PDFs, with the goal of
# MAGIC - converting pdf to image files
# MAGIC - getting one image per page
# MAGIC - hashing the input
# MAGIC - making a table

# COMMAND ----------

# MAGIC %pip install pymupdf
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest Files
# MAGIC Download process and instrumentation diagrams. We use an md5 hash to encode the file name and ensure uniqueness.

# COMMAND ----------

import mlflow
import requests
import hashlib
import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image
import io
from process import tile_image_with_overlap


# COMMAND ----------

config = mlflow.models.ModelConfig(development_config="config.yaml").to_dict()
raw_vol_path = config["raw_vol_path"]
image_vol_path = config["image_vol_path"]
tile_vol_path = config["tile_vol_path"]


# COMMAND ----------

url = "https://open.alberta.ca/dataset/46ddba1a-7b86-4d7c-b8b6-8fe33a60fada/resource/a82b9bc3-37a9-4447-8d2f-f5b55a5c3353/download/facilitydrawings.pdf"
hashed_url = hashlib.md5(url.encode()).hexdigest()

raw_pdf_file_path = raw_vol_path + hashed_url + ".pdf"

response = requests.get(url)
with open(raw_pdf_file_path, "wb") as file:
    file.write(response.content)

# COMMAND ----------

# MAGIC %md
# MAGIC We currently use PyMuPDF for splitting the pages. The license isn't great, but there are many other options. This is the simplest for now.

# COMMAND ----------

doc_dir = Path(image_vol_path) / hashed_url
doc_dir.mkdir(exist_ok=True)

doc = fitz.open(raw_pdf_file_path)
for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    pix = page.get_pixmap(dpi=400)
    pix.save(doc_dir / f"page_{page_num+1}.jpeg")

# COMMAND ----------

# Get the first image from the image directory
image_dir_glob = Path(image_vol_path).rglob("*.jpeg")
test_image = next(image_dir_glob)
test_image = Path("processed_pdfs/5a82c87214d47c8af93fb443908548ee/page_29.jpeg")
tile_image_with_overlap(test_image, tile_vol_path, hashed_url)
