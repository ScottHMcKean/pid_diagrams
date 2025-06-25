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

# MAGIC %pip install -U --quiet pdfplumber mlflow
# MAGIC %restart_python

# COMMAND ----------

import requests
import hashlib
import io
import os
from pathlib import Path
from PIL import Image

import mlflow
from mlflow.models import ModelConfig
import pdfplumber
import pandas as pd

from src.preprocess import get_tile_positions

# COMMAND ----------

config = ModelConfig(development_config='config.yaml').to_dict()
ppconfig = config['preprocessing']

# COMMAND ----------

# MAGIC %md
# MAGIC # Page Tiling
# MAGIC We use [pdfplumber](https://github.com/jsvine/pdfplumber) to deal with multipage pdfs, and the python image library ([PIL](https://pillow.readthedocs.io/en/stable/)) to crop them into tiles. This has the advantage of keeping a relatively consistent tile size (edges excluded) and maintaining a consistent resolution among different pdf pages.

# COMMAND ----------

from PIL import ImageEnhance

# Single file (multipage) workflow
pdf_file_path = Path(ppconfig['raw_path'])
file_path_hash = hashlib.md5(str(pdf_file_path).encode()).hexdigest()
pdf_name = pdf_file_path.stem

# Hashed output dir
output_dir = Path(ppconfig['processed_path']) / file_path_hash
(output_dir / 'tiles').mkdir(parents=True, exist_ok=True)

metadata = []
with pdfplumber.open(pdf_file_path) as pdf:
    for page_num, page in enumerate(pdf.pages):
        tile_count = 1
        page_img = page.to_image(resolution=ppconfig['dpi']).original

        # convert to grayscale
        page_img = page_img.convert('L')

        # enhance contrast
        contrast = ImageEnhance.Contrast(page_img)
        page_img = contrast.enhance(2.0)

        # apply thresholding (background suppression)
        threshold = 128  # mid thresholding (50%)
        page_img = page_img.point(
            lambda x: 255 if x > threshold else 0, mode='1'
            )

        page_path = output_dir / f"{file_path_hash}_p{page_num+1}.jpg"
        page_img.save(page_path, "JPEG")
        
        width, height = page_img.size
        x_positions = get_tile_positions(
            width, 
            ppconfig['tile_width_px'], 
            ppconfig['overlap_px']
            )
        
        y_positions = get_tile_positions(
            height, 
            ppconfig['tile_height_px'], 
            ppconfig['overlap_px']
            )
        
        for upper in y_positions:
            for left in x_positions:
                right = left + ppconfig['tile_width_px']
                lower = upper + ppconfig['tile_height_px']
                tile = page_img.crop((left, upper, right, lower))
                tile_filename = f"{file_path_hash}_p{page_num+1}_t{tile_count}.jpg"
                tile_path = output_dir / 'tiles' / tile_filename
                tile.save(tile_path, "JPEG")

                metadata.append({
                    'filename': pdf_name,
                    'file_path_hash': str(file_path_hash),
                    'file_width': width,
                    'file_height': height,
                    'file_dpi': ppconfig['dpi'],
                    'page_number': page_num + 1,
                    'page_path': str(page_path),
                    'tile_number': tile_count,
                    'left': left,
                    'upper': upper,
                    'right': right,
                    'lower': lower,
                    'tile_path': str(tile_path)
                })

                tile_count += 1

# COMMAND ----------

# MAGIC %md
# MAGIC ## Metadata
# MAGIC When dealing with a huge number of files, it is important to keep track of metadata. We will be doing inference on both the whole pages and individual tiles, so need both logged and ready to go. We write this file into spark for future use and driving our parsing workflow.

# COMMAND ----------

(
  spark.createDataFrame(pd.DataFrame(metadata))
  .write.mode('overwrite')
  .saveAsTable(f"{config['catalog']}.{config['schema']}.tile_info")
)

# COMMAND ----------

spark.sql(f"SELECT * FROM {config['catalog']}.{config['schema']}.tile_info").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Sheets
# MAGIC Most document vendors provide load sheets that can serve as examples for documents, few shot prompts, and evaluation. We load an example sheet made for a couple of the P&IDs in the example.
