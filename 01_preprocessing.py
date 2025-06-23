# Databricks notebook source
# MAGIC %md
# MAGIC # Preprocessing
# MAGIC
# MAGIC This notebook does preprocessing of PDFs, with the goal of
# MAGIC - converting pdf to image files
# MAGIC - getting one image per page
# MAGIC - hashing the input
# MAGIC - making a table
# MAGIC
# MAGIC It has been tested with serverless

# COMMAND ----------

# MAGIC %pip install pymupdf mlflow==2.22.0
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
from src.process import tile_image_with_overlap

# COMMAND ----------

config = mlflow.models.ModelConfig(development_config="config.yaml").to_dict()

# COMMAND ----------

raw_pdf_file_path = config["raw_path"] 


# COMMAND ----------

# url = "https://open.alberta.ca/dataset/46ddba1a-7b86-4d7c-b8b6-8fe33a60fada/resource/a82b9bc3-37a9-4447-8d2f-f5b55a5c3353/download/facilitydrawings.pdf"
# hashed_url = hashlib.md5(url.encode()).hexdigest()

# raw_pdf_file_path = config["raw_path"] + hashed_url + ".pdf"

# response = requests.get(url)
# with open(raw_pdf_file_path, "wb") as file:
#     file.write(response.content)

# COMMAND ----------

# MAGIC %md
# MAGIC We currently use PyMuPDF for splitting the pages. The license isn't great, but there are many other options. This is the simplest for now.

# COMMAND ----------

from pathlib import Path
import fitz  # PyMuPDF

raw_dir   = Path(config["raw_path"])
output_dir = Path(config["processed_path"])
output_dir.mkdir(exist_ok=True)  # ensure base output exists

# Loop over every PDF in the raw directory:
for raw_pdf_path in raw_dir.glob("*.pdf"):
    print(f"Processing {raw_pdf_path.name}…")
    # Make a subfolder named after the PDF (e.g. "invoice_123.pdf" → "invoice_123")
    pdf_stem = raw_pdf_path.stem
    pdf_out_dir = output_dir / pdf_stem
    pdf_out_dir.mkdir(exist_ok=True)

    doc = fitz.open(raw_pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=200)
        # Save as "invoice_123_page_1.jpeg", "invoice_123_page_2.jpeg", etc.
        out_name = f"{pdf_stem}_page_{page_num+1}.jpeg"
        pix.save(pdf_out_dir / out_name)

    doc.close()


# COMMAND ----------

# doc_dir = Path(config["processed_path"])
# doc_dir.mkdir(exist_ok=True)

# doc = fitz.open(raw_pdf_file_path)
# for page_num in range(len(doc)):
#     page = doc.load_page(page_num)
#     pix = page.get_pixmap(dpi=200)
#     pix.save(doc_dir / f"page_{page_num+1}.jpeg")

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's tile each pdf image

# COMMAND ----------

from pathlib import Path
from PIL import Image

def tile_image_with_overlap(image_path, output_dir, overlap_percent=10):
    img = Image.open(image_path)
    img_name_suffix = image_path.stem
    hash_name = image_path.parent.stem
    width, height = img.size

    cols, rows = 4, 2  # 8 tiles: 4 columns x 2 rows

    # Compute base tile size (without overlap)
    base_tile_width = width // cols
    base_tile_height = height // rows

    # Compute overlap in pixels
    overlap_x = int(base_tile_width * overlap_percent / 100)
    overlap_y = int(base_tile_height * overlap_percent / 100)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_num = 1
    for row in range(rows):
        for col in range(cols):
            # Calculate the starting x/y
            left = col * (base_tile_width - overlap_x)
            upper = row * (base_tile_height - overlap_y)

            # For the last column/row, ensure we reach the image edge
            if col == cols - 1:
                right = width
            else:
                right = left + base_tile_width

            if row == rows - 1:
                lower = height
            else:
                lower = upper + base_tile_height

            # Clamp to image boundaries
            left = max(0, left)
            upper = max(0, upper)
            right = min(width, right)
            lower = min(height, lower)

            tile = img.crop((left, upper, right, lower))
            tile_path = output_dir / hash_name / f"{img_name_suffix}_tile_{str(tile_num)}.webp"
            tile_path.parent.mkdir(parents=True, exist_ok=True)
            tile.save(tile_path)
            tile_num += 1

# COMMAND ----------

#test_path = Path('/Volumes/shm/pid/processed_pdfs/5a82c87214d47c8af93fb443908548ee/page_25.jpeg')
test_path = Path('/Volumes/shm/pid/processed_pdfs/ALB/with_load_sheet/ALB-701-PID-PR-005046-003_3DD0.1/ALB-701-PID-PR-005046-003_3DD0.1_page_1.jpeg')

# COMMAND ----------

# Tile the images
for test_path in Path(config['processed_path']).rglob("*.jpeg"):
    tile_image_with_overlap(test_path, config["tiled_path"])
