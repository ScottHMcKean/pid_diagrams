# P&ID Diagrams

This repository implements a multimodal processing pipeline for P&ID (Process & Instrumentation Diagrams) using Claude Sonnet models. The pipeline includes comprehensive data preprocessing with PDF-to-image conversion and intelligent tiling, zero-shot and few-shot extraction capabilities for metadata and tag identification, and robust evaluation metrics. The codebase is organized into modular components: `src/preprocess.py` handles PDF processing and image tiling with configurable DPI and overlap settings, `src/parser.py` manages OpenAI API integration for LLM-based extraction with retry logic and JSON parsing, and `src/config.py` provides type-safe Pydantic configuration management with field validation. Key features include automated tile generation from multi-page PDFs, intelligent image preprocessing with contrast enhancement and thresholding, and structured output parsing from LLM responses.

We showed that tiles are the path forward for tag extraction. But that the model is hallucinating titles that don't exist. In order to improve our metadata capture, we are going to do two passes on the image. The first is a few-shot extraction of image metadata - titles, locations, revisions etc. The second is a tile by tile few shot extraction where we will curate samples ourselves to guide the model

## Evaluation

To evaluate the performance of tag and stream parsing, we use precision, recall, and Jaccard Similarity for tags and streams. We use Normalized Levenshtein Distance for string similarity in each metadata field.

## Precision and Recall

Precision measures the proportion of your predicted positive cases that are actually correct. In string matching contexts, precision answers: "Of all the strings my model predicted as matches, how many were actually correct?" This is calculated as:

Precision = $\frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$

Recall measures the proportion of actual positive cases that your model successfully identified. For string matching, recall answers: "Of all the strings that should have been matched, how many did my model actually find?" This is calculated as:

Recall = $\frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$
 
Precision is more important when false positives are costly or problematic. For example, in email filtering systems, high precision ensures that important emails aren't incorrectly classified as spam. You want to minimize the risk of hiding legitimate emails from users, even if some spam gets through.

Recall is more important when missing true positives has serious consequences. In medical screening or security applications, high recall ensures that few or no positive cases are missed. It's better to have false alarms than to miss critical cases.

## Jaccard Similarity

Jaccard Similarity is a measure of similarity between two sets. It is calculated as the size of the intersection of the two sets divided by the size of the union of the two sets.

Jaccard Similarity = $\frac{\text{Intersection}}{\text{Union}}$

It is a useful metrics for comparing sets and punishes missing tags and hallucinations at the same time.

## Normalized Levenshtein Distance?

Normalized Levenshtein Distance is a measure of similarity between two strings. It is calculated as the Levenshtein distance between the two strings divided by the length of the longer string.

Normalized Levenshtein Distance = $\frac{\text{Levenshtein Distance}}{\text{Max(Length of String 1, Length of String 2)}}$

It is a useful metric for comparing strings, while allowing a small amount of room for typos or other symbol differences.

## Local Development

If you want to run the code locally, use local_notebooks and uv for dependency management.

```bash
uv venv --python=3.11.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Activate your environment using `source .venv/bin/activate`

## Testing

The project uses `pytest` for comprehensive testing with 30 tests organized into three categories: configuration tests (`test_config.py`) for YAML loading and Pydantic validation, preprocessing tests (`test_preprocess.py`) for PDF processing and image tiling algorithms, and parser tests (`test_parser.py`) for OpenAI API integration and JSON extraction. Run tests using `python run_tests.py` (all tests), `python run_tests.py unit/config/parser/preprocess` (specific categories), or `pytest tests/ -v` (direct pytest). Tests use pytest markers (`@pytest.mark.unit`, `@pytest.mark.integration`, etc.) and comprehensive mocking with `unittest.mock` to avoid external dependencies. All 30 tests currently pass with 100% success rate, providing robust coverage of configuration validation, image processing algorithms, PDF tiling logic, and API response handling. For local development, create a virtual environment with `uv venv --python=3.11.11`, activate it with `source .venv/bin/activate`, and install dependencies with `uv pip install -r requirements.txt`.

## Improvements
[ ] Try and get formal structured outputs working with OpenAI API 



