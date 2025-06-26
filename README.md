# P&ID Diagrams

This repository implements a multimodal processing pipeline for P&ID (Process & Instrumentation Diagrams) using Claude Sonnet models. The pipeline includes comprehensive data preprocessing with PDF-to-image conversion and intelligent tiling, zero-shot and few-shot extraction capabilities for metadata and tag identification, and robust evaluation metrics. The codebase is organized into modular components: `src/preprocess.py` handles PDF processing and image tiling with configurable DPI and overlap settings, `src/parser.py` manages OpenAI API integration for LLM-based extraction with retry logic and JSON parsing, and `src/config.py` provides type-safe Pydantic configuration management with field validation. Key features include automated tile generation from multi-page PDFs, intelligent image preprocessing with contrast enhancement and thresholding, and structured output parsing from LLM responses.

We showed that tiles are the path forward for tag extraction. But that the model is hallucinating titles that don't exist. In order to improve our metadata capture, we are going to do two passes on the image. The first is a few-shot extraction of image metadata - titles, locations, revisions etc. The second is a tile by tile few shot extraction where we will curate samples ourselves to guide the model

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



