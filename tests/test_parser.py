"""
Test parser functionality.
"""

import pytest
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.parser import OpenAIRequestHandler, ImageProcessor
from src.config import ParseConfig, get_parse_config


@pytest.mark.parser
@pytest.mark.unit
class TestOpenAIRequestHandler:
    """Test OpenAI request handler."""

    def test_init(self):
        """Test initialization of request handler."""
        mock_client = Mock()
        config = ParseConfig(
            output_path="/tmp",
            fm_endpoint="test-model",
            metadata_prompt="test metadata prompt",
            metadata_example="test example",
            tag_prompt="test tag prompt",
            tag_example="test example",
            example_path="/tmp/examples",
        )

        handler = OpenAIRequestHandler(mock_client, config)

        assert handler.client == mock_client
        assert handler.config == config

    def test_make_request(self):
        """Test making API request."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "test response"
        mock_client.chat.completions.create.return_value = mock_response

        config = ParseConfig(
            output_path="/tmp",
            fm_endpoint="test-model",
            temperature=0.5,
            thinking_budget_tokens=1024,
            metadata_prompt="test metadata prompt",
            metadata_example="test example",
            tag_prompt="test tag prompt",
            tag_example="test example",
            example_path="/tmp/examples",
        )

        handler = OpenAIRequestHandler(mock_client, config)
        response = handler.make_request("base64_image_data", "test prompt")

        assert response == "test response"

        # Verify the API call was made correctly
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args

        assert call_args.kwargs["model"] == "test-model"
        assert call_args.kwargs["temperature"] == 0.5
        assert call_args.kwargs["extra_body"]["thinking"]["budget_tokens"] == 1024


@pytest.mark.parser
@pytest.mark.unit
class TestImageProcessor:
    """Test image processor functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_handler = Mock()
        self.config = ParseConfig(
            output_path="/tmp/test_output",
            fm_endpoint="test-model",
            max_retries=2,
            retry_delay_s=0,  # No delay in tests
            metadata_prompt="metadata prompt",
            metadata_example="metadata example",
            tag_prompt="tag prompt",
            tag_example="tag example",
            example_path="/tmp/examples",
        )

    def test_init(self):
        """Test initialization of image processor."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ParseConfig(
                output_path=temp_dir,
                example_path="/tmp/examples",
                fm_endpoint="test-model",
                metadata_prompt="test",
                metadata_example="test",
                tag_prompt="test",
                tag_example="test",
            )

            processor = ImageProcessor(self.mock_handler, config)

            assert processor.request_handler == self.mock_handler
            assert processor.config == config
            assert processor.output_dir == Path(temp_dir)

    def test_load_image(self):
        """Test image loading and base64 encoding with real file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ParseConfig(
                output_path=temp_dir,
                example_path="/tmp/examples",
                fm_endpoint="test-model",
                metadata_prompt="test",
                metadata_example="test",
                tag_prompt="test",
                tag_example="test",
            )

            processor = ImageProcessor(self.mock_handler, config)

            # Create a real test file
            test_file = Path(temp_dir) / "test_image.jpg"
            test_file.write_bytes(b"fake image data")

            result = processor._load_image(str(test_file))

            # Should return base64 encoded string
            assert isinstance(result, str)
            assert len(result) > 0

    def test_extract_json_valid(self):
        """Test JSON extraction from valid response."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ParseConfig(
                output_path=temp_dir,
                example_path="/tmp/examples",
                fm_endpoint="test-model",
                metadata_prompt="test",
                metadata_example="test",
                tag_prompt="test",
                tag_example="test",
            )

            processor = ImageProcessor(self.mock_handler, config)

            # Test valid JSON
            valid_json = '{"key": "value", "number": 123}'
            result = processor._extract_json(valid_json)

            assert result == {"key": "value", "number": 123}

    def test_extract_json_with_markdown(self):
        """Test JSON extraction from markdown-wrapped response."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ParseConfig(
                output_path=temp_dir,
                example_path="/tmp/examples",
                fm_endpoint="test-model",
                metadata_prompt="test",
                metadata_example="test",
                tag_prompt="test",
                tag_example="test",
            )

            processor = ImageProcessor(self.mock_handler, config)

            # Test JSON wrapped in markdown
            markdown_json = '```json\n{"key": "value"}\n```'
            result = processor._extract_json(markdown_json)

            assert result == {"key": "value"}

    def test_extract_json_with_quoted_numbers(self):
        """Test JSON extraction with quoted number fixes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ParseConfig(
                output_path=temp_dir,
                example_path="/tmp/examples",
                fm_endpoint="test-model",
                metadata_prompt="test",
                metadata_example="test",
                tag_prompt="test",
                tag_example="test",
            )

            processor = ImageProcessor(self.mock_handler, config)

            # Test malformed JSON with quoted numbers and hyphens
            malformed_json = '{"tag": "250"-LT-1610"}'
            result = processor._extract_json(malformed_json)

            assert result == {"tag": "250-LT-1610"}

    def test_extract_json_invalid(self):
        """Test JSON extraction from invalid response."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ParseConfig(
                output_path=temp_dir,
                example_path="/tmp/examples",
                fm_endpoint="test-model",
                metadata_prompt="test",
                metadata_example="test",
                tag_prompt="test",
                tag_example="test",
            )

            processor = ImageProcessor(self.mock_handler, config)

            # Test completely invalid JSON
            invalid_json = "This is not JSON at all"
            result = processor._extract_json(invalid_json)

            # Should return original string when can't parse
            assert result == invalid_json

    def test_save_result(self):
        """Test saving results to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ParseConfig(
                output_path=temp_dir,
                example_path="/tmp/examples",
                fm_endpoint="test-model",
                metadata_prompt="test",
                metadata_example="test",
                tag_prompt="test",
                tag_example="test",
            )

            processor = ImageProcessor(self.mock_handler, config)

            test_data = {"test": "data", "number": 42}
            filename = "test_output.json"

            processor._save_result(filename, test_data)

            # Check that file was created and contains correct data
            output_file = Path(temp_dir) / filename
            assert output_file.exists()

            with open(output_file, "r") as f:
                saved_data = json.load(f)

            assert saved_data == test_data


@pytest.mark.parser
@pytest.mark.config
@pytest.mark.integration
class TestGetParseConfigFromFile:
    """Test configuration loading for parser."""

    def test_get_parse_config_from_file(self):
        """Test loading parse config from file."""
        config = get_parse_config("config.yaml")

        assert isinstance(config, ParseConfig)
        assert config.fm_endpoint == "databricks-claude-3-7-sonnet"
        assert config.temperature == 0.1
        assert config.max_retries == 3
