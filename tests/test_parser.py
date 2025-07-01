"""
Test parser functionality.
"""

import pytest
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime

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
            parsed_path="/tmp",
            local_tables_path="/tmp/local_tables",
            metadata_table_name="test_metadata",
            tags_table_name="test_tags",
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
            parsed_path="/tmp",
            local_tables_path="/tmp/local_tables",
            metadata_table_name="test_metadata",
            tags_table_name="test_tags",
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
            parsed_path="/tmp/test_output",
            local_tables_path="/tmp/local_tables",
            metadata_table_name="test_metadata",
            tags_table_name="test_tags",
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
                parsed_path=temp_dir,
                local_tables_path="/tmp/local_tables",
                metadata_table_name="test_metadata",
                tags_table_name="test_tags",
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
            # Test new raw response directory creation
            assert processor.raw_response_dir == Path(temp_dir) / "raw_responses"
            assert processor.raw_response_dir.exists()

    def test_check_existing_extraction_exists(self):
        """Test checking for existing extraction when file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ParseConfig(
                parsed_path=temp_dir,
                local_tables_path="/tmp/local_tables",
                metadata_table_name="test_metadata",
                tags_table_name="test_tags",
                example_path="/tmp/examples",
                fm_endpoint="test-model",
                metadata_prompt="test",
                metadata_example="test",
                tag_prompt="test",
                tag_example="test",
            )

            processor = ImageProcessor(self.mock_handler, config)

            # Create existing extraction file
            test_data = {"tag": "test_value", "confidence": 0.95}
            test_file = processor.output_dir / "test_extraction.json"
            with open(test_file, "w") as f:
                json.dump(test_data, f)

            result = processor._check_existing_extraction("test_extraction.json")

            assert result == test_data

    def test_check_existing_extraction_not_exists(self):
        """Test checking for existing extraction when file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ParseConfig(
                parsed_path=temp_dir,
                local_tables_path="/tmp/local_tables",
                metadata_table_name="test_metadata",
                tags_table_name="test_tags",
                example_path="/tmp/examples",
                fm_endpoint="test-model",
                metadata_prompt="test",
                metadata_example="test",
                tag_prompt="test",
                tag_example="test",
            )

            processor = ImageProcessor(self.mock_handler, config)

            result = processor._check_existing_extraction("nonexistent.json")

            assert result is None

    def test_check_existing_extraction_corrupted(self):
        """Test checking for existing extraction with corrupted JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ParseConfig(
                parsed_path=temp_dir,
                local_tables_path="/tmp/local_tables",
                metadata_table_name="test_metadata",
                tags_table_name="test_tags",
                example_path="/tmp/examples",
                fm_endpoint="test-model",
                metadata_prompt="test",
                metadata_example="test",
                tag_prompt="test",
                tag_example="test",
            )

            processor = ImageProcessor(self.mock_handler, config)

            # Create corrupted JSON file
            corrupted_file = processor.output_dir / "corrupted.json"
            corrupted_file.write_text("{ invalid json")

            result = processor._check_existing_extraction("corrupted.json")

            assert result is None

    def test_save_raw_response(self):
        """Test saving raw response with metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ParseConfig(
                parsed_path=temp_dir,
                local_tables_path="/tmp/local_tables",
                metadata_table_name="test_metadata",
                tags_table_name="test_tags",
                example_path="/tmp/examples",
                fm_endpoint="test-model",
                metadata_prompt="test",
                metadata_example="test",
                tag_prompt="test",
                tag_example="test",
            )

            processor = ImageProcessor(self.mock_handler, config)

            # Test data
            task_key = "test_key_001"
            task = "tag"
            raw_response = '{"test": "response"}'
            metadata = {"page_number": 1, "tile_number": 1}

            # Mock datetime to get predictable timestamp
            with patch("src.parser.datetime") as mock_datetime:
                mock_datetime.now.return_value.isoformat.return_value = (
                    "2024-01-01T12:00:00"
                )

                processor._save_raw_response(task_key, task, raw_response, metadata)

            # Check that raw response file was created
            expected_file = processor.raw_response_dir / f"{task_key}_{task}_raw.json"
            assert expected_file.exists()

            # Check file contents
            with open(expected_file, "r") as f:
                saved_data = json.load(f)

            expected_data = {
                "task_key": task_key,
                "task": task,
                "timestamp": "2024-01-01T12:00:00",
                "metadata": metadata,
                "raw_response": raw_response,
            }

            assert saved_data == expected_data

    def test_save_result_with_raw_response(self):
        """Test saving result with raw response included."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ParseConfig(
                parsed_path=temp_dir,
                local_tables_path="/tmp/local_tables",
                metadata_table_name="test_metadata",
                tags_table_name="test_tags",
                example_path="/tmp/examples",
                fm_endpoint="test-model",
                metadata_prompt="test",
                metadata_example="test",
                tag_prompt="test",
                tag_example="test",
            )

            processor = ImageProcessor(self.mock_handler, config)

            test_data = {"tag": "test_value", "confidence": 0.95}
            raw_response = '{"tag": "test_value", "confidence": 0.95}'
            filename = "test_with_raw.json"

            processor._save_result(filename, test_data, raw_response)

            # Check that file was created
            output_file = processor.output_dir / filename
            assert output_file.exists()

            # Check file contents include both data and raw response
            with open(output_file, "r") as f:
                saved_data = json.load(f)

            expected_data = {
                "tag": "test_value",
                "confidence": 0.95,
                "_raw_response": raw_response,
            }

            assert saved_data == expected_data

    def test_save_result_without_raw_response(self):
        """Test saving result without raw response (backward compatibility)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ParseConfig(
                parsed_path=temp_dir,
                local_tables_path="/tmp/local_tables",
                metadata_table_name="test_metadata",
                tags_table_name="test_tags",
                example_path="/tmp/examples",
                fm_endpoint="test-model",
                metadata_prompt="test",
                metadata_example="test",
                tag_prompt="test",
                tag_example="test",
            )

            processor = ImageProcessor(self.mock_handler, config)

            test_data = {"tag": "test_value", "confidence": 0.95}
            filename = "test_without_raw.json"

            processor._save_result(filename, test_data)

            # Check that file was created
            output_file = processor.output_dir / filename
            assert output_file.exists()

            # Check file contents don't include raw response
            with open(output_file, "r") as f:
                saved_data = json.load(f)

            assert saved_data == test_data
            assert "_raw_response" not in saved_data

    def test_parse_row_with_existing_extraction(self):
        """Test parse row with existing extraction (should skip API call)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ParseConfig(
                parsed_path=temp_dir,
                local_tables_path="/tmp/local_tables",
                metadata_table_name="test_metadata",
                tags_table_name="test_tags",
                example_path="examples",  # Use real examples path
                fm_endpoint="test-model",
                metadata_prompt="test",
                metadata_example="test",
                tag_prompt="test",
                tag_example="test",
                num_few_shot_examples=0,  # Disable few shot
            )

            processor = ImageProcessor(self.mock_handler, config)

            # Create existing extraction
            existing_data = {
                "tag": "existing_value",
                "confidence": 0.8,
                "_raw_response": "old_response",
            }
            existing_file = processor.output_dir / "test_key.json"
            with open(existing_file, "w") as f:
                json.dump(existing_data, f)

            # Create test image
            test_image = Path(temp_dir) / "test_image.jpg"
            test_image.write_bytes(b"fake image data")

            test_row = {
                "unique_key": "test_key",
                "tile_path": str(test_image),
                "page_number": 1,
                "tile_number": 1,
            }

            result_row = processor._parse_row(test_row, "tag")
            raw_response = result_row.get("raw_response")

            # Should not have called the API
            assert not self.mock_handler.make_request.called

            # Should return existing data with raw response
            expected_result = {"tag": "existing_value", "confidence": 0.8}
            assert result_row["parsed_tag"] == expected_result
            assert raw_response == "old_response"

    def test_parse_row_new_extraction(self):
        """Test parse row with new extraction (should make API call)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ParseConfig(
                parsed_path=temp_dir,
                local_tables_path="/tmp/local_tables",
                metadata_table_name="test_metadata",
                tags_table_name="test_tags",
                example_path="examples",  # Use real examples path
                fm_endpoint="test-model",
                metadata_prompt="test",
                metadata_example="test",
                tag_prompt="test",
                tag_example="test",
                num_few_shot_examples=0,  # Disable few shot
            )

            # Mock the API response
            self.mock_handler.make_request.return_value = (
                '{"tag": "new_value", "confidence": 0.9}'
            )

            processor = ImageProcessor(self.mock_handler, config)

            # Create test image
            test_image = Path(temp_dir) / "test_image.jpg"
            test_image.write_bytes(b"fake image data")

            test_row = {
                "unique_key": "new_test_key",
                "tile_path": str(test_image),
                "page_number": 1,
                "tile_number": 1,
            }

            with patch("src.parser.datetime") as mock_datetime:
                mock_datetime.now.return_value.isoformat.return_value = (
                    "2024-01-01T12:00:00"
                )

                result_row = processor._parse_row(test_row, "tag")
                raw_response = result_row.get("raw_response")

            # Should have called the API
            assert self.mock_handler.make_request.called

            # Should return parsed data
            expected_result = {"tag": "new_value", "confidence": 0.9}
            assert result_row["parsed_tag"] == expected_result
            assert raw_response == '{"tag": "new_value", "confidence": 0.9}'

            # Check files were created
            output_file = processor.output_dir / "new_test_key.json"
            raw_file = processor.raw_response_dir / "new_test_key_tag_raw.json"

            assert output_file.exists()
            assert raw_file.exists()

            # Check output file contents
            with open(output_file, "r") as f:
                saved_data = json.load(f)

            assert saved_data["tag"] == "new_value"
            assert saved_data["confidence"] == 0.9
            assert "_raw_response" in saved_data

    def test_load_image(self):
        """Test image loading and base64 encoding with real file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ParseConfig(
                parsed_path=temp_dir,
                local_tables_path="/tmp/local_tables",
                metadata_table_name="test_metadata",
                tags_table_name="test_tags",
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
                parsed_path=temp_dir,
                local_tables_path="/tmp/local_tables",
                metadata_table_name="test_metadata",
                tags_table_name="test_tags",
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
                parsed_path=temp_dir,
                local_tables_path="/tmp/local_tables",
                metadata_table_name="test_metadata",
                tags_table_name="test_tags",
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
                parsed_path=temp_dir,
                local_tables_path="/tmp/local_tables",
                metadata_table_name="test_metadata",
                tags_table_name="test_tags",
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
                parsed_path=temp_dir,
                local_tables_path="/tmp/local_tables",
                metadata_table_name="test_metadata",
                tags_table_name="test_tags",
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
                parsed_path=temp_dir,
                local_tables_path="/tmp/local_tables",
                metadata_table_name="test_metadata",
                tags_table_name="test_tags",
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
        assert config.max_retries == 2  # Updated to match config.yaml
