import base64
import json
import random
import re
import time
from pathlib import Path
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import copy
from datetime import datetime

from src.config import ParseConfig
from src.preprocess import load_image_w_max_size
from src.utils import get_page_and_tag_files

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class ParsingHandler(ABC):
    """Abstract base class for different API request handlers"""

    @abstractmethod
    def make_request(self, tile_image_data: str, prompt: str) -> str:
        """Make API request and return raw response text"""
        pass


class OpenAIRequestHandler:
    def __init__(self, client, config: ParseConfig):
        self.client = client
        self.config = config

    def make_request(self, prompt: str, content: List[Dict[str, Any]]) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
            model=self.config.fm_endpoint,
            temperature=self.config.temperature,
            extra_body={
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": self.config.thinking_budget_tokens,
                }
            },
        )
        return chat_completion.choices[0].message.content


class ImageProcessor:
    def __init__(self, request_handler: OpenAIRequestHandler, config: ParseConfig):
        self.request_handler = request_handler
        self.config = config
        self.output_dir = Path(config.parsed_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create raw response directory
        self.raw_response_dir = self.output_dir / "raw_responses"
        self.raw_response_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)

        self.page_files, self.tag_files = get_page_and_tag_files(
            self.config.example_path
        )

    def _check_existing_extraction(
        self, label_filename: str
    ) -> Optional[Dict[str, Any]]:
        """Check if extraction already exists and return it if found"""
        filepath = self.output_dir / label_filename
        if filepath.exists():
            try:
                with open(filepath, "r") as f:
                    existing_data = json.load(f)
                return existing_data
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(
                    f"Failed to load existing extraction {filepath}: {e}"
                )
                return None
        return None

    def _save_raw_response(
        self, task_key: str, task: str, raw_response: str, metadata: Dict[str, Any]
    ) -> None:
        """Save raw response with metadata to raw_response directory"""
        timestamp = datetime.now().isoformat()
        raw_response_data = {
            "task_key": task_key,
            "task": task,
            "timestamp": timestamp,
            "metadata": metadata,
            "raw_response": raw_response,
        }

        raw_filename = f"{task_key}_{task}_raw.json"
        raw_filepath = self.raw_response_dir / raw_filename

        with open(raw_filepath, "w") as f:
            json.dump(raw_response_data, f, indent=4)

        self.logger.info(f"Raw response saved to: {raw_filepath}")

    def _get_few_shot_content(self, task: str) -> List[bytes]:
        """Get few shot images from the example paths"""
        if self.config.num_few_shot_examples == 0:
            return []

        # get few shot paths
        if task == "tag":
            few_shot_paths = self.tag_files
        elif task == "metadata":
            few_shot_paths = []
            for page_file in self.page_files:
                matching_tag_files = [
                    file
                    for file in self.tag_files
                    if file.stem.startswith(page_file.stem)
                ]
                # The last tag file contains the title block (lower right)
                last_tag_file = (
                    max(matching_tag_files, key=lambda x: int(x.stem.split("_t")[-1]))
                    if matching_tag_files
                    else None
                )
                few_shot_paths.append(last_tag_file)

        # limit few shot examples
        # TODO: Replace this with VLM embedding lookups eventually
        example_paths = random.sample(
            few_shot_paths, min(len(few_shot_paths), self.config.num_few_shot_examples)
        )

        few_shot_content = []
        for path in example_paths:

            # image
            image_data: bytes = self._load_image(str(path).replace(".json", ".jpg"))
            few_shot_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                }
            )

            # label
            with open(str(path), "r") as f:
                label: str = json.dumps(json.load(f), indent=4)
            few_shot_content.append(
                {
                    "type": "text",
                    "text": label,
                }
            )

        return few_shot_content

    def make_parse_content(self, row: Dict[str, Any], task: str) -> str:
        """Make parse context for the row"""
        content = self._get_few_shot_content(task)

        # if no few shot content, use base example
        if not content:
            self.logger.warning("No few shot content found, using base example")
            example_text = (
                self.config.tag_example
                if task == "tag"
                else self.config.metadata_example
            )
            content.append({"type": "text", "text": example_text})

        if task == "tag":
            tile_image_data = self._load_image(row["tile_path"])
            content.extend(
                [
                    {
                        "type": "text",
                        "text": "Here is the tile image to extract tags from:",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{tile_image_data}"
                        },
                    },
                ]
            )
        elif task == "metadata":
            page_image_data = self._load_image(row["page_path"])
            tile_image_data = self._load_image(row["tile_path"])
            content.extend(
                [
                    {
                        "type": "text",
                        "text": "Here is the entire diagram and a zoomed title block to extract metadata from:",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{page_image_data}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{tile_image_data}"
                        },
                    },
                ]
            )
        return content

    def _load_image(self, image_path: str) -> str:
        """Load and encode image to base64"""
        return load_image_w_max_size(image_path)

    def _extract_json(self, response: str) -> str:
        """Robustly extract json from the response"""
        cleaned = response.strip()

        # remove markdown markers
        if cleaned.startswith("```"):
            first_newline = cleaned.find("\n")
            if first_newline != -1:
                cleaned = cleaned[first_newline + 1 :]

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        # Fix quoted numbers with hyphens
        cleaned = re.sub(r'"(\d+)"-([A-Z\-0-9]+)"', r'"\1-\2"', cleaned)

        # Remove trailing commas before closing braces/brackets
        cleaned = re.sub(r",(\s*[}$$])", r"\1", cleaned)

        # Fix single quotes to double quotes
        cleaned = re.sub(r"'([^']*)':", r'"\1":', cleaned)

        # Strip whitespace again
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            # Look for content between first { and last }
            first_brace = cleaned.find("{")
            last_brace = cleaned.rfind("}")

            if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
                json_candidate = cleaned[first_brace : last_brace + 1]
                try:
                    return json.loads(json_candidate)
                except json.JSONDecodeError:
                    pass

            # if all fails, return response for manual fix
            return response

    def _save_result(
        self, filename: str, data: Any, raw_response: Optional[str] = None
    ) -> None:
        """Save result to file, optionally including raw response"""
        # Create a copy to avoid modifying the original data
        save_data = copy.deepcopy(data)

        # Add raw response to saved data if provided
        if raw_response is not None:
            save_data["_raw_response"] = raw_response

        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(save_data, f, indent=4)

    def _parse_row(self, row: Dict[str, Any], task: str = "tag") -> Dict[str, Any]:
        """Process a single image with retry logic"""
        # avoid side effects
        row = copy.deepcopy(row)

        assert task in ["tag", "metadata"]

        task_key = row["unique_key"]
        page_number = row.get("page_number", "Unknown")
        tile_number = row.get("tile_number", "Unknown")

        # Determine output filename
        label_filename = f"{task_key}.json"
        if task == "metadata":
            # Remove _t# from label for metadata task
            label_filename = re.sub(r"_t\d+$", "", task_key)
            label_filename = label_filename + ".json"

        # Check if extraction already exists
        existing_extraction = self._check_existing_extraction(label_filename)
        if existing_extraction is not None:
            self.logger.info(
                f"Existing {task} extraction found for {task_key}, skipping API call"
            )
            # Remove raw response from returned data if it exists
            clean_extraction = {
                k: v for k, v in existing_extraction.items() if k != "_raw_response"
            }
            row[f"parsed_{task}"] = clean_extraction
            return row, None  # No raw response since we didn't make a request

        self.logger.info(f"Starting {task} parsing for document: {task_key}")
        self.logger.info(f"  Page: {page_number}, Tile: {tile_number}")

        content = self.make_parse_content(row, task)
        prompt = (
            self.config.tag_prompt if task == "tag" else self.config.metadata_prompt
        )

        # Metadata for raw response
        metadata = {
            "page_number": page_number,
            "tile_number": tile_number,
            "task": task,
            "prompt_used": (
                prompt[:100] + "..." if len(prompt) > 100 else prompt
            ),  # Truncate for readability
        }

        raw_response = None
        for attempt in range(self.config.max_retries + 1):
            try:
                self.logger.info(
                    f"  Attempt {attempt + 1}/{self.config.max_retries + 1} for {task_key}"
                )
                raw_response = self.request_handler.make_request(prompt, content)

                # Save raw response with metadata
                self._save_raw_response(task_key, task, raw_response, metadata)

                parsed_dict = self._extract_json(raw_response)

                # Save result with raw response included in file
                self._save_result(label_filename, parsed_dict, raw_response)

                # Return clean parsed dict without raw response
                row[f"parsed_{task}"] = parsed_dict
                self.logger.info(f"  Success")
                break
            except json.JSONDecodeError as e:
                if attempt < self.config.max_retries:
                    self.logger.warning(
                        f"  JSON parsing failed. Retrying in {self.config.retry_delay_s} seconds..."
                    )
                    time.sleep(self.config.retry_delay_s)
                else:
                    self.logger.error(
                        f"  Max retries exceeded for {task_key}. Saving raw response."
                    )
                    # Save raw response even on failure
                    if raw_response:
                        self._save_raw_response(task_key, task, raw_response, metadata)
                    self._save_result(label_filename, raw_response)
                    row[f"parsed_{task}"] = raw_response
            except Exception as e:
                self.logger.error(f"  Non-parsing error for {task_key}: {str(e)}")
                # Save raw response on any error
                if raw_response:
                    self._save_raw_response(task_key, task, raw_response, metadata)
                self._save_result(label_filename, str(e))
                row[f"parsed_{task}"] = str(e)
                break
        return row, raw_response


def save_raw_responses_to_json(
    raw_responses: list,
    output_path: Path = Path("outputs/raw_responses.json"),
) -> None:
    """
    Save raw responses to JSON file for debugging/analysis.

    Args:
        raw_responses: List of raw responses
        output_path: Full path to the output JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    responses_data = []
    for i, response in enumerate(raw_responses):
        response_entry = {
            "response_number": i + 1,
            "raw_response": response,
        }
        responses_data.append(response_entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(responses_data, f, indent=2, ensure_ascii=False)

    print(f"Raw responses saved to: {output_path}")
