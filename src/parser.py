import base64
import json
import re
import time
from pathlib import Path
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import copy

from src.config import ParseConfig
from src.preprocess import load_image_w_max_size


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
            top_p=self.config.top_p,
        )
        return chat_completion.choices[0].message.content


class ImageProcessor:
    def __init__(self, request_handler: OpenAIRequestHandler, config: ParseConfig):
        self.request_handler = request_handler
        self.config = config
        self.output_dir = Path(config.output_path)
        self.logger = logging.getLogger(__name__)

        self._get_few_shot_paths()
        self._get_few_shot_content()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_few_shot_paths(self) -> List[str]:
        """Get few shot examples from the example path"""
        self.few_shot_paths = []
        if self.config.num_few_shot_examples > 0:
            self.few_shot_paths = list(Path(self.config.example_path).glob("*.jpg"))[
                : self.config.num_few_shot_examples
            ]

    def _get_few_shot_content(self) -> List[bytes]:
        """Get few shot images from the example paths"""
        self.few_shot_content = []
        for path in self.few_shot_paths:

            # load label
            with open(str(path).replace(".jpg", ".json"), "r") as f:
                label: str = json.dumps(json.load(f), indent=4)

            # load image
            image_data: bytes = self._load_image(str(path))

            self.few_shot_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                }
            )
            self.few_shot_content.append(
                {
                    "type": "text",
                    "text": label,
                }
            )
        return self.few_shot_content

    def make_parse_content(self, row: Dict[str, Any], task: str) -> str:
        """Make parse context for the row"""
        content = copy.deepcopy(self.few_shot_content)
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

    def _save_result(self, filename: str, data: Any) -> None:
        """Save result to file"""
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

    def _parse_row(self, row: Dict[str, Any], task: str = "tag") -> Dict[str, Any]:
        """Process a single image with retry logic"""
        # avoid side effects
        row = copy.deepcopy(row)

        assert task in ["tag", "metadata"]

        content = self.make_parse_content(row, task)

        prompt = (
            self.config.tag_prompt if task == "tag" else self.config.metadata_prompt
        )

        task_key = row["unique_key"]

        label_filename = f"{task_key}.json"
        if task == "metadata":
            # Remove _t# from label for metadata task
            label_filename = re.sub(r"_t\d+$", "", task_key)
            label_filename = label_filename + ".json"

        for attempt in range(self.config.max_retries + 1):
            try:
                self.logger.info(f"Attempt {attempt + 1} for {task_key}")
                raw_response = self.request_handler.make_request(prompt, content)
                parsed_dict = self._extract_json(raw_response)
                self._save_result(label_filename, parsed_dict)
                row[f"parsed_{task}"] = parsed_dict
                self.logger.info(f"Successfully processed {label_filename}")
                break
            except json.JSONDecodeError as e:
                self.logger.warning(
                    f"Parsing attempt {attempt + 1} failed for {task_key}"
                )

                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay_s)
                else:
                    self.logger.error(f"Max retries for {task_key} exceeded.")
                    self._save_result(label_filename, raw_response)
                    row[f"parsed_{task}"] = raw_response
            except Exception as e:
                self.logger.error(f"Non-parsing error for {label_filename}: {e}")
                self._save_result(label_filename, str(e))
                row[f"parsed_{task}"] = str(e)
                break
        return row
