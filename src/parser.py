import base64
import json
import re
import time
from pathlib import Path
import logging
from abc import ABC, abstractmethod

class ParseConfig(BaseModel):
    output_path: str
    fm_endpoint: str
    temperature: float = 0.1
    top_p: float = 0.5
    max_retries: int = 3
    retry_delay_s: int = 1
    max_image_size_mb: float = 0.5
    metadata_prompt: str
    metadata_example: str
    tag_prompt: str
    tag_example: str

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
    
    def make_request(self, tile_image_data: str, prompt: str) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{tile_image_data}"},
                        },
                    ]
                }
            ],
            model=self.config.fm_endpoint,
            temperature=self.config.temperature,
            top_p=self.config.top_p
        )
        return chat_completion.choices[0].message.content

class ImageProcessor:
    def __init__(self, request_handler: OpenAIRequestHandler, config: ParseConfig):
        self.request_handler = request_handler
        self.config = config
        self.output_dir = Path(config.output_path)
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_image(self, image_path: str) -> str:
        """Load and encode image to base64"""
        return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    
    def _extract_json(self, response: str) -> str:
        """Robustly extract json from the response"""
        cleaned = response.strip()
    
        # remove markdown markers
        if cleaned.startswith('```'):
            first_newline = cleaned.find('\n')
            if first_newline != -1:
                cleaned = cleaned[first_newline + 1:]
        
        if cleaned.endswith('```')
            cleaned = cleaned[:-3]
        
        # Fix quoted numbers with hyphens
        cleaned = re.sub(r'"(\d+)"-([A-Z\-0-9]+)"', r'"\1-\2"', cleaned)
        
        # Remove trailing commas before closing braces/brackets
        cleaned = re.sub(r',(\s*[}$$])', r'\1', cleaned)
        
        # Fix single quotes to double quotes
        cleaned = re.sub(r"'([^']*)':", r'"\1":', cleaned)
        
        # Strip whitespace again
        cleaned = cleaned.strip()
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            # Look for content between first { and last }
            first_brace = cleaned.find('{')
            last_brace = cleaned.rfind('}')
            
            if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
                json_candidate = cleaned[first_brace:last_brace + 1]
                try:
                    return json.loads(json_candidate)
                except json.JSONDecodeError:
                    pass
            
            # if all fails, return response for manual fix
            return response 
    
    def _save_result(self, filename: str, data: Any) -> None:
        """Save result to file"""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
    
    def process_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Process image for metadata extraction"""
        return self._process_image(row, self.config.metadata_prompt, 'metadata')
    
    def process_tags(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Process image for tag extraction"""
        return self._process_image(row, self.config.tag_prompt, 'tags')
    
    def _process_image(self, row: Dict[str, Any], prompt: str, task_type: str) -> Dict[str, Any]:
        """Process a single image with retry logic"""
        image_path_key = 'tile_path' if 'tile_path' in row else 'page_path'
        tile_image_data = self._load_image(row[image_path_key])
        
        # Generate filename based on task type
        base_filename = Path(row[image_path_key]).name.replace('.jpg', '')
        label_filename = f"{base_filename}_{task_type}.json"
        
        for attempt in range(self.config.max_retries + 1):
            try:
                self.logger.info(f"Attempt {attempt + 1} for {row[image_path_key]} ({task_type})")
                
                # Make API request
                raw_response = self.request_handler.make_request(tile_image_data, prompt)
                
                # Fix and parse JSON
                fixed_json_str = self._fix_json_string(raw_response)
                parsed_dict = json.loads(fixed_json_str)
                
                # Save successful result
                self._save_result(label_filename, parsed_dict)
                row[f'{task_type}_info'] = parsed_dict
                
                self.logger.info(f"Successfully processed {label_filename}")
                break
                
            except json.JSONDecodeError as e:
                self.logger.warning(f"JSON parsing failed for {label_filename} "
                                  f"(attempt {attempt + 1}/{self.config.max_retries + 1}): {e}")
                
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay_s)
                else:
                    self.logger.error(f"Failed to parse {label_filename} after all attempts")
                    self._save_result(label_filename, fixed_json_str)
                    row[f'{task_type}_info'] = fixed_json_str
                    
            except Exception as e:
                self.logger.error(f"Non-parsing error for {label_filename}: {e}")
                self._save_result(label_filename, str(e))
                row[f'{task_type}_info'] = str(e)
                break
        
        return row

def load_config(config_path: str) -> ParseConfig:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    return ParseConfig(**config_data['parse'])