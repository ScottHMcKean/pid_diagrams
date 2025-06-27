from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
from pathlib import Path
from typing import List, Tuple


def get_token(workspace_client: WorkspaceClient) -> str:
    """
    Get the token for the Databricks workspace.
    """
    try:
        token = (
            workspace_client.dbutils.notebook.entry_point.getDbutils()
            .notebook()
            .getContext()
            .apiToken()
            .get()
        )
    except:
        token = workspace_client.config.token
    return token


def get_spark() -> DatabricksSession:
    """
    Check if Spark context is available.

    Returns:
        True if Spark context exists, False otherwise
    """
    try:
        spark = DatabricksSession.builder.serverless(True).getOrCreate()
        return spark
    except Exception as e:
        print(f"Error getting Spark session: {e}")
        return None


def get_page_and_tag_files(
    examples_path: str, suffix: str = ".json"
) -> Tuple[List[str], List[str]]:
    """Get page and tag files from examples directory or volume."""
    examples_path = Path(examples_path)
    all_files = list(examples_path.glob(f"*{suffix}"))
    tag_files = list(examples_path.glob(f"*_t*{suffix}"))
    page_files = list(set(all_files) - set(tag_files))
    return page_files, tag_files
