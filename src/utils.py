def _is_spark_available() -> bool:
    """
    Check if Spark context is available.

    Returns:
        True if Spark context exists, False otherwise
    """
    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        return spark is not None
    except ImportError:
        return False
    except Exception:
        return False
