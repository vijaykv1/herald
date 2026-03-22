"""Cloudflare R2 storage client for Herald.

Downloads the CV PDF from a Cloudflare R2 bucket using the S3-compatible API.
boto3 is used as the client since R2 is S3-compatible.

Required environment variables:
    R2_ACCOUNT_ID        - Cloudflare account ID (shown on the R2 overview page)
    R2_ACCESS_KEY_ID     - R2 API token access key
    R2_SECRET_ACCESS_KEY - R2 API token secret key
    R2_BUCKET_NAME       - Name of the R2 bucket holding the CV
    CV_OBJECT_KEY        - Object key of the CV file (default: cv.pdf)
"""

import logging
import os

import boto3

logger = logging.getLogger(__name__)


def _build_r2_client():
    """Build and return a boto3 client pointed at Cloudflare R2.

    :raises ValueError: If any required R2 credential env var is missing.
    :return: Configured boto3 S3 client
    """
    account_id = os.getenv("R2_ACCOUNT_ID")
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")

    missing = [name for name, val in [
        ("R2_ACCOUNT_ID", account_id),
        ("R2_ACCESS_KEY_ID", access_key),
        ("R2_SECRET_ACCESS_KEY", secret_key),
    ] if not val]

    if missing:
        raise ValueError(
            f"Missing R2 credential environment variable(s): {', '.join(missing)}"
        )

    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )


def download_cv_bytes() -> bytes:
    """Download the CV PDF from Cloudflare R2 and return its raw bytes.

    :raises ValueError: If R2_BUCKET_NAME is not set or credentials are missing.
    :return: PDF file contents as bytes
    :rtype: bytes
    """
    bucket = os.getenv("R2_BUCKET_NAME")
    object_key = os.getenv("CV_OBJECT_KEY", "cv.pdf")

    if not bucket:
        raise ValueError("R2_BUCKET_NAME environment variable is not set.")

    client = _build_r2_client()

    logger.info("Downloading CV from R2 bucket '%s', key '%s'", bucket, object_key)
    response = client.get_object(Bucket=bucket, Key=object_key)
    pdf_bytes = response["Body"].read()
    logger.info("CV downloaded successfully (%d bytes)", len(pdf_bytes))

    return pdf_bytes
