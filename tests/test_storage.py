"""Tests for the Cloudflare R2 storage client."""

import pytest
from unittest.mock import MagicMock, patch
from botocore.exceptions import ClientError

from herald.storage.r2 import _build_r2_client, download_cv_bytes


class TestBuildR2Client:
    """Tests for _build_r2_client."""

    @patch.dict("os.environ", {
        "R2_ACCOUNT_ID": "test-account",
        "R2_ACCESS_KEY_ID": "test-key",
        "R2_SECRET_ACCESS_KEY": "test-secret",
    })
    @patch("herald.storage.r2.boto3.client")
    def test_builds_client_with_correct_endpoint(self, mock_boto_client):
        """Client endpoint URL is constructed from R2_ACCOUNT_ID."""
        _build_r2_client()
        mock_boto_client.assert_called_once_with(
            "s3",
            endpoint_url="https://test-account.r2.cloudflarestorage.com",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            region_name="auto",
        )

    @patch.dict("os.environ", {}, clear=True)
    def test_raises_when_all_credentials_missing(self):
        """ValueError raised when all R2 credential vars are absent."""
        with pytest.raises(ValueError, match="Missing R2 credential"):
            _build_r2_client()

    @patch.dict("os.environ", {
        "R2_ACCESS_KEY_ID": "test-key",
        "R2_SECRET_ACCESS_KEY": "test-secret",
    }, clear=True)
    def test_raises_when_account_id_missing(self):
        """ValueError lists the specific missing variable."""
        with pytest.raises(ValueError, match="R2_ACCOUNT_ID"):
            _build_r2_client()

    @patch.dict("os.environ", {
        "R2_ACCOUNT_ID": "test-account",
        "R2_SECRET_ACCESS_KEY": "test-secret",
    }, clear=True)
    def test_raises_when_access_key_missing(self):
        """ValueError lists the specific missing variable."""
        with pytest.raises(ValueError, match="R2_ACCESS_KEY_ID"):
            _build_r2_client()


class TestDownloadCvBytes:
    """Tests for download_cv_bytes."""

    @patch.dict("os.environ", {}, clear=True)
    def test_raises_when_bucket_not_set(self):
        """ValueError raised when R2_BUCKET_NAME is absent."""
        with pytest.raises(ValueError, match="R2_BUCKET_NAME"):
            download_cv_bytes()

    @patch.dict("os.environ", {
        "R2_ACCOUNT_ID": "acct",
        "R2_ACCESS_KEY_ID": "key",
        "R2_SECRET_ACCESS_KEY": "secret",
        "R2_BUCKET_NAME": "herald-cv",
        "CV_OBJECT_KEY": "cv.pdf",
    })
    @patch("herald.storage.r2._build_r2_client")
    def test_downloads_bytes_from_correct_bucket_and_key(self, mock_build_client):
        """get_object called with the configured bucket and key."""
        fake_pdf = b"%PDF-1.4 fake content"
        mock_client = MagicMock()
        mock_client.get_object.return_value = {"Body": MagicMock(read=lambda: fake_pdf)}
        mock_build_client.return_value = mock_client

        result = download_cv_bytes()

        mock_client.get_object.assert_called_once_with(Bucket="herald-cv", Key="cv.pdf")
        assert result == fake_pdf

    @patch.dict("os.environ", {
        "R2_ACCOUNT_ID": "acct",
        "R2_ACCESS_KEY_ID": "key",
        "R2_SECRET_ACCESS_KEY": "secret",
        "R2_BUCKET_NAME": "herald-cv",
    })
    @patch("herald.storage.r2._build_r2_client")
    def test_default_object_key_is_cv_pdf(self, mock_build_client):
        """CV_OBJECT_KEY defaults to 'cv.pdf' when not set."""
        mock_client = MagicMock()
        mock_client.get_object.return_value = {"Body": MagicMock(read=lambda: b"pdf")}
        mock_build_client.return_value = mock_client

        download_cv_bytes()

        _, kwargs = mock_client.get_object.call_args
        assert kwargs["Key"] == "cv.pdf"

    @patch.dict("os.environ", {
        "R2_ACCOUNT_ID": "acct",
        "R2_ACCESS_KEY_ID": "key",
        "R2_SECRET_ACCESS_KEY": "secret",
        "R2_BUCKET_NAME": "herald-cv",
    })
    @patch("herald.storage.r2._build_r2_client")
    def test_propagates_client_error(self, mock_build_client):
        """ClientError from boto3 propagates to the caller."""
        mock_client = MagicMock()
        mock_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}, "GetObject"
        )
        mock_build_client.return_value = mock_client

        with pytest.raises(ClientError):
            download_cv_bytes()
