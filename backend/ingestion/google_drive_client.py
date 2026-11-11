"""Google Drive and Docs API client for the RAG ingestion pipeline.

This module provides an async-friendly wrapper around the Google Drive v3
and Google Docs v1 REST APIs, using a service-account credential for
server-to-server authentication.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import structlog
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from backend.core.config import get_settings

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
_GOOGLE_DOC_MIME = "application/vnd.google-apps.document"
_SUPPORTED_MIMES = {
    _GOOGLE_DOC_MIME,
    "application/pdf",
    "text/plain",
    "text/markdown",
}


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class GoogleDriveError(Exception):
    """Raised when a Google Drive or Docs API call fails."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        """Initialise GoogleDriveError.

        Args:
            message: Human-readable description of the error.
            status_code: HTTP status code returned by the API, if available.
        """
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class GoogleDriveClient:
    """Async-friendly client for Google Drive v3 and Google Docs v1 APIs.

    Wraps all blocking ``googleapiclient`` calls in :func:`asyncio.to_thread`
    so they don't block the event loop.

    Example::

        client = GoogleDriveClient()
        files = await client.get_all_files_in_folder(folder_id)
    """

    def __init__(self) -> None:
        """Load service-account credentials and build API service clients.

        Reads ``GOOGLE_SERVICE_ACCOUNT_JSON`` from application settings,
        parses the JSON key, and constructs scoped credentials used to build
        the Drive v3 and Docs v1 service objects.

        Raises:
            GoogleDriveError: If the service account JSON is missing, malformed,
                or the API client cannot be constructed.
        """
        settings = get_settings()
        log = logger.bind(component="GoogleDriveClient.__init__")

        raw_config: str = settings.GOOGLE_SERVICE_ACCOUNT_JSON.strip()
        if not raw_config:
            raise GoogleDriveError("GOOGLE_SERVICE_ACCOUNT_JSON setting is empty.")

        # Support both file path (e.g. ./credentials/service_account.json) and raw JSON string
        candidate_paths = [
            Path(raw_config),
            Path.cwd() / raw_config,
            Path("/app") / raw_config if not Path(raw_config).is_absolute() else None,
        ]
        file_path = next((p for p in candidate_paths if p and p.is_file()), None)

        try:
            if file_path:
                log.info("loading_service_account_from_file", path=str(file_path))
                credentials = service_account.Credentials.from_service_account_file(
                    str(file_path), scopes=_DRIVE_SCOPES
                )
            else:
                log.info("loading_service_account_from_raw_json")
                sa_info: dict[str, Any] = json.loads(raw_config)
                credentials = service_account.Credentials.from_service_account_info(
                    sa_info, scopes=_DRIVE_SCOPES
                )

            self._drive = build("drive", "v3", credentials=credentials, cache_discovery=False)
            self._docs = build("docs", "v1", credentials=credentials, cache_discovery=False)
        except Exception as exc:
            raise GoogleDriveError(
                f"Failed to load Google service account credentials: {exc}"
            ) from exc

        log.info("Google Drive and Docs clients initialised")

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def list_changed_files(
        self, page_token: str | None
    ) -> tuple[list[dict[str, Any]], str]:
        """Fetch files changed since the given Drive changes page token.

        Uses ``drive.changes().list()`` with the provided *page_token* to
        retrieve incremental changes in the authenticated user's Drive.

        Args:
            page_token: The start-page token from a previous call, or
                ``None`` to begin a new change listing session. When
                ``None``, a new start page token is fetched first.

        Returns:
            A two-tuple ``(file_list, next_page_token)`` where *file_list*
            is a list of file-metadata dicts (may be empty) and
            *next_page_token* is the token to pass on the next call.

        Raises:
            GoogleDriveError: On any API-level error.
        """
        log = logger.bind(method="list_changed_files", page_token=page_token)
        log.debug("Listing changed files")

        def _fetch_start_token() -> str:
            resp = (
                self._drive.changes()
                .getStartPageToken()
                .execute(num_retries=3)
            )
            return resp["startPageToken"]

        def _list_changes(token: str) -> dict[str, Any]:
            return (
                self._drive.changes()
                .list(
                    pageToken=token,
                    spaces="drive",
                    fields=(
                        "nextPageToken,newStartPageToken,"
                        "changes(type,removed,fileId,"
                        "file(id,name,mimeType,modifiedTime,webViewLink,owners,permissions))"
                    ),
                )
                .execute(num_retries=3)
            )

        try:
            if page_token is None:
                page_token = await asyncio.to_thread(_fetch_start_token)
                log.info("Obtained fresh start page token", token=page_token)

            response = await asyncio.to_thread(_list_changes, page_token)
        except HttpError as exc:
            raise GoogleDriveError(
                f"Drive changes.list failed: {exc.reason}", status_code=exc.status_code
            ) from exc

        changes: list[dict[str, Any]] = response.get("changes", [])
        # Only keep non-removed file changes
        file_list: list[dict[str, Any]] = [
            ch["file"]
            for ch in changes
            if not ch.get("removed") and ch.get("type") == "file" and "file" in ch
        ]

        next_token: str = response.get("nextPageToken") or response.get(
            "newStartPageToken", page_token
        )

        log.info(
            "Changed files fetched",
            total_changes=len(changes),
            file_changes=len(file_list),
            next_page_token=next_token,
        )
        return file_list, next_token

    async def get_all_files_in_folder(
        self, folder_id: str
    ) -> list[dict[str, Any]]:
        """Recursively list all Google Docs files within a Drive folder.

        Performs a breadth-first traversal of the folder hierarchy, returning
        only files whose MIME type is ``application/vnd.google-apps.document``.

        Args:
            folder_id: The Drive folder ID to start the traversal from.

        Returns:
            A list of file-metadata dicts, each containing ``id``, ``name``,
            ``mimeType``, ``modifiedTime``, ``webViewLink``, ``owners``, and
            ``permissions``.

        Raises:
            GoogleDriveError: On any API-level error.
        """
        log = logger.bind(method="get_all_files_in_folder", folder_id=folder_id)
        log.info("Starting recursive folder listing")

        def _list_children(parent_id: str, page_token_inner: str | None) -> dict[str, Any]:
            query = f"'{parent_id}' in parents and trashed = false"
            kwargs: dict[str, Any] = {
                "q": query,
                "spaces": "drive",
                "fields": (
                    "nextPageToken,"
                    "files(id,name,mimeType,modifiedTime,webViewLink,owners,permissions)"
                ),
                "pageSize": 1000,
            }
            if page_token_inner:
                kwargs["pageToken"] = page_token_inner
            return self._drive.files().list(**kwargs).execute(num_retries=3)

        all_docs: list[dict[str, Any]] = []
        folders_to_visit: list[str] = [folder_id]

        try:
            while folders_to_visit:
                current_folder = folders_to_visit.pop(0)
                page_token_inner: str | None = None
                while True:
                    response = await asyncio.to_thread(
                        _list_children, current_folder, page_token_inner
                    )
                    items: list[dict[str, Any]] = response.get("files", [])
                    for item in items:
                        mime = item.get("mimeType", "")
                        if mime in _SUPPORTED_MIMES:
                            all_docs.append(item)
                        elif mime == "application/vnd.google-apps.folder":
                            folders_to_visit.append(item["id"])

                    page_token_inner = response.get("nextPageToken")
                    if not page_token_inner:
                        break
        except HttpError as exc:
            raise GoogleDriveError(
                f"Drive files.list failed: {exc.reason}", status_code=exc.status_code
            ) from exc

        log.info("Folder listing complete", total_docs=len(all_docs))
        return all_docs

    async def get_document_as_markdown(
        self, file_id: str, mime_type: str = _GOOGLE_DOC_MIME
    ) -> str:
        """Export or extract text from a Drive file (Google Doc, PDF, or text).

        Args:
            file_id: The Drive file ID.
            mime_type: The MIME type of the file.

        Returns:
            The extracted plain-text content as a UTF-8 string.

        Raises:
            GoogleDriveError: If the export or download request fails.
        """
        log = logger.bind(method="get_document_as_markdown", file_id=file_id, mime_type=mime_type)
        log.debug("Extracting document content")

        def _fetch_content() -> str:
            if mime_type == _GOOGLE_DOC_MIME:
                raw: bytes = (
                    self._drive.files()
                    .export(fileId=file_id, mimeType="text/plain")
                    .execute(num_retries=3)
                )
                return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
            elif mime_type == "application/pdf":
                import io

                from googleapiclient.http import MediaIoBaseDownload
                from pypdf import PdfReader

                req = self._drive.files().get_media(fileId=file_id)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, req)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                fh.seek(0)
                reader = PdfReader(fh)
                pages = [page.extract_text() or "" for page in reader.pages]
                return "\n\n".join(pages)
            else:
                raw = self._drive.files().get_media(fileId=file_id).execute()
                return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)

        try:
            text = await asyncio.to_thread(_fetch_content)
        except Exception as exc:
            raise GoogleDriveError(
                f"Failed to fetch content for file {file_id}: {exc}"
            ) from exc

        log.debug("Document content extracted", chars=len(text))
        return text

    async def get_file_permissions(
        self, file_id: str
    ) -> tuple[list[str], str | None]:
        """Fetch and parse sharing permissions for a Drive file.

        Calls ``drive.permissions().list()`` and classifies each permission as
        either an individual email address or a domain-wide grant.

        Args:
            file_id: The Drive file ID whose permissions to retrieve.

        Returns:
            A two-tuple ``(allowed_emails, allowed_domain)`` where
            *allowed_emails* is the list of individual email addresses with
            access, and *allowed_domain* is the first domain-wide grant found
            (or ``None`` if no domain permissions exist).

        Raises:
            GoogleDriveError: If the permissions list request fails.
        """
        log = logger.bind(method="get_file_permissions", file_id=file_id)
        log.debug("Fetching file permissions")

        def _list_permissions() -> dict[str, Any]:
            return (
                self._drive.permissions()
                .list(
                    fileId=file_id,
                    fields="permissions(id,type,emailAddress,domain,role)",
                    supportsAllDrives=True,
                )
                .execute(num_retries=3)
            )

        try:
            response = await asyncio.to_thread(_list_permissions)
        except HttpError as exc:
            raise GoogleDriveError(
                f"Drive permissions.list failed for {file_id}: {exc.reason}",
                status_code=exc.status_code,
            ) from exc

        permissions: list[dict[str, Any]] = response.get("permissions", [])
        allowed_emails: list[str] = []
        allowed_domain: str | None = None

        for perm in permissions:
            perm_type = perm.get("type", "")
            if perm_type == "user":
                email = perm.get("emailAddress")
                if email:
                    allowed_emails.append(email)
            elif perm_type == "group":
                email = perm.get("emailAddress")
                if email:
                    allowed_emails.append(email)
            elif perm_type == "domain" and allowed_domain is None:
                allowed_domain = perm.get("domain")
            elif perm_type == "anyone":
                # Publicly shared — represent as wildcard domain
                allowed_domain = allowed_domain or "*"

        log.debug(
            "Permissions resolved",
            email_count=len(allowed_emails),
            domain=allowed_domain,
        )
        return allowed_emails, allowed_domain
