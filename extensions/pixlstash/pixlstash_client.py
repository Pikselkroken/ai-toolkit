"""
PixlStash API client — corrected against live API (v1.0.0b3).

Key endpoint notes (verified from /redoc):
  - Picture sets:    /picture_sets/{id}  (underscore, not hyphen)
  - Picture file:    /pictures/{id}.{ext}  (not /pictures/{id}/image.{ext})
  - Thumbnail:       /pictures/thumbnails/{id}.webp
  - Set members:     /picture_sets/{id}/members  -> returns integer IDs only
  - Char pictures:   /pictures/list?character_id={id}
  - Login:           POST /login  {"token": "..."}
"""

from __future__ import annotations

from typing import List, Optional

import requests

_EMPTY_TAG_SENTINEL = ""


class PixlStashError(RuntimeError):
    """Raised when the PixlStash server returns an unexpected response."""


class PixlStashClient:
    """Thin wrapper around the PixlStash REST API."""

    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._session = requests.Session()
        self._authenticated = False

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def login(self) -> None:
        """Exchange the API token for a session cookie."""
        r = self._session.post(
            f"{self.base_url}/login",
            json={"token": self.token},
            timeout=30,
        )
        if not r.ok:
            raise PixlStashError(
                f"Login failed ({r.status_code}): {r.text[:200]}"
            )
        self._authenticated = True

    def _get(self, path: str, **params) -> requests.Response:
        if not self._authenticated:
            raise PixlStashError("Call login() before making API requests.")
        r = self._session.get(f"{self.base_url}{path}", params=params or None, timeout=60)
        if not r.ok:
            raise PixlStashError(
                f"GET {path} failed ({r.status_code}): {r.text[:200]}"
            )
        return r

    # ------------------------------------------------------------------
    # Characters
    # ------------------------------------------------------------------

    def get_character(self, character_id: int) -> dict:
        return self._get(f"/characters/{character_id}").json()

    def list_characters(self, name: Optional[str] = None) -> List[dict]:
        params = {}
        if name:
            params["name"] = name
        return self._get("/characters", **params).json()

    # ------------------------------------------------------------------
    # Picture sets
    # ------------------------------------------------------------------

    def get_picture_set(self, set_id: int) -> dict:
        """Return picture set metadata."""
        return self._get(f"/picture_sets/{set_id}", info=True).json()

    def list_picture_sets(self) -> List[dict]:
        """Return all non-reference picture sets."""
        all_sets = self._get("/picture_sets").json()
        # Reference sets are auto-created per character for face recognition;
        # they have a non-null `reference_character` field and are not useful
        # as training datasets.
        return [s for s in all_sets if s.get("reference_character") is None]

    def get_picture_set_member_ids(self, set_id: int) -> List[int]:
        """Return the list of picture IDs belonging to a set."""
        # Response shape: {"picture_ids": [...]}
        return self._get(f"/picture_sets/{set_id}/members").json()["picture_ids"]

    # ------------------------------------------------------------------
    # Picture listing
    # ------------------------------------------------------------------

    def list_pictures_for_character(self, character_id: int) -> List[dict]:
        """Return all pictures where this character appears."""
        # GET /pictures?character_id={id} returns full picture objects
        return self._get("/pictures", character_id=character_id).json()

    def list_pictures_for_set(self, set_id: int) -> List[dict]:
        """
        Return full picture metadata records for every member of a set.
        The members endpoint returns IDs only, so we fetch metadata for each.
        """
        pic_ids: List[int] = self.get_picture_set_member_ids(set_id)
        return [self.get_picture_metadata(pid) for pid in pic_ids]

    def get_picture_metadata(self, pic_id: int) -> dict:
        return self._get(f"/pictures/{pic_id}/metadata").json()

    # ------------------------------------------------------------------
    # Image download
    # ------------------------------------------------------------------

    def download_image_bytes(self, pic_id: int, fmt: str = "jpg") -> bytes:
        """Return raw image bytes. Endpoint: GET /pictures/{id}.{ext}"""
        r = self._session.get(
            f"{self.base_url}/pictures/{pic_id}.{fmt}",
            timeout=120,
        )
        if not r.ok:
            raise PixlStashError(
                f"Image download for id={pic_id} failed ({r.status_code})"
            )
        return r.content

    # ------------------------------------------------------------------
    # Thumbnail URLs (used by the UI browse modal)
    # ------------------------------------------------------------------

    def thumbnail_url(self, pic_id: int) -> str:
        """Full URL for a picture's WebP thumbnail."""
        return f"{self.base_url}/pictures/thumbnails/{pic_id}.webp"

    def picture_set_thumbnail_url(self, set_id: int) -> str:
        return f"{self.base_url}/picture_sets/{set_id}/thumbnail"

    def character_thumbnail_url(self, character_id: int) -> str:
        return f"{self.base_url}/characters/{character_id}/thumbnail"

    # ------------------------------------------------------------------
    # Caption building
    # ------------------------------------------------------------------

    @staticmethod
    def tags_to_string(meta: dict) -> str:
        return ", ".join(
            t["tag"]
            for t in (meta.get("tags") or [])
            if t.get("tag") and t["tag"] != _EMPTY_TAG_SENTINEL
        )

    @classmethod
    def build_caption(
        cls,
        meta: dict,
        mode: str = "description",
        trigger: str = "",
    ) -> str:
        """
        Build a caption string for one picture.

        mode: "description" | "tags" | "both"
        trigger: optional token prepended to every caption.
        """
        parts: List[str] = []

        if trigger:
            parts.append(trigger)

        if mode in ("description", "both"):
            desc = (meta.get("description") or "").strip()
            if desc:
                parts.append(desc)

        if mode in ("tags", "both"):
            tag_str = cls.tags_to_string(meta)
            if tag_str:
                parts.append(tag_str)

        content_count = len(parts) - (1 if trigger else 0)
        if content_count == 0:
            fallback = (meta.get("description") or "").strip() or cls.tags_to_string(meta)
            if fallback:
                parts.append(fallback)

        return ", ".join(parts)
