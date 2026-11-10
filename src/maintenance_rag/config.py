"""Chargement et validation de la configuration : config/config.yaml (paramètres) et variables
d'environnement (.env, secrets) via pydantic-settings.

Ce module expose pour l'instant :

- ``LayoutProfile`` / ``load_layout_profile`` : profil de mise en page d'une famille de PDF
  (section ``layout_profiles`` de config/config.yaml), utilisé par ``pdf_reader`` ;
- ``ManifestEntry`` / ``load_manifest`` : entrées validées de corpus/manifest.yaml.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"
MANIFEST_PATH = ROOT / "corpus" / "manifest.yaml"


class TextSignature(BaseModel):
    """Signature typographique d'une ligne : taille (pt), couleur (hex RRGGBB) et graisse.

    La police n'en fait pas partie : seules ces trois propriétés sont comparées.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    size: float = Field(gt=0)
    color: str = Field(pattern=r"^[0-9a-f]{6}$")
    bold: bool

    @field_validator("color", mode="before")
    @classmethod
    def _lower_color(cls, value: Any) -> Any:
        return value.lower().lstrip("#") if isinstance(value, str) else value


class LayoutProfile(BaseModel):
    """Profil de mise en page relevé par l'exploration (scripts/inspect_pdf.py).

    Les distances sont en points PDF.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Signatures des rôles typographiques (une ou plusieurs par rôle).
    chapter: list[TextSignature]
    section: list[TextSignature]
    subsection: list[TextSignature]
    intertitle: list[TextSignature]
    warning: list[TextSignature]
    caption: list[TextSignature]
    page_number: list[TextSignature]
    size_tolerance: float = Field(default=0.3, ge=0)
    color_tolerance: int = Field(default=8, ge=0, le=255)  # écart maximal par canal (0–255)

    warning_prefix: str = "Attention"
    glossary_chapter: str = "Définitions"

    # Géométrie de la page.
    margin_ratio: float = Field(default=0.12, gt=0, lt=0.5)
    column_tolerance: float = Field(default=10.0, ge=0)
    min_lines_per_column: int = Field(default=3, ge=1)
    paragraph_gap_margin: float = Field(default=1.5, ge=0)
    short_line_margin: float = Field(default=15.0, ge=0)
    list_indent: float = Field(default=2.0, ge=0)

    # Notes de bas de page.
    note_max_size: float = Field(default=8.5, gt=0)
    note_min_y_ratio: float = Field(default=0.75, gt=0, lt=1)

    # Dessins : encadrés, triangles d'intertitre, schémas, légendes de photos.
    frame_min_width: float = Field(default=40.0, gt=0)
    frame_min_height: float = Field(default=15.0, gt=0)
    arrow_max_size: float = Field(default=12.0, gt=0)
    arrow_max_distance: float = Field(default=12.0, ge=0)
    diagram_merge_distance: float = Field(default=40.0, ge=0)
    diagram_min_items: int = Field(default=3, ge=1)
    caption_max_gap: float = Field(default=12.0, ge=0)


class ManifestEntry(BaseModel):
    """Une entrée de corpus/manifest.yaml (un document source)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    title: str
    path: str
    language: str
    machine: str | None = None
    version: str | None = None
    license: str
    source_url: str | None = None
    page_offset: int = Field(ge=0)
    sha256: str | None = None
    layout_profile: str | None = None
    exclude_pages: list[int] = Field(default_factory=list)

    @field_validator("exclude_pages")
    @classmethod
    def _pages_positive(cls, pages: list[int]) -> list[int]:
        if any(page < 1 for page in pages):
            raise ValueError("exclude_pages : les pages PDF commencent à 1")
        return sorted(set(pages))


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} : un dictionnaire YAML est attendu")
    return data


def load_layout_profile(name: str, config_path: Path = CONFIG_PATH) -> LayoutProfile:
    """Charge le profil ``layout_profiles.<name>`` de la configuration."""
    profiles = _read_yaml(config_path).get("layout_profiles") or {}
    if name not in profiles:
        raise KeyError(f"profil de mise en page inconnu : {name!r} ({config_path})")
    return LayoutProfile.model_validate(profiles[name])


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> list[ManifestEntry]:
    """Charge et valide toutes les entrées du manifeste du corpus."""
    documents = _read_yaml(manifest_path).get("documents") or []
    return [ManifestEntry.model_validate(document) for document in documents]
