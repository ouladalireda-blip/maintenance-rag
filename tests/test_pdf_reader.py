"""Tests du lecteur PDF structuré.

Le PDF de test est SYNTHÉTIQUE : il est généré par PyMuPDF dans un répertoire temporaire, avec
les signatures du profil ``inrs`` de config/config.yaml. Aucun texte de l'INRS n'est écrit.
Un dernier test lit le vrai PDF INRS s'il est présent localement (jamais versionné).
"""

from pathlib import Path

import pymupdf
import pytest

from maintenance_rag.config import (
    ROOT,
    LayoutProfile,
    ManifestEntry,
    load_layout_profile,
    load_manifest,
)
from maintenance_rag.pdf_reader import Block, read_pdf

BLACK = (0.0, 0.0, 0.0)
CHAPTER_BLUE = (0x49 / 255, 0x8C / 255, 0xA7 / 255)
SECTION_GREEN = (0x79 / 255, 0x9A / 255, 0x2C / 255)
SUBSECTION_BLUE = (0x35 / 255, 0x6D / 255, 0x83 / 255)
WARNING_RED = (0xD3 / 255, 0x23 / 255, 0x2A / 255)

LEFT, RIGHT = 60.0, 310.0  # bords gauches des deux colonnes
COLUMN_WIDTH = 220.0

CHAPTER = "3 Chapitre de test"
SECTION_1 = "3.1. Section gauche"
SECTION_2 = "3.2. Seconde section traversant toute la largeur de la page"
SUBSECTION = "3.1.1. Sous-section droite"


def _put(
    page: pymupdf.Page,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 11,
    bold: bool = False,
    color: tuple[float, float, float] = BLACK,
    max_width: float | None = COLUMN_WIDTH,
) -> None:
    font = "hebo" if bold else "helv"
    if max_width is not None:  # garantit qu'une ligne de colonne ne déborde pas du milieu
        assert pymupdf.get_text_length(text, fontname=font, fontsize=size) <= max_width, text
    page.insert_text((x, y), text, fontname=font, fontsize=size, color=color)


def _make_pdf(path: Path) -> None:
    document = pymupdf.open()

    # Page 1 : couverture, exclue par le manifeste.
    cover = document.new_page()
    _put(cover, LEFT, 300, "Couverture du document", size=30, bold=True, max_width=None)

    # Page 2 : deux colonnes, page imprimée 7.
    page = document.new_page()
    _put(page, 40, 420, "7", bold=True, color=CHAPTER_BLUE)  # numéro dans la marge gauche
    # Bande 1, colonne gauche.
    _put(page, LEFT, 100, CHAPTER, size=20, bold=True, color=CHAPTER_BLUE)
    _put(page, LEFT, 140, SECTION_1, size=14, bold=True, color=SECTION_GREEN)
    _put(page, LEFT, 165, "Les équipements de travail doivent être")
    _put(page, LEFT, 179, "consignés avant toute opération.")
    # Intertitre en gras noir (aucune signature de titre) : seul le triangle le désigne.
    page.draw_polyline(
        [(LEFT, 196), (LEFT + 7, 200), (LEFT, 204)], color=None, fill=SECTION_GREEN, closePath=True
    )
    _put(page, LEFT + 12, 205, "Séparation", bold=True)
    _put(page, LEFT, 225, "Séparer les circuits des équipe-")
    _put(page, LEFT, 239, "ments situés en amont.")
    _put(page, LEFT, 270, "- premier élément de la liste ;")
    _put(page, LEFT, 290, "- second élément qui se poursuit")
    # Bande 1, colonne droite : la liste continue (ligne en retrait), puis la suite.
    _put(page, RIGHT + 10, 140, "dans la colonne de droite ;")
    _put(page, RIGHT, 160, "- troisième élément de la liste.")
    _put(page, RIGHT, 190, SUBSECTION, size=12, bold=True, color=SUBSECTION_BLUE)
    _put(page, RIGHT, 215, "Attention !", bold=True, color=WARNING_RED)
    _put(page, RIGHT, 232, "Couper l'alimentation électrique avant")
    _put(page, RIGHT, 246, "toute intervention sur la machine.")
    page.draw_rect(pymupdf.Rect(RIGHT, 262, RIGHT + 230, 300), color=None, fill=(0.95, 0.93, 0.9))
    _put(page, RIGHT + 8, 285, "Contenu important de l'encadré.")
    # Élément pleine largeur : il sépare la page en deux bandes.
    _put(page, LEFT, 340, SECTION_2, size=14, bold=True, color=SECTION_GREEN, max_width=None)
    # Bande 2.
    _put(page, LEFT, 370, "Paragraphe de la seconde bande, à gauche.")
    _put(page, RIGHT, 370, "Paragraphe de la seconde bande, à droite.")
    # Note de bas de page en 8 pt, sous un filet.
    page.draw_line((LEFT, 765), (LEFT + 200, 765), color=(0.4, 0.4, 0.4))
    _put(page, LEFT, 780, "1. Note de bas de page en huit points.", size=8)

    # Page 3 : glossaire, sans numéro imprimé.
    glossary = document.new_page()
    _put(glossary, LEFT, 100, "4 Définitions", size=20, bold=True, color=CHAPTER_BLUE)
    _put(glossary, LEFT, 140, "Consignation", bold=True, color=SECTION_GREEN)
    _put(glossary, LEFT, 160, "Procédure de mise en sécurité.")
    _put(glossary, LEFT, 190, "Déconsignation", bold=True, color=SECTION_GREEN)
    _put(glossary, LEFT, 210, "Remise en service sûre.")

    document.save(path)
    document.close()


@pytest.fixture(scope="module")
def profile() -> LayoutProfile:
    return load_layout_profile("inrs")


@pytest.fixture(scope="module")
def blocks(tmp_path_factory: pytest.TempPathFactory, profile: LayoutProfile) -> list[Block]:
    path = tmp_path_factory.mktemp("pdf") / "synthetique.pdf"
    _make_pdf(path)
    entry = ManifestEntry(
        id="synthetique",
        title="PDF synthétique",
        path=str(path),
        language="fr",
        license="test",
        page_offset=0,
        layout_profile="inrs",
        exclude_pages=[1],
    )
    return read_pdf(path, entry, profile)


def _find(blocks: list[Block], start: str) -> Block:
    matches = [block for block in blocks if block.text.startswith(start)]
    assert len(matches) == 1, f"{start!r} : {len(matches)} bloc(s)"
    return matches[0]


# --------------------------------------------------------------------------------------------
# PDF synthétique
# --------------------------------------------------------------------------------------------
def test_excluded_page_is_skipped(blocks: list[Block]) -> None:
    assert {block.pdf_page for block in blocks} == {2, 3}
    assert not any("Couverture" in block.text for block in blocks)


def test_reading_order_bands_then_columns(blocks: list[Block]) -> None:
    page = [(block.kind, block.text) for block in blocks if block.pdf_page == 2]
    assert page == [
        ("chapter", CHAPTER),
        ("section", SECTION_1),
        ("paragraph", "Les équipements de travail doivent être consignés avant toute opération."),
        ("intertitle", "Séparation"),
        ("paragraph", "Séparer les circuits des équipements situés en amont."),
        ("list_item", "- premier élément de la liste ;"),
        ("list_item", "- second élément qui se poursuit dans la colonne de droite ;"),
        ("list_item", "- troisième élément de la liste."),
        ("subsection", SUBSECTION),
        (
            "warning",
            "Attention\xa0!\nCouper l'alimentation électrique avant toute intervention sur "
            "la machine.",
        ),
        ("box", "Contenu important de l'encadré."),
        ("section", SECTION_2),
        ("paragraph", "Paragraphe de la seconde bande, à gauche."),
        ("paragraph", "Paragraphe de la seconde bande, à droite."),
        ("note", "1. Note de bas de page en huit points."),
    ]


def test_heading_levels_follow_profile_signatures(blocks: list[Block]) -> None:
    assert _find(blocks, CHAPTER).kind == "chapter"
    assert _find(blocks, SECTION_1).kind == "section"
    assert _find(blocks, SUBSECTION).kind == "subsection"
    assert _find(blocks, SECTION_2).kind == "section"
    assert all(_find(blocks, title).bold for title in (CHAPTER, SECTION_1, SUBSECTION))


def test_intertitle_detected_by_drawn_triangle(blocks: list[Block]) -> None:
    # Gras noir 11 pt : aucune signature du profil, seul le triangle dessiné en fait un titre.
    intertitle = _find(blocks, "Séparation")
    assert intertitle.kind == "intertitle"
    assert intertitle.section_path == [CHAPTER, SECTION_1, "Séparation"]


def test_section_path(blocks: list[Block]) -> None:
    assert _find(blocks, "Les équipements").section_path == [CHAPTER, SECTION_1]
    assert _find(blocks, "Séparer les circuits").section_path == [
        CHAPTER,
        SECTION_1,
        "Séparation",
    ]
    # Une sous-section ferme l'intertitre (niveau plus fin) mais garde la section.
    assert _find(blocks, "Attention").section_path == [CHAPTER, SECTION_1, SUBSECTION]
    # Une nouvelle section remplace la précédente et ferme la sous-section.
    assert _find(blocks, "Paragraphe de la seconde bande, à droite").section_path == [
        CHAPTER,
        SECTION_2,
    ]


def test_list_item_split_across_columns_is_joined(blocks: list[Block]) -> None:
    item = _find(blocks, "- second élément")
    assert item.kind == "list_item"
    assert item.text.endswith("dans la colonne de droite ;")  # espace insécable
    assert not any(block.text.startswith("dans la colonne") for block in blocks)


def test_warning_is_attached_to_following_block(blocks: list[Block]) -> None:
    warning = _find(blocks, "Attention")
    assert warning.kind == "warning"
    assert "Couper l'alimentation électrique" in warning.text
    assert not any(block.text.startswith("Couper") for block in blocks)


def test_box_and_dehyphenation(blocks: list[Block]) -> None:
    assert _find(blocks, "Contenu important").kind == "box"
    # « équipe- / ments » recollé grâce au mot « équipements » présent ailleurs dans le PDF.
    assert "équipements situés" in _find(blocks, "Séparer").text


def test_note_is_attached_to_its_page(blocks: list[Block]) -> None:
    note = _find(blocks, "1. Note de bas de page")
    assert note.kind == "note"
    assert (note.pdf_page, note.printed_page) == (2, 7)
    page = [block for block in blocks if block.pdf_page == 2]
    assert page[-1] == note  # après tous les autres blocs de sa page


def test_printed_page_number(blocks: list[Block]) -> None:
    assert {block.printed_page for block in blocks if block.pdf_page == 2} == {7}
    assert {block.printed_page for block in blocks if block.pdf_page == 3} == {None}


def test_glossary_terms_followed_by_definitions(blocks: list[Block]) -> None:
    page = [(block.kind, block.text) for block in blocks if block.pdf_page == 3]
    assert page == [
        ("chapter", "4 Définitions"),
        ("glossary_term", "Consignation"),
        ("paragraph", "Procédure de mise en sécurité."),
        ("glossary_term", "Déconsignation"),
        ("paragraph", "Remise en service sûre."),
    ]
    assert _find(blocks, "Remise en service").section_path == ["4 Définitions", "Déconsignation"]


# --------------------------------------------------------------------------------------------
# Vrai PDF INRS (local uniquement, non versionné)
# --------------------------------------------------------------------------------------------
INRS_ID = "inrs-ed6109"


def _inrs_entry() -> ManifestEntry:
    return next(entry for entry in load_manifest() if entry.id == INRS_ID)


@pytest.mark.skipif(
    not (ROOT / _inrs_entry().path).exists(), reason="PDF INRS absent (non versionné)"
)
def test_real_inrs_page_16() -> None:
    entry = _inrs_entry()
    assert entry.layout_profile is not None
    blocks = read_pdf(ROOT / entry.path, entry, load_layout_profile(entry.layout_profile))
    page = [block for block in blocks if block.pdf_page == 16]
    assert page and {block.printed_page for block in page} == {14}

    positions = []
    for number in range(1, 6):
        found = [
            i
            for i, block in enumerate(page)
            if f"opération\xa0{number})" in block.text or f"opération {number})" in block.text
        ]
        assert len(found) == 1, f"opération {number} : {found}"
        assert page[found[0]].kind == "list_item"
        positions.append(found[0])
    assert positions == sorted(positions)

    attention = next(i for i, block in enumerate(page) if block.text.startswith("Attention"))
    assert positions[-1] < attention
