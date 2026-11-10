"""Lecture structurée d'un PDF du corpus : blocs typés, dans l'ordre de lecture.

Reprend les règles validées par l'exploration (scripts/inspect_pdf.py) :

- ordre de lecture : la page est découpée en bandes horizontales par les éléments pleine
  largeur ; dans chaque bande, colonne gauche puis colonne droite. Le milieu est celui de la
  ZONE DE TEXTE (marges intérieures et extérieures différentes entre pages paires et impaires) ;
- niveaux de titre reconnus par leur signature (taille, couleur, gras) du profil de mise en
  page ; intertitres repérés aussi par le petit triangle dessiné devant eux ;
- dans le chapitre « Définitions », les intertitres sont des termes du glossaire, chacun suivi
  de sa définition ;
- « Attention ! » est fusionné avec le bloc qui le suit (une liste entière si c'est une liste) ;
- texte d'un encadré en ``box`` ; légendes de photos en ``caption`` ; texte des schémas
  (tracés vectoriels) ignoré ;
- notes de bas de page en ``note``, placées à la fin des blocs de LEUR page ;
- césures recollées : trait d'union conditionnel supprimé, puis ``dehyphenate`` avec un
  vocabulaire construit sur le document lui-même.

Les blocs ne franchissent jamais une page : un paragraphe coupé par un saut de page donne
deux blocs, chacun avec sa page exacte. Un saut de colonne dans la même page est recollé.

Ce module ne fait qu'extraire : il n'écrit aucun fichier.
"""

import itertools
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pymupdf
from pydantic import BaseModel, ConfigDict, Field

from maintenance_rag.config import LayoutProfile, ManifestEntry, TextSignature
from maintenance_rag.normalize import build_vocabulary, dehyphenate

BlockKind = Literal[
    "chapter",
    "section",
    "subsection",
    "intertitle",
    "glossary_term",
    "paragraph",
    "list_item",
    "note",
    "box",
    "caption",
    "warning",
]

HEADING_LEVELS: dict[str, int] = {
    "chapter": 0,
    "section": 1,
    "subsection": 2,
    "intertitle": 3,
    "glossary_term": 3,
}

_BULLET = re.compile(r"^(?:[•–▪■►➤◦‣·-]\s|[•▪■►➤◦‣·])")
_NUMBERED = re.compile(r"^(?:\d{1,2}\.|[a-z]\))\s")
_NOTE_START = re.compile(r"^\d{1,2}\.\s")
_CHAPTER_NUMBER = re.compile(r"^\d+\s+")
_SENTENCE_END = re.compile(r"[.!?]$")
_FLOW_END = re.compile(r"[.:;!?»)]$")
_PAGE_NUMBER = re.compile(r"\d{1,3}")


class Block(BaseModel):
    """Bloc de texte typé, dans l'ordre de lecture du document."""

    model_config = ConfigDict(extra="forbid")

    pdf_page: int = Field(ge=1, description="page PDF (1 = première page du fichier)")
    printed_page: int | None = Field(description="numéro imprimé sur la page, None si absent")
    kind: BlockKind
    text: str
    bold: bool = Field(description="majorité des lignes du bloc en gras")
    section_path: list[str] = Field(
        default_factory=list,
        description="titres en vigueur, du chapitre au plus fin (le titre lui-même inclus)",
    )


# ---------------------------------------------------------------------------------------------
# Lignes et dessins d'une page
# ---------------------------------------------------------------------------------------------
@dataclass
class _Line:
    page: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    color: str
    bold: bool
    margin: bool = False
    column: str = "L"  # L (gauche ou colonne unique), R (droite), F (pleine largeur)
    band: int = 0
    role: str = "body"  # rôle typographique : chapter, section, …, note, caption, box, body
    frame: int | None = None  # indice de l'encadré qui contient la ligne
    arrow: bool = False
    paragraph_gap: float = 0.0
    column_left: float = 0.0
    column_right: float = 0.0
    short_margin: float = 0.0
    list_indent: float = 0.0

    @property
    def style(self) -> tuple[float, str, bool]:
        return (self.size, self.color, self.bold)

    @property
    def starts_item(self) -> bool:
        return bool(_BULLET.match(self.text) or _NUMBERED.match(self.text))


@dataclass
class _Draft:
    kind: str
    page: int
    lines: list[_Line] = field(default_factory=list)
    text: str = ""

    @property
    def first(self) -> _Line:
        return self.lines[0]

    @property
    def last(self) -> _Line:
        return self.lines[-1]


@dataclass
class _Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, x: float, y: float, pad: float = 2.0) -> bool:
        return self.x0 - pad <= x <= self.x1 + pad and self.y0 - pad <= y <= self.y1 + pad

    def intersects(self, other: "_Rect") -> bool:
        return not (
            other.x1 < self.x0 or other.x0 > self.x1 or other.y1 < self.y0 or other.y0 > self.y1
        )

    def distance(self, other: "_Rect") -> float:
        dx = max(other.x0 - self.x1, self.x0 - other.x1, 0.0)
        dy = max(other.y0 - self.y1, self.y0 - other.y1, 0.0)
        return max(dx, dy)

    def union(self, other: "_Rect") -> "_Rect":
        return _Rect(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )


@dataclass
class _Drawings:
    frames: list[_Rect]
    arrows: list[_Rect]
    diagrams: list[_Rect]
    images: list[_Rect]


def _clean(text: str) -> str:
    """Normalise les caractères de mise en page ; les césures sont traitées à la jointure."""
    text = text.replace("\x07", " ").replace("\t", " ")
    text = re.sub("[-]", "", text)  # pictogrammes en zone privée (flèches, etc.)
    text = re.sub("[  \xa0]", " ", text)
    text = re.sub(r" {2,}", " ", text).strip()
    # Espace insécable devant la ponctuation haute : évite un « ; » isolé au retour à la ligne.
    text = re.sub(r" ([;:!?»])", lambda match: "\xa0" + match.group(1), text)
    return text.replace("« ", "«\xa0")


def _hex(color: int) -> str:
    return f"{color:06x}"


def _extract_lines(page: pymupdf.Page, number: int) -> list[_Line]:
    lines: list[_Line] = []
    content: dict[str, Any] = page.get_text("dict")  # type: ignore[no-untyped-call]
    for block in content["blocks"]:
        if block["type"] != 0:
            continue
        for raw in block["lines"]:
            spans = [span for span in raw["spans"] if span["text"].strip()]
            if not spans:
                continue
            parts = []
            for span in raw["spans"]:
                call = span["text"].strip()
                superscript = span["flags"] & 1 and re.fullmatch(r"\w{1,3}", call)
                parts.append(f"[^{call}]" if superscript else span["text"])  # appel de note
            regular = [span for span in spans if not span["flags"] & 1] or spans
            main = max(regular, key=lambda span: len(span["text"].strip()))
            text = _clean("".join(parts))
            if not text:
                continue
            x0, y0, x1, y1 = raw["bbox"]
            lines.append(
                _Line(
                    page=number,
                    text=text,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    size=round(main["size"], 1),
                    color=_hex(main["color"]),
                    bold=bool(main["flags"] & 16) or "bold" in main["font"].lower(),
                )
            )
    return lines


def _rect(raw: Any) -> _Rect:
    return _Rect(float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))


def _is_white(fill: Sequence[float] | None) -> bool:
    return fill is not None and all(channel > 0.98 for channel in fill)


def _find_drawings(page: pymupdf.Page, profile: LayoutProfile) -> _Drawings:
    """Encadrés, triangles d'intertitre, schémas et photos de la page."""
    width, height = page.rect.width, page.rect.height
    frames: list[_Rect] = []
    arrows: list[_Rect] = []
    diagram_parts: list[_Rect] = []
    for drawing in page.get_drawings():
        rect = _rect(drawing["rect"])
        inside = rect.x0 >= 0 and rect.y0 >= 0 and rect.x1 <= width and rect.y1 <= height
        if not inside:  # cercles décoratifs débordant de la page
            continue
        kinds = {item[0] for item in drawing["items"]}
        fill = drawing.get("fill")
        stroke = drawing.get("color")
        w, h = rect.x1 - rect.x0, rect.y1 - rect.y0
        if kinds == {"re"} and fill is not None and stroke is None and not _is_white(fill):
            big = w >= profile.frame_min_width and h >= profile.frame_min_height
            if big and w * h < 0.9 * width * height:  # le fond de page n'est pas un encadré
                frames.append(rect)
        elif (
            kinds == {"l"}
            and fill is not None
            and stroke is None
            and max(w, h) <= profile.arrow_max_size
        ):
            arrows.append(rect)
        if kinds & {"c", "qu"} or ("re" in kinds and stroke is not None):
            diagram_parts.append(rect)
    images = [_rect(info["bbox"]) for info in page.get_image_info()]
    return _Drawings(frames, arrows, _cluster(diagram_parts, profile), images)


def _cluster(parts: list[_Rect], profile: LayoutProfile) -> list[_Rect]:
    """Regroupe les tracés proches ; un groupe assez fourni est un schéma."""
    groups: list[tuple[_Rect, int]] = []
    for part in parts:
        box, count = part, 1
        merged = True
        while merged:
            merged = False
            for index, (other, n) in enumerate(groups):
                if box.distance(other) <= profile.diagram_merge_distance:
                    box, count = box.union(other), count + n
                    del groups[index]
                    merged = True
                    break
        groups.append((box, count))
    return [box for box, count in groups if count >= profile.diagram_min_items]


# ---------------------------------------------------------------------------------------------
# Classement des lignes
# ---------------------------------------------------------------------------------------------
def _matches(line: _Line, signatures: Iterable[TextSignature], profile: LayoutProfile) -> bool:
    for signature in signatures:
        if abs(line.size - signature.size) > profile.size_tolerance:
            continue
        if line.bold != signature.bold:
            continue
        channels = zip(bytes.fromhex(line.color), bytes.fromhex(signature.color), strict=True)
        if all(abs(a - b) <= profile.color_tolerance for a, b in channels):
            return True
    return False


def _assign_columns(body: list[_Line], profile: LayoutProfile) -> dict[str, tuple[float, float]]:
    """Classe chaque ligne en colonne gauche, droite ou pleine largeur ; renvoie leurs bords."""
    if not body:
        return {}
    left_edge = min(line.x0 for line in body)
    right_edge = max(line.x1 for line in body)
    middle = (left_edge + right_edge) / 2
    tolerance = profile.column_tolerance
    for line in body:
        if line.x1 <= middle + tolerance:
            line.column = "L"
        elif line.x0 >= middle - tolerance:
            line.column = "R"
        else:
            line.column = "F"
    counts = Counter(line.column for line in body)
    if min(counts["L"], counts["R"]) < profile.min_lines_per_column:
        for line in body:
            line.column = "L"
        return {"L": (left_edge, right_edge), "F": (left_edge, right_edge)}
    # Une ligne courte posée juste au-dessus d'un élément pleine largeur, sans texte de l'autre
    # colonne à sa hauteur, en fait partie (« Attention ! » au-dessus d'une liste pleine largeur).
    for line in sorted(body, key=lambda line: -line.y0):
        if line.column == "F":
            continue
        below = [other for other in body if other.y0 > line.y0]
        following = min(below, key=lambda other: other.y0, default=None)
        beside = any(
            other.column not in (line.column, "F") and other.y0 < line.y1 and other.y1 > line.y0
            for other in body
        )
        if following is not None and following.column == "F" and not beside:
            if following.y0 - line.y1 < line.size and following.x0 >= line.x0 - tolerance:
                line.column = "F"
    ranges = {"F": (left_edge, right_edge)}
    for column in ("L", "R"):
        members = [line for line in body if line.column == column]
        ranges[column] = (min(line.x0 for line in members), max(line.x1 for line in members))
    return ranges


def _reading_order(body: list[_Line]) -> list[_Line]:
    """Bandes séparées par les lignes pleine largeur ; dans chaque bande, gauche puis droite."""
    full = sorted((line for line in body if line.column == "F"), key=lambda line: line.y0)
    keyed: list[tuple[tuple[int, int, float, float], _Line]] = []
    for line in body:
        if line.column == "F":
            line.band = 2 * full.index(line) + 1
            key = (line.band, 0, line.y0, line.x0)
        else:
            line.band = 2 * sum(1 for other in full if other.y0 <= line.y0)
            key = (line.band, 0 if line.column == "L" else 1, line.y0, line.x0)
        keyed.append((key, line))
    return [line for _, line in sorted(keyed, key=lambda pair: pair[0])]


def _classify(line: _Line, drawings: _Drawings, profile: LayoutProfile) -> str:
    """Rôle typographique d'une ligne de corps (hors marge et hors note)."""
    cx, cy = (line.x0 + line.x1) / 2, (line.y0 + line.y1) / 2
    if any(diagram.contains(cx, cy) for diagram in drawings.diagrams):
        return "diagram"
    box = _Rect(line.x0, line.y0, line.x1, line.y1)
    if line.frame is not None:
        frame = drawings.frames[line.frame]
        if any(frame.intersects(image) for image in drawings.images):
            return "caption"  # encadré de commentaire collé à une photo
        return "warning" if _is_warning(line, profile) else "box"
    if _matches(line, profile.chapter, profile):
        return "chapter"
    if _matches(line, profile.section, profile):
        return "section"
    if _matches(line, profile.subsection, profile):
        return "subsection"
    if _is_warning(line, profile):
        return "warning"
    if line.arrow:
        return "intertitle"
    if _matches(line, profile.intertitle, profile) and not line.starts_item:
        return "intertitle"
    if _matches(line, profile.caption, profile):
        return "caption"
    for image in drawings.images:
        overlap = min(box.x1, image.x1) - max(box.x0, image.x0)
        gap = max(image.y0 - box.y1, box.y0 - image.y1)
        if overlap > 0 and gap <= profile.caption_max_gap:
            return "caption"
    return "body"


def _is_warning(line: _Line, profile: LayoutProfile) -> bool:
    return _matches(line, profile.warning, profile) and line.text.startswith(profile.warning_prefix)


@dataclass
class _Page:
    number: int
    printed: int | None
    flow: list[_Line]  # lignes de corps dans l'ordre de lecture (schémas exclus)
    notes: list[_Line]


def _analyse_page(page: pymupdf.Page, number: int, profile: LayoutProfile) -> _Page:
    width, height = page.rect.width, page.rect.height
    lines = _extract_lines(page, number)
    drawings = _find_drawings(page, profile)

    printed: int | None = None
    for line in lines:
        centre = (line.x0 + line.x1) / 2
        line.margin = (
            centre < profile.margin_ratio * width or centre > (1 - profile.margin_ratio) * width
        )
        if (
            line.margin
            and _PAGE_NUMBER.fullmatch(line.text)
            and _matches(line, profile.page_number, profile)
        ):
            printed = int(line.text)

    notes: list[_Line] = []
    body: list[_Line] = []
    for line in lines:
        if line.margin:
            continue
        if line.size <= profile.note_max_size and line.y0 >= profile.note_min_y_ratio * height:
            line.role = "note"
            notes.append(line)
            continue
        cx, cy = (line.x0 + line.x1) / 2, (line.y0 + line.y1) / 2
        line.frame = next(
            (i for i, frame in enumerate(drawings.frames) if frame.contains(cx, cy)), None
        )
        line.arrow = any(
            arrow.y0 - 2 <= cy <= arrow.y1 + 2
            and -2 <= line.x0 - arrow.x1 <= profile.arrow_max_distance
            for arrow in drawings.arrows
        )
        line.role = _classify(line, drawings, profile)
        if line.role != "diagram":
            body.append(line)

    ranges = _assign_columns(body, profile)
    flow = _reading_order(body)
    gaps = sorted(
        b.y0 - a.y1
        for a, b in itertools.pairwise(flow)
        if (a.band, a.column) == (b.band, b.column) and -a.size < b.y0 - a.y1 < a.size
    )
    line_gap = gaps[len(gaps) // 2] if gaps else 0.0
    for line in flow:
        line.paragraph_gap = line_gap + profile.paragraph_gap_margin
        line.column_left, line.column_right = ranges.get(line.column, (line.x0, line.x1))
        line.short_margin = profile.short_line_margin
        line.list_indent = profile.list_indent
    notes.sort(key=lambda line: (line.y0, line.x0))
    return _Page(number=number, printed=printed, flow=flow, notes=notes)


# ---------------------------------------------------------------------------------------------
# Assemblage des lignes en blocs
# ---------------------------------------------------------------------------------------------
def _join(previous: str, following: str) -> str:
    following = following.lstrip("\xad")
    if previous.endswith("\xad"):  # césure typographique (trait d'union conditionnel)
        return previous[:-1] + following
    if previous.endswith("-") and not previous.endswith(" -"):  # césure saisie comme un tiret
        return previous + following  # tranché ensuite par dehyphenate et le vocabulaire
    return f"{previous} {following}"


def _line_kind(line: _Line) -> str:
    if line.role == "body":
        return "list_item" if line.starts_item else "paragraph"
    return line.role  # titres, warning, box, caption


def _is_heading(kind: str) -> bool:
    return kind in HEADING_LEVELS or kind == "warning"


def _continues(draft: _Draft, line: _Line) -> bool:
    """La ligne prolonge-t-elle le bloc en cours (même page) ?"""
    last = draft.last
    kind = _line_kind(line)
    if last.frame != line.frame:
        return False
    gap = line.y0 - last.y1
    # Même flux : même bande et même colonne, ou ligne posée juste sous la précédente (la
    # suite d'une puce, plus large que l'item, peut passer en pleine largeur).
    overlap = min(line.x1, last.x1) - max(line.x0, last.x0)
    stacked = -line.size / 2 <= gap <= last.paragraph_gap and overlap > 0
    same_flow = (last.band, last.column) == (line.band, line.column) or stacked
    if kind == "intertitle" and not line.arrow and draft.kind in ("paragraph", "list_item"):
        # Ligne à la signature d'intertitre sans triangle qui poursuit un item écrit dans le
        # même style (item numéroté en gras coloré coupé en fin de ligne, p. 9 de l'ED 6109).
        flowing = same_flow and line.y0 - last.y1 <= last.paragraph_gap
        return flowing and last.style == line.style and not _FLOW_END.search(last.text)
    if _is_heading(kind):
        # Titre sur plusieurs lignes : même rôle, même style, lignes proches (parfois
        # chevauchantes) ; chaque ligne pleine largeur ouvrant une bande, on ignore la bande.
        same = draft.kind == kind and last.style == line.style
        return same and -line.size / 2 <= line.y0 - last.y1 < line.size
    if _is_heading(draft.kind) or line.starts_item:
        return False
    if (draft.kind == "caption") != (kind == "caption"):
        return False
    item = draft.first.starts_item
    if same_flow:
        if line.y0 - last.y1 > last.paragraph_gap:
            return False
        if item and line.x0 > draft.first.x0 + line.list_indent:
            return True  # ligne en retrait : suite de la puce
        # Texte justifié : une ligne courte terminée par un point clôt le paragraphe, même
        # sans espace supplémentaire avant le suivant.
        short = last.x1 < last.column_right - last.short_margin
        if short and _SENTENCE_END.search(last.text):
            return False
        # Suite d'une puce sans retrait : seulement si la phrase de l'item n'est pas finie.
        return not (item and _FLOW_END.search(last.text))
    # Saut de colonne dans la page : on continue si la phrase n'est pas terminée. Une légende
    # ne se poursuit jamais dans une autre colonne (elle est collée à sa photo).
    if draft.kind == "caption" or _FLOW_END.search(last.text):
        return False
    if item:  # la suite d'un item est en retrait par rapport au bord de sa colonne
        return line.x0 > line.column_left + line.list_indent
    return True


def _assemble(lines: list[_Line], page: int) -> list[_Draft]:
    drafts: list[_Draft] = []
    for line in lines:
        if drafts and _continues(drafts[-1], line):
            drafts[-1].lines.append(line)
            drafts[-1].text = _join(drafts[-1].text, line.text)
        else:
            drafts.append(_Draft(kind=_line_kind(line), page=page, lines=[line], text=line.text))
    return drafts


def _assemble_notes(lines: list[_Line], page: int) -> list[_Draft]:
    drafts: list[_Draft] = []
    for line in lines:
        if drafts and not _NOTE_START.match(line.text):
            drafts[-1].lines.append(line)
            drafts[-1].text = _join(drafts[-1].text, line.text)
        else:
            drafts.append(_Draft(kind="note", page=page, lines=[line], text=line.text))
    return drafts


def _attach_warnings(drafts: list[_Draft]) -> list[_Draft]:
    """Fusionne chaque « Attention ! » avec le bloc qui suit (toute la liste si c'en est une)."""
    result: list[_Draft] = []
    index = 0
    while index < len(drafts):
        draft = drafts[index]
        index += 1
        if draft.kind == "warning":
            attached: list[_Draft] = []
            if index < len(drafts) and drafts[index].kind in ("paragraph", "list_item", "box"):
                attached.append(drafts[index])
                index += 1
                if attached[0].kind == "list_item":
                    while index < len(drafts) and drafts[index].kind == "list_item":
                        attached.append(drafts[index])
                        index += 1
            for other in attached:
                draft.text = f"{draft.text}\n{other.text}"
                draft.lines.extend(other.lines)
        result.append(draft)
    return result


def _bold(draft: _Draft) -> bool:
    return 2 * sum(line.bold for line in draft.lines) > len(draft.lines)


def _final_text(text: str, vocabulary: set[str]) -> str:
    return dehyphenate(text.replace("\xad", ""), vocabulary).strip()


# ---------------------------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------------------------
def read_pdf(path: Path | str, entry: ManifestEntry, profile: LayoutProfile) -> list[Block]:
    """Lit un PDF et renvoie ses blocs typés dans l'ordre de lecture.

    Les pages ``entry.exclude_pages`` sont ignorées. ``printed_page`` est le numéro lu dans la
    marge de la page (None s'il n'y en a pas). Les notes d'une page suivent les autres blocs de
    cette page ; leur ``section_path`` est celui du bloc qui porte l'appel de note (« [^n] »),
    à défaut celui de la fin de la page.
    """
    excluded = set(entry.exclude_pages)
    with pymupdf.open(Path(path)) as document:  # type: ignore[no-untyped-call]
        pages = [
            _analyse_page(document[number - 1], number, profile)
            for number in range(1, document.page_count + 1)
            if number not in excluded
        ]

    vocabulary = build_vocabulary(
        line.text for page in pages for line in itertools.chain(page.flow, page.notes)
    )
    blocks: list[Block] = []
    stack: list[tuple[int, str]] = []  # (niveau, titre) des titres en vigueur
    in_glossary = False
    for page in pages:
        page_blocks: list[Block] = []
        for draft in _attach_warnings(_assemble(page.flow, page.number)):
            text = _final_text(draft.text, vocabulary)
            kind = draft.kind
            if kind == "intertitle" and in_glossary:
                kind = "glossary_term"
            if kind in HEADING_LEVELS:
                level = HEADING_LEVELS[kind]
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, text))
                if kind == "chapter":
                    title = _CHAPTER_NUMBER.sub("", text).strip()
                    in_glossary = title.casefold() == profile.glossary_chapter.casefold()
            page_blocks.append(_block(page, kind, text, draft, [title for _, title in stack]))
        end_path = [title for _, title in stack]
        for draft in _assemble_notes(page.notes, page.number):
            text = _final_text(draft.text, vocabulary)
            call = f"[^{text.split('.', 1)[0]}]"
            caller = next((block for block in page_blocks if call in block.text), None)
            note_path = list(caller.section_path) if caller else end_path
            page_blocks.append(_block(page, "note", text, draft, note_path))
        blocks.extend(page_blocks)
    return blocks


def _block(page: _Page, kind: str, text: str, draft: _Draft, path: list[str]) -> Block:
    return Block.model_validate(
        {
            "pdf_page": page.number,
            "printed_page": page.printed,
            "kind": kind,
            "text": text,
            "bold": _bold(draft),
            "section_path": path,
        }
    )
