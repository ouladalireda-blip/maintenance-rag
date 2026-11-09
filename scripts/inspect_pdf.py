"""Exploration de la mise en page d'un PDF du corpus (outil de développement, hors du package).

Affiche dans le terminal, pour chaque page demandée : numéro de page imprimé, colonnes, niveaux de
titre, notes, encadrés et puces, puis le texte reconstruit dans l'ordre de lecture.

N'écrit AUCUN fichier : le texte des documents peut être protégé par le droit d'auteur.

Usage :
    uv run python scripts/inspect_pdf.py corpus/inrs/ed6109.pdf --pages 14 15 16 17
"""

import argparse
import itertools
import re
import sys
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf
import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.yaml"

# ---------------------------------------------------------------------------------------------
# Seuils (en points PDF) : valeurs d'exploration, à valider avant de les reporter dans le code.
# ---------------------------------------------------------------------------------------------
MARGIN_RATIO = 0.12  # bande latérale (fraction de la largeur) où se trouvent les numéros de page
COLUMN_TOLERANCE = 10.0  # marge autour du milieu de la zone de texte pour classer une ligne
MIN_LINES_PER_COLUMN = 3  # en dessous, la page est considérée à une seule colonne
CHAPTER_MIN_SIZE = 18.0  # chapitre : très grande police
SECTION_MIN_SIZE = 13.0  # section : grande police grasse
SUBSECTION_MIN_SIZE = 10.5  # sous-section : police grasse colorée (ou précédée d'un triangle ➤)
NOTE_MAX_SIZE = 9.0  # note de bas de page : petite police...
NOTE_MIN_Y_RATIO = 0.75  # ... dans le dernier quart de la page
# Nouveau paragraphe si l'écart vertical dépasse l'interligne médian de la page + cette marge (pt).
# L'interligne varie d'une page à l'autre (≈ 1,5 pt p. 14, ≈ 3,4 pt p. 17) : pas de seuil absolu.
PARAGRAPH_GAP_MARGIN = 1.5
SHORT_LINE_MARGIN = 15.0  # ligne « courte » : s'arrête à plus de 15 pt du bord droit de colonne
FRAME_MIN_WIDTH, FRAME_MIN_HEIGHT = 40.0, 15.0  # taille minimale d'un rectangle d'encadré
ARROW_MAX_SIZE = 12.0  # taille maximale du petit triangle dessiné devant les intertitres
ARROW_MAX_DISTANCE = 12.0  # distance maximale entre le triangle et le début de la ligne
LIST_INDENT = 2.0  # retrait minimal (pt) d'une ligne de continuation d'une puce
BULLET_CHARS = "•–▪■►➤◦‣·-"
BLACK = "000000"
HEADINGS = ("CHAPITRE", "SECTION", "SOUS-SECTION")
NUMBERED_SUBSECTION = re.compile(r"^\d+(\.\d+){2,}\.?\s")


@dataclass
class Line:
    page: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    color: str
    bold: bool
    font: str
    margin: bool = False
    column: str = "L"  # L (gauche ou colonne unique), R (droite), F (pleine largeur)
    band: int = 0  # bande horizontale délimitée par les lignes pleine largeur
    tag: str = ""  # CHAPITRE, SECTION, SOUS-SECTION, NOTE, PUCE ou "" (corps)
    framed: bool = False
    arrow: bool = False
    paragraph_gap: float = 0.0  # seuil de changement de paragraphe propre à la page
    column_right: float = 0.0  # bord droit de la colonne (pour repérer les lignes courtes)

    @property
    def style(self) -> tuple[float, str, bool]:
        return (self.size, self.color, self.bold)


@dataclass
class Block:
    tag: str
    framed: bool
    lines: list[Line] = field(default_factory=list)
    text: str = ""

    @property
    def first(self) -> Line:
        return self.lines[0]

    @property
    def last(self) -> Line:
        return self.lines[-1]


@dataclass
class PageReport:
    number: int
    printed: str | None
    columns: int
    column_ranges: dict[str, tuple[float, float]]
    middle: float
    frames: int
    arrows: int
    margin_lines: int
    line_gap: float
    lines: list[Line]


# ---------------------------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------------------------
def clean(text: str) -> str:
    """Normalise les caractères de mise en page ; les césures sont traitées à la jointure."""
    text = text.replace("\x07", "").replace("\t", " ")
    text = re.sub("[\u2009\u202f\xa0]", " ", text)
    text = re.sub(r" {2,}", " ", text).strip()
    # Espace insécable devant la ponctuation haute : évite un « ; » isolé au retour à la ligne.
    text = re.sub(r" ([;:!?»])", lambda match: "\xa0" + match.group(1), text)
    return text.replace("« ", "«\xa0")


def extract_lines(page: Any, number: int) -> list[Line]:
    lines: list[Line] = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for raw in block["lines"]:
            spans = [s for s in raw["spans"] if s["text"].strip()]
            if not spans:
                continue
            parts = []
            for span in raw["spans"]:
                superscript = span["flags"] & 1 and span["text"].strip()
                parts.append(f"[^{span['text'].strip()}]" if superscript else span["text"])
            regular = [s for s in spans if not s["flags"] & 1] or spans
            main = max(regular, key=lambda s: len(s["text"].strip()))
            x0, y0, x1, y1 = raw["bbox"]
            lines.append(
                Line(
                    page=number,
                    text=clean("".join(parts)),
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    size=round(main["size"], 1),
                    color=f"{main['color']:06x}",
                    bold=bool(main["flags"] & 16) or "bold" in main["font"].lower(),
                    font=main["font"],
                )
            )
    return lines


def find_drawings(page: Any) -> tuple[list[Any], list[Any]]:
    """Rectangles d'encadré et petits triangles (➤ dessinés) de la page."""
    frames, arrows = [], []
    area = page.rect.width * page.rect.height
    for drawing in page.get_drawings():
        rect = drawing["rect"]
        kinds = {item[0] for item in drawing["items"]}
        inside = rect.x0 >= 0 and rect.y0 >= 0 and rect.x1 <= page.rect.x1
        inside = inside and rect.y1 <= page.rect.y1
        if not inside or "c" in kinds:  # les cercles décoratifs (souvent hors page) sont ignorés
            continue
        big = rect.width >= FRAME_MIN_WIDTH and rect.height >= FRAME_MIN_HEIGHT
        if big and rect.width * rect.height < 0.9 * area:
            frames.append(rect)
        elif drawing.get("fill") is not None and max(rect.width, rect.height) <= ARROW_MAX_SIZE:
            arrows.append(rect)
    return frames, arrows


def assign_columns(body: list[Line]) -> tuple[int, float, dict[str, tuple[float, float]]]:
    """Classe chaque ligne en colonne gauche, droite ou pleine largeur.

    Le milieu est celui de la ZONE DE TEXTE, pas de la page : les marges intérieures et
    extérieures diffèrent entre pages paires et impaires.
    """
    if not body:  # page vide ou contenu uniquement en marge
        return 0, 0.0, {}
    left_edge = min(line.x0 for line in body)
    right_edge = max(line.x1 for line in body)
    middle = (left_edge + right_edge) / 2
    for line in body:
        if line.x1 <= middle + COLUMN_TOLERANCE:
            line.column = "L"
        elif line.x0 >= middle - COLUMN_TOLERANCE:
            line.column = "R"
        else:
            line.column = "F"
    counts = Counter(line.column for line in body)
    if counts["L"] < MIN_LINES_PER_COLUMN or counts["R"] < MIN_LINES_PER_COLUMN:
        for line in body:
            line.column = "L"
        return 1, middle, {"L": (left_edge, right_edge)}
    ranges = {}
    for column in ("L", "R"):
        members = [line for line in body if line.column == column]
        ranges[column] = (min(line.x0 for line in members), max(line.x1 for line in members))
    return 2, middle, ranges


def reading_order(body: list[Line]) -> list[Line]:
    """Bandes séparées par les lignes pleine largeur ; dans chaque bande : gauche puis droite."""
    full = sorted((line for line in body if line.column == "F"), key=lambda line: line.y0)
    keyed: list[tuple[tuple[int, int, float, float], Line]] = []
    for line in body:
        if line.column == "F":
            line.band = full.index(line)
            key = (2 * line.band + 1, 0, line.y0, line.x0)
        else:
            line.band = sum(1 for f in full if f.y0 <= line.y0)
            key = (2 * line.band, 0 if line.column == "L" else 1, line.y0, line.x0)
        keyed.append((key, line))
    return [line for _, line in sorted(keyed, key=lambda pair: pair[0])]


def classify(line: Line, page_height: float) -> str:
    if line.margin:
        return "MARGE"
    if not line.framed:  # le texte d'un encadré (ex. pancarte) n'est jamais un titre
        if line.size >= CHAPTER_MIN_SIZE:
            return "CHAPITRE"
        if line.bold and line.size >= SECTION_MIN_SIZE:
            return "SECTION"
        if line.arrow or (line.bold and line.color != BLACK and line.size >= SUBSECTION_MIN_SIZE):
            return "SOUS-SECTION"
    if line.size <= NOTE_MAX_SIZE and line.y0 >= NOTE_MIN_Y_RATIO * page_height:
        return "NOTE"
    if line.text[:1] in BULLET_CHARS:
        return "PUCE"
    return ""


def analyse_page(document: Any, number: int, page_offset: int | None) -> PageReport:
    page = document[number - 1]
    width, height = page.rect.width, page.rect.height
    lines = extract_lines(page, number)

    printed = None
    for line in lines:
        centre = (line.x0 + line.x1) / 2
        line.margin = centre < MARGIN_RATIO * width or centre > (1 - MARGIN_RATIO) * width
        if line.margin and re.fullmatch(r"\d{1,3}", line.text):
            printed = line.text
    body = [line for line in lines if not line.margin]

    columns, middle, ranges = assign_columns(body)
    frames, arrows = find_drawings(page)
    for line in body:
        cx, cy = (line.x0 + line.x1) / 2, (line.y0 + line.y1) / 2
        line.framed = any(f.x0 - 2 <= cx <= f.x1 + 2 and f.y0 - 2 <= cy <= f.y1 + 2 for f in frames)
        line.arrow = any(
            a.y0 - 2 <= cy <= a.y1 + 2 and -2 <= line.x0 - a.x1 <= ARROW_MAX_DISTANCE
            for a in arrows
        )
    for line in lines:
        line.tag = classify(line, height)

    ordered = reading_order(body)
    gaps = sorted(
        b.y0 - a.y1
        for a, b in itertools.pairwise(ordered)
        if (a.band, a.column) == (b.band, b.column) and 0 <= b.y0 - a.y1 < a.size
    )
    line_gap = gaps[len(gaps) // 2] if gaps else 0.0
    for line in body:
        line.paragraph_gap = line_gap + PARAGRAPH_GAP_MARGIN
        line.column_right = ranges.get(line.column, (0.0, page.rect.width))[1]

    return PageReport(
        number=number,
        printed=printed,
        columns=columns,
        column_ranges=ranges,
        middle=middle,
        frames=len(frames),
        arrows=len(arrows),
        margin_lines=len(lines) - len(body),
        line_gap=line_gap,
        lines=ordered + [line for line in lines if line.margin],
    )


# ---------------------------------------------------------------------------------------------
# Reconstruction du texte
# ---------------------------------------------------------------------------------------------
def join(previous: str, following: str) -> str:
    following = following.lstrip("\xad")
    if previous.endswith("\xad"):  # césure typographique (trait d'union conditionnel)
        return previous[:-1] + following
    if previous.endswith("-") and not previous.endswith(" -"):  # césure saisie comme un tiret
        return previous + following
    return f"{previous} {following}"


def same_flow(a: Line, b: Line) -> bool:
    return (a.page, a.band, a.column) == (b.page, b.band, b.column)


def build_blocks(
    reports: list[PageReport],
) -> tuple[list[Block], list[str]]:
    blocks: list[Block] = []
    events: list[str] = []
    column_x0 = {
        (r.number, column): span[0] for r in reports for column, span in r.column_ranges.items()
    }

    def where(line: Line) -> str:
        names = {"L": "gauche", "R": "droite", "F": "pleine largeur"}
        return f"p. {line.page}, colonne {names[line.column]}"

    for report in reports:
        for line in (line for line in report.lines if not line.margin):
            previous = blocks[-1] if blocks else None
            if previous is None or not attach(previous, line, column_x0, events, where):
                block = Block(tag=line.tag, framed=line.framed, lines=[line], text=line.text)
                if previous is not None and line.tag == "PUCE" and previous.tag == "PUCE":
                    if not same_flow(previous.last, line):
                        events.append(
                            f"Liste poursuivie (nouvel item) : {where(previous.last)} → "
                            f"{where(line)}"
                        )
                blocks.append(block)
    return blocks, events


def attach(
    previous: Block,
    line: Line,
    column_x0: dict[tuple[int, str], float],
    events: list[str],
    where: Any,
) -> bool:
    """Rattache la ligne au bloc précédent si elle le prolonge ; signale les sauts de colonne."""
    last = previous.last
    if previous.framed != line.framed:
        return False
    if line.tag in HEADINGS:
        # Titre sur plusieurs lignes : chaque ligne pleine largeur ouvre une nouvelle bande, on
        # ne compare donc que la page, le style et l'écart vertical (lignes parfois chevauchantes).
        ok = previous.tag == line.tag and last.style == line.style and last.page == line.page
        ok = ok and -line.size / 2 <= line.y0 - last.y1 < line.size
    elif line.tag == "NOTE":
        ok = previous.tag == "NOTE" and not re.match(r"^\d+\.\s", line.text)
    elif line.tag == "PUCE" or previous.tag not in ("", "PUCE"):
        ok = False
    elif same_flow(last, line):
        ok = line.y0 - last.y1 <= last.paragraph_gap
        # Texte justifié : une ligne courte terminée par un point clôt le paragraphe, même
        # sans espace supplémentaire avant le suivant.
        short = last.x1 < last.column_right - SHORT_LINE_MARGIN
        ok = ok and not (short and re.search(r"[.!?]$", last.text))
        if previous.tag == "PUCE":
            ok = ok and line.x0 > previous.first.x0 + LIST_INDENT
    else:
        # Saut de colonne ou de page : on continue si la phrase n'est pas terminée.
        ok = not re.search(r"[.:;!?»)]$", last.text)
        if previous.tag == "PUCE":
            x0 = column_x0.get((line.page, line.column), line.x0)
            ok = ok and line.x0 > x0 + LIST_INDENT
        if ok:
            kind = "Liste" if previous.tag == "PUCE" else "Paragraphe"
            events.append(f"{kind} poursuivi(e) : {where(last)} → {where(line)}")
    if ok:
        previous.lines.append(line)
        previous.text = join(previous.text, line.text)
    return ok


# ---------------------------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------------------------
def style_table(reports: list[PageReport]) -> None:
    stats: dict[tuple[float, str, bool], dict[str, Any]] = {}
    for report in reports:
        for line in report.lines:
            entry = stats.setdefault(line.style, {"n": 0, "tags": Counter(), "ex": line.text})
            entry["n"] += 1
            entry["tags"][line.tag or "corps"] += 1
            if line.arrow:
                entry["tags"]["avec ➤"] += 1
    print("TABLEAU DES STYLES DE LIGNE (pages analysées)")
    header = (
        f"{'taille':>6} | {'couleur':7} | {'gras':4} | {'lignes':>6} | {'classement':32} | exemple"
    )
    print(header)
    print("-" * len(header))
    for (size, color, bold), entry in sorted(stats.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
        tags = ", ".join(f"{tag}×{n}" for tag, n in entry["tags"].most_common())
        example = entry["ex"][:45]
        gras = "oui" if bold else "non"
        print(f"{size:>6} | {color:7} | {gras:4} | {entry['n']:>6} | {tags:32} | {example}")
    print()


def print_page_header(report: PageReport, page_offset: int | None) -> None:
    expected = None if page_offset is None else report.number - page_offset
    if report.printed is None:
        verdict = "aucun numéro trouvé en marge"
    elif expected is None:
        verdict = "page_offset inconnu (document absent du manifeste)"
    else:
        ok = int(report.printed) == expected
        verdict = (
            f"attendu {expected} = {report.number} − {page_offset} → {'OK' if ok else 'ÉCART'}"
        )
    print("=" * 100)
    print(f"PAGE PDF {report.number} — numéro imprimé détecté : {report.printed} ({verdict})")
    spans = " | ".join(
        f"{'gauche' if c == 'L' else 'droite'} x {a:.0f}–{b:.0f}"
        for c, (a, b) in report.column_ranges.items()
    )
    print(
        f"  Colonnes : {report.columns} ({spans} ; milieu de la zone de texte {report.middle:.1f})"
    )
    print(
        f"  Encadrés : {report.frames} ; triangles ➤ dessinés : {report.arrows} ; "
        f"lignes de marge ignorées : {report.margin_lines}"
    )
    print("=" * 100)


def print_blocks(blocks: list[Block], page: int) -> None:
    wrapper = textwrap.TextWrapper(width=100, subsequent_indent="    ")
    for index, block in enumerate(blocks):
        if block.first.page != page:
            continue
        markers = []
        if block.framed:
            markers.append("[ENCADRÉ]")
        if block.tag:
            label = block.tag
            if block.tag == "SOUS-SECTION":
                numbered = NUMBERED_SUBSECTION.match(block.text)
                label += " numérotée" if numbered else (" ➤" if block.first.arrow else "")
            markers.append(f"[{label}]")
        if (
            len({line.page for line in block.lines}) > 1
            or len({ln.column for ln in block.lines}) > 1
        ):
            markers.append("[↪ suite colonne/page]")
        text = block.text.replace("\xad", "")
        print(wrapper.fill(" ".join(markers + [text])))
        following = blocks[index + 1] if index + 1 < len(blocks) else None
        list_ends = block.tag == "PUCE" and (following is None or following.tag != "PUCE")
        if block.tag in HEADINGS or block.tag == "" or list_ends:
            print()


def print_summary(
    reports: list[PageReport], blocks: list[Block], events: list[str], page_offset: int | None
) -> None:
    print("=" * 100)
    print("BILAN AUTOMATIQUE")
    print("=" * 100)
    for report in reports:
        expected = None if page_offset is None else report.number - page_offset
        found = report.printed
        ok = found is not None and expected is not None and int(found) == expected
        print(
            f"  p. {report.number} : imprimé {found} / attendu {expected} "
            f"{'OK' if ok else 'À VÉRIFIER'} ; {report.columns} colonne(s)"
        )
    counts = Counter(block.tag or "corps" for block in blocks)
    counts["ENCADRÉ (blocs)"] = sum(1 for block in blocks if block.framed)
    print("  Blocs : " + ", ".join(f"{tag} {n}" for tag, n in counts.most_common()))
    print("  Continuités colonne/page :")
    for event in events or ["aucune"]:
        print(f"    - {event}")
    suspects = [
        line
        for report in reports
        for line in report.lines
        if line.bold and line.tag in ("", "PUCE") and not line.framed
    ]
    suspects += [
        line
        for report in reports
        for line in report.lines
        if line.column == "F" and line.tag not in (*HEADINGS, "NOTE", "MARGE")
    ]
    print("  À vérifier (gras non classé en titre, ou pleine largeur hors titre/note) :")
    for line in suspects or []:
        print(f"    - p. {line.page} {line.style} : {line.text[:60]}")
    if not suspects:
        print("    - aucun")


# ---------------------------------------------------------------------------------------------
# Bilan du document entier (--summary) : une ligne par page, sans le texte complet.
# ---------------------------------------------------------------------------------------------
# Signatures exactes observées sur les pages 14–17 : (police, taille, couleur) → niveau.
HEADING_SIGNATURES = {
    ("TemplateGothicBold", 20.0, "498ca7"): "CHAPITRE",
    ("TemplateGothicBold", 14.0, "799a2c"): "SECTION",
    ("BellGothicStd-Bold", 12.0, "356d83"): "SOUS-SECTION",
    ("BellGothicStd-Bold", 11.0, "799a2c"): "INTERTITRE",
}
OTHER_SIGNATURES = {
    ("BellGothicStd-Light", 11.0, "000000"): "corps",
    ("BellGothicStd-Light", 8.0, "000000"): "note",
    ("TemplateGothicBold", 11.0, "498ca7"): "numéro de page",
}
EXPECTED_NUMBERING = {
    "CHAPITRE": re.compile(r"^\d+\s"),
    "SECTION": re.compile(r"^\d+\.\d+\.\s"),
    "SOUS-SECTION": re.compile(r"^\d+\.\d+\.\d+\.\s"),
}
LOOKS_NUMBERED = re.compile(r"^\d+(\.\d+)+\.?\s+[A-ZÉÈÀÂÎ]")
EXCLUSION_HINTS = {
    "sommaire": r"\bsommaire\b",
    "mentions légales / ISBN": r"\bisbn\b|dépôt légal|tous droits|reproduction",
    "crédits": r"\bcrédits?\b|photos?\s*:|illustrations?\s*:|maquette",
    "bibliographie": r"bibliographie|pour en savoir plus",
}


def signature(line: Line) -> tuple[str, float, str]:
    return (line.font.split("+")[-1], line.size, line.color)


def heading_level(block: Block) -> str:
    if block.tag == "SOUS-SECTION" and not NUMBERED_SUBSECTION.match(block.text):
        return "INTERTITRE"
    return block.tag


def summarize(document: Any, reports: list[PageReport]) -> None:
    blocks, events = build_blocks(reports)
    signatures: dict[tuple[str, float, str], dict[str, Any]] = {}
    problems: list[str] = []

    print(f"{'p.':>3} {'imp.':>4} {'col':>3} {'lign':>4} {'tab':>3} {'img':>3} {'enc':>3}  début")
    for report in reports:
        page = document[report.number - 1]
        tables = len(page.find_tables().tables)
        images = len(page.get_images())
        body = [line for line in report.lines if not line.margin]
        start = " / ".join(line.text[:35] for line in body[:2]) or "(page vide)"
        printed = report.printed or "—"
        print(
            f"{report.number:>3} {printed:>4} {report.columns:>3} {len(body):>4} {tables:>3} "
            f"{images:>3} {report.frames:>3}  {start}"
        )
        text = " ".join(line.text for line in body).lower()
        hints = [name for name, pattern in EXCLUSION_HINTS.items() if re.search(pattern, text)]
        if hints:
            print(f"{'':>26}indices : {', '.join(hints)}")
        for line in report.lines:
            key = signature(line)
            entry = signatures.setdefault(key, {"n": 0, "pages": set(), "ex": line.text})
            entry["n"] += 1
            entry["pages"].add(report.number)

    print("\nTITRES PAR PAGE (blocs fusionnés)")
    for report in reports:
        heads = [b for b in blocks if b.first.page == report.number and b.tag in HEADINGS]
        if heads:
            print(f"  p. {report.number}")
        for block in heads:
            level = heading_level(block)
            print(f"    {level:12} {block.text.replace(chr(0xAD), '')[:80]}")
            expected = [
                lvl for sig, lvl in HEADING_SIGNATURES.items() if sig == signature(block.first)
            ]
            if expected != [level]:
                problems.append(
                    f"p. {report.number} titre {level} hors signature {signature(block.first)} : "
                    f"{block.text[:50]}"
                )
            pattern = EXPECTED_NUMBERING.get(level)
            if pattern and not pattern.match(block.text):
                problems.append(
                    f"p. {report.number} numérotation inattendue ({level}) : {block.text[:50]}"
                )

    for report in reports:
        for line in report.lines:
            level = HEADING_SIGNATURES.get(signature(line))
            if level and line.tag not in HEADINGS:
                problems.append(
                    f"p. {report.number} signature de {level} non classée en titre "
                    f"({line.tag or 'corps'}{', encadré' if line.framed else ''}) : "
                    f"{line.text[:50]}"
                )
            if line.tag in ("", "PUCE") and LOOKS_NUMBERED.match(line.text):
                problems.append(f"p. {report.number} ligne numérotée en corps : {line.text[:50]}")

    print("\nSIGNATURES (police, taille, couleur) SUR LES PAGES ANALYSÉES")
    for key, entry in sorted(signatures.items(), key=lambda kv: -kv[1]["n"]):
        known = HEADING_SIGNATURES.get(key) or OTHER_SIGNATURES.get(key) or "INCONNUE"
        pages = sorted(entry["pages"])
        shown = ",".join(map(str, pages[:12])) + ("…" if len(pages) > 12 else "")
        font, size, color = key
        print(
            f"  {font[:22]:22} {size:>5} {color}  {entry['n']:>5} l.  {known:15} "
            f"p. {shown:28} ex. {entry['ex'][:30]}"
        )

    print("\nANOMALIES")
    one_column = [r.number for r in reports if r.columns == 1]
    no_number = [r.number for r in reports if r.printed is None]
    print(f"  Pages à 1 colonne : {one_column or 'aucune'}")
    print(f"  Pages sans numéro imprimé : {no_number or 'aucune'}")
    cross_page = [e for e in events if re.search(r"p\. (\d+).*→ p\. (?!\1)\d+", e)]
    print("  Continuités d'une page à l'autre :")
    for event in cross_page or ["aucune"]:
        print(f"    - {event}")
    print("  Titres et signatures :")
    for problem in problems or ["aucune"]:
        print(f"    - {problem}")


# ---------------------------------------------------------------------------------------------
def load_page_offset(pdf: Path) -> int | None:
    if not MANIFEST.exists():
        return None
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    for entry in manifest.get("documents", []):
        if (ROOT / entry["path"]).resolve() == pdf.resolve():
            return int(entry["page_offset"])
    return None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("pdf", type=Path, help="chemin du PDF")
    parser.add_argument("--pages", type=int, nargs="+", help="pages PDF (dès 1) ; défaut : toutes")
    parser.add_argument(
        "--summary", action="store_true", help="bilan par page sans le texte reconstruit"
    )
    args = parser.parse_args()

    document = pymupdf.open(args.pdf)
    args.pages = args.pages or list(range(1, document.page_count + 1))
    invalid = [p for p in args.pages if not 1 <= p <= document.page_count]
    if invalid:
        parser.error(f"pages hors du document (1–{document.page_count}) : {invalid}")

    page_offset = load_page_offset(args.pdf)
    print(f"Document : {args.pdf} ({document.page_count} pages) ; page_offset : {page_offset}\n")
    reports = [analyse_page(document, number, page_offset) for number in args.pages]
    if args.summary:
        summarize(document, reports)
        return 0
    blocks, events = build_blocks(reports)

    style_table(reports)
    for report in reports:
        print_page_header(report, page_offset)
        print_blocks(blocks, report.number)
    print_summary(reports, blocks, events, page_offset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
