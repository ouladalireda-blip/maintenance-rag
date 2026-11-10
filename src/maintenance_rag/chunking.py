"""Découpage des documents en passages (max_tokens, sans jamais couper une liste) avec titre de
section et plage de pages pour la citation.

Découpage STRUCTUREL des blocs produits par ``pdf_reader.read_pdf`` :

- une section (``section_path``) n'est jamais mélangée à une autre : tout changement de chemin
  ferme le passage en cours ;
- les titres ne forment pas de passage : ils vivent dans ``section_path`` et dans l'en-tête ;
- unités insécables : une liste complète (items consécutifs), un groupe d'encadrés consécutifs,
  un « Attention ! » (avec le bloc qui lui est fusionné), un paragraphe, une légende, une note ;
- les unités s'accumulent jusqu'à ``max_tokens`` (comptés sur ``embedding_text``). Une unité qui
  dépasse seule est coupée entre deux items, à défaut entre deux phrases, jamais au milieu d'une
  phrase ; l'en-tête des morceaux suivants se termine par « (suite) » ;
- deux blocs consécutifs d'une même section sur deux pages restent ensemble ;
- glossaire : chaque terme est un intertitre, donc une section : un passage par terme ;
- notes de bas de page : déplacées derrière le dernier bloc de leur section qui les précède
  (``read_pdf`` les place en fin de page, souvent après un changement de section).

Le résultat est déterministe : mêmes blocs, mêmes passages, mêmes ``chunk_index``.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache

import tiktoken

from maintenance_rag.pdf_reader import HEADING_LEVELS, Block
from maintenance_rag.schemas import Chunk

SECTION_SEPARATOR = " › "
BLOCK_SEPARATOR = "\n\n"  # entre deux blocs d'un passage
CONTINUATION = " (suite)"
ENCODING = "cl100k_base"

_SENTENCE = re.compile(r"(?<=[.!?…])\s+(?=\S)")


@cache
def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING)


def count_tokens(text: str) -> int:
    """Nombre de tokens tiktoken (cl100k_base) du texte."""
    return len(_encoder().encode(text))


@dataclass(frozen=True)
class _Part:
    """Morceau de texte d'un bloc ; ``joiner`` vide = début de bloc."""

    text: str
    kind: str
    pdf_page: int
    printed_page: int | None
    joiner: str = ""


_Unit = list[_Part]


def _is_heading(block: Block) -> bool:
    return block.kind in HEADING_LEVELS


def _relocate_notes(blocks: Sequence[Block]) -> list[Block]:
    """Place chaque note derrière le dernier bloc de sa section qui la précède."""
    ordered: list[Block] = []
    for block in blocks:
        if block.kind == "note":
            target = next(
                (
                    i
                    for i in range(len(ordered) - 1, -1, -1)
                    if ordered[i].section_path == block.section_path
                ),
                None,
            )
            if target is not None:
                # Après les éventuelles notes déjà rattachées au même endroit.
                while (
                    target + 1 < len(ordered)
                    and ordered[target + 1].kind == "note"
                    and ordered[target + 1].section_path == block.section_path
                ):
                    target += 1
                ordered.insert(target + 1, block)
                continue
        ordered.append(block)
    return ordered


def _sections(blocks: Sequence[Block]) -> list[tuple[list[str], list[Block]]]:
    """Suites de blocs consécutifs au même ``section_path`` (titres compris)."""
    runs: list[tuple[list[str], list[Block]]] = []
    for block in blocks:
        if runs and runs[-1][0] == block.section_path:
            runs[-1][1].append(block)
        else:
            runs.append((list(block.section_path), [block]))
    return runs


def _part(block: Block) -> _Part:
    return _Part(block.text, block.kind, block.pdf_page, block.printed_page)


def _units(blocks: Iterable[Block]) -> list[_Unit]:
    """Regroupe les blocs (hors titres) en unités insécables."""
    units: list[_Unit] = []
    previous: str | None = None
    for block in blocks:
        if _is_heading(block):
            previous = None  # un titre interrompt une liste ou un groupe d'encadrés
            continue
        grouped = block.kind in ("list_item", "box") and block.kind == previous
        if grouped:
            units[-1].append(_part(block))
        else:
            units.append([_part(block)])
        previous = block.kind
    return units


def _finer(part: _Part) -> list[_Part]:
    """Morceaux d'un bloc trop long : ses lignes (items fusionnés), sinon ses phrases."""
    lines = part.text.split("\n")
    if len(lines) > 1:
        pieces, joiner = lines, "\n"
    else:
        pieces, joiner = _SENTENCE.split(part.text), " "
    if len(pieces) == 1:
        return [part]  # une seule phrase : on ne la coupe jamais
    return [
        _Part(text, part.kind, part.pdf_page, part.printed_page, part.joiner if i == 0 else joiner)
        for i, text in enumerate(pieces)
    ]


def _content(parts: Sequence[_Part]) -> str:
    text = parts[0].text
    for part in parts[1:]:
        text += (part.joiner or BLOCK_SEPARATOR) + part.text
    return text


def _embedding_text(header: str, content: str) -> str:
    return f"{header}\n{content}" if header else content


def _fits(header: str, parts: Sequence[_Part], max_tokens: int) -> bool:
    return count_tokens(_embedding_text(header, _content(parts))) <= max_tokens


def _split_unit(unit: _Unit, header: str, max_tokens: int) -> list[list[_Part]]:
    """Coupe une unité trop longue entre deux items (ou deux phrases), jamais dans une phrase."""
    suite = header + CONTINUATION
    atoms: list[_Part] = []
    for part in unit:
        if _fits(suite, [part], max_tokens):
            atoms.append(part)
        else:
            for piece in _finer(part):
                atoms.extend(_finer(piece) if not _fits(suite, [piece], max_tokens) else [piece])
    pieces: list[list[_Part]] = []
    for atom in atoms:
        current_header = suite if pieces else header
        if pieces and _fits(current_header, [*pieces[-1], atom], max_tokens):
            pieces[-1].append(atom)
        else:
            pieces.append([atom])
    return pieces


def chunk_blocks(blocks: Sequence[Block], max_tokens: int, *, document_slug: str) -> list[Chunk]:
    """Découpe les blocs d'un document en passages (voir le docstring du module)."""
    if max_tokens < 1:
        raise ValueError("max_tokens doit être positif")
    drafts: list[tuple[list[str], list[_Part], bool]] = []  # (chemin, parties, suite ?)
    for path, section in _sections(_relocate_notes(blocks)):
        header = SECTION_SEPARATOR.join(path)
        current: list[_Part] = []
        for unit in _units(section):
            if current and _fits(header, [*current, *unit], max_tokens):
                current.extend(unit)
                continue
            if current:
                drafts.append((path, current, False))
                current = []
            if _fits(header, unit, max_tokens):
                current = list(unit)
            else:
                for index, piece in enumerate(_split_unit(unit, header, max_tokens)):
                    drafts.append((path, piece, index > 0))
        if current:
            drafts.append((path, current, False))

    chunks: list[Chunk] = []
    for index, (path, parts, continued) in enumerate(drafts):
        title = SECTION_SEPARATOR.join(path)
        content = _content(parts)
        embedding_text = _embedding_text(title + CONTINUATION if continued else title, content)
        printed = [part.printed_page for part in parts if part.printed_page is not None]
        kinds = list(dict.fromkeys(part.kind for part in parts))
        chunks.append(
            Chunk(
                document_slug=document_slug,
                chunk_index=index,
                section_path=path,
                section_title=title,
                page_start=min(part.pdf_page for part in parts),
                page_end=max(part.pdf_page for part in parts),
                printed_page_start=min(printed) if printed else None,
                printed_page_end=max(printed) if printed else None,
                content=content,
                embedding_text=embedding_text,
                token_count=count_tokens(embedding_text),
                has_warning="warning" in kinds,
                block_types=kinds,
            )
        )
    return chunks
