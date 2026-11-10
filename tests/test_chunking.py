"""Tests du découpage structurel en passages.

Partie 1 : blocs SYNTHÉTIQUES (aucun PDF, tournent en CI).
Partie 2 : vrai PDF INRS (skipif absent). La sortie console ne contient que des nombres et des
ids, jamais de texte du PDF ; les échecs passent par ``pytest.fail`` pour la même raison.
"""

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from maintenance_rag.chunking import BLOCK_SEPARATOR, CONTINUATION, chunk_blocks, count_tokens
from maintenance_rag.config import ROOT, load_layout_profile, load_manifest
from maintenance_rag.normalize import normalize_for_match, quote_matches
from maintenance_rag.pdf_reader import HEADING_LEVELS, Block
from maintenance_rag.schemas import Chunk

SLUG = "test-doc"
CH = "1 Chapitre"
SEC_A = "1.1. Section A"
SEC_B = "1.2. Section B"


def _b(
    kind: str,
    text: str,
    path: list[str],
    page: int = 3,
    printed: int | None = 1,
    bold: bool = False,
) -> Block:
    return Block(
        pdf_page=page,
        printed_page=printed,
        kind=kind,  # type: ignore[arg-type]
        text=text,
        bold=bold,
        section_path=path,
    )


def _chunks(blocks: list[Block], max_tokens: int = 500) -> list[Chunk]:
    return chunk_blocks(blocks, max_tokens, document_slug=SLUG)


def _sample() -> list[Block]:
    """Petit document couvrant tous les types de blocs."""
    a, b = [CH, SEC_A], [CH, SEC_B]
    return [
        _b("chapter", CH, [CH]),
        _b("section", SEC_A, a),
        _b("paragraph", "Premier paragraphe de la section A.", a),
        _b("list_item", "- premier item ;", a),
        _b("list_item", "- second item.", a),
        _b("warning", "Attention !\nNe jamais travailler sous tension.", a),
        _b("box", "Contenu d'un encadré.", a),
        _b("caption", "Légende d'une photo.", a),
        _b("section", SEC_B, b),
        _b("paragraph", "Paragraphe de la section B, page 3.", b),
        _b("note", "1. Note de la section A.", a),  # fin de page : après la section B
        _b("paragraph", "Suite de la section B, page 4.", b, page=4, printed=2),
    ]


# --------------------------------------------------------------------------------------------
# Règles a) à i)
# --------------------------------------------------------------------------------------------
def test_a_section_change_closes_chunk() -> None:
    chunks = _chunks(_sample())
    assert [chunk.section_path for chunk in chunks] == [[CH, SEC_A], [CH, SEC_B]]
    assert all(len({tuple(chunk.section_path)}) == 1 for chunk in chunks)
    assert chunks[0].section_title == f"{CH} › {SEC_A}"


def test_b_units_are_never_split_and_headings_are_not_content() -> None:
    a = [CH, SEC_A]
    paragraph = "Un paragraphe introductif assez long pour remplir le passage."
    items = ["- premier item de la liste ;", "- deuxième item de la liste ;", "- dernier item."]
    blocks = [_b("section", SEC_A, a), _b("paragraph", paragraph, a)] + [
        _b("list_item", item, a) for item in items
    ]
    header = f"{CH} › {SEC_A}"
    # Le paragraphe tient avec un item, pas avec la liste entière : la liste passe d'un bloc.
    limit = count_tokens(f"{header}\n{paragraph}{BLOCK_SEPARATOR}{items[0]}")
    chunks = _chunks(blocks, limit)
    assert [chunk.content for chunk in chunks] == [paragraph, BLOCK_SEPARATOR.join(items)]
    assert not any(SEC_A in chunk.content for chunk in chunks)
    # Un titre seul ne produit aucun passage.
    assert _chunks([_b("chapter", CH, [CH]), _b("section", SEC_A, a)]) == []


def test_b_box_group_and_warning_are_units() -> None:
    a = [CH, SEC_A]
    boxes = ["Encadré, premier paragraphe.", "Encadré, second paragraphe."]
    warning = "Attention !\nCouper l'alimentation avant toute intervention."
    blocks = [_b("box", text, a) for text in boxes] + [_b("warning", warning, a)]
    limit = count_tokens(f"{CH} › {SEC_A}\n{BLOCK_SEPARATOR.join(boxes)}")
    chunks = _chunks(blocks, limit)
    assert [chunk.content for chunk in chunks] == [BLOCK_SEPARATOR.join(boxes), warning]


def test_c_oversized_unit_is_cut_between_items_with_suite_header() -> None:
    a = [CH, SEC_A]
    items = [f"- item numéro {i} de la très longue liste de contrôle ;" for i in range(1, 7)]
    blocks = [_b("list_item", item, a) for item in items]
    header = f"{CH} › {SEC_A}"
    limit = count_tokens(f"{header}{CONTINUATION}\n{BLOCK_SEPARATOR.join(items[:2])}")
    chunks = _chunks(blocks, limit)
    assert len(chunks) >= 3
    assert chunks[0].embedding_text.startswith(header + "\n")
    for chunk in chunks[1:]:
        assert chunk.embedding_text.split("\n", 1)[0] == header + CONTINUATION
    # Coupé entre deux items : chaque morceau est une suite d'items entiers.
    pieces = [piece for chunk in chunks for piece in chunk.content.split(BLOCK_SEPARATOR)]
    assert pieces == items
    assert all(chunk.token_count <= limit for chunk in chunks)


def test_c_oversized_paragraph_is_cut_between_sentences() -> None:
    a = [CH, SEC_A]
    sentences = [f"Phrase numéro {i} du paragraphe trop long." for i in range(1, 9)]
    chunks = _chunks([_b("paragraph", " ".join(sentences), a)], 40)
    assert len(chunks) > 1
    assert " ".join(chunk.content for chunk in chunks) == " ".join(sentences)
    assert all(chunk.content.endswith(".") for chunk in chunks)  # jamais dans une phrase


def test_d_same_section_across_two_pages_is_joined() -> None:
    chunk_b = _chunks(_sample())[1]
    assert (chunk_b.page_start, chunk_b.page_end) == (3, 4)
    assert (chunk_b.printed_page_start, chunk_b.printed_page_end) == (1, 2)
    assert "page 3" in chunk_b.content and "page 4" in chunk_b.content


def test_e_one_chunk_per_glossary_term() -> None:
    g = ["2 Définitions"]
    blocks = [_b("chapter", g[0], g)]
    for term, definition in [("Consignation", "Mise en sécurité."), ("Vérification", "Contrôle.")]:
        blocks += [
            _b("glossary_term", term, [*g, term], bold=True),
            _b("paragraph", definition, [*g, term]),
        ]
    chunks = _chunks(blocks)
    assert [(chunk.section_path[-1], chunk.content) for chunk in chunks] == [
        ("Consignation", "Mise en sécurité."),
        ("Vérification", "Contrôle."),
    ]
    assert chunks[0].embedding_text == "2 Définitions › Consignation\nMise en sécurité."


def test_f_note_goes_to_the_chunk_of_its_section() -> None:
    chunks = _chunks(_sample())
    assert "1. Note de la section A." in chunks[0].content
    assert "note" in chunks[0].block_types
    assert not any("Note de la section A" in chunk.content for chunk in chunks[1:])


def test_g_caption_is_kept_in_its_section_chunk() -> None:
    chunk_a = _chunks(_sample())[0]
    assert "Légende d'une photo." in chunk_a.content
    assert "caption" in chunk_a.block_types


def test_h_no_overlap_and_deterministic_indexes() -> None:
    blocks = _sample()
    chunks = _chunks(blocks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    for block in blocks:
        if block.kind not in HEADING_LEVELS:
            assert sum(block.text in chunk.content for chunk in chunks) == 1, block.text


def test_i_has_warning() -> None:
    chunks = _chunks(_sample())
    assert [chunk.has_warning for chunk in chunks] == [True, False]


# --------------------------------------------------------------------------------------------
# Invariants
# --------------------------------------------------------------------------------------------
def test_no_empty_chunk() -> None:
    assert all(chunk.content.strip() and chunk.token_count > 0 for chunk in _chunks(_sample()))


def test_nothing_lost_nothing_duplicated() -> None:
    blocks = [block for block in _sample() if block.kind != "note"]  # ordre inchangé sans note
    expected = BLOCK_SEPARATOR.join(b.text for b in blocks if b.kind not in HEADING_LEVELS)
    assert BLOCK_SEPARATOR.join(chunk.content for chunk in _chunks(blocks)) == expected
    # Avec la note déplacée : même multiensemble de blocs.
    contents = [p for c in _chunks(_sample()) for p in c.content.split(BLOCK_SEPARATOR)]
    texts = [b.text for b in _sample() if b.kind not in HEADING_LEVELS]
    assert Counter(contents) == Counter(texts)


def test_same_input_same_output() -> None:
    assert _chunks(_sample()) == _chunks(_sample())


def test_embedding_text_and_token_count() -> None:
    for chunk in _chunks(_sample()):
        assert chunk.embedding_text == f"{chunk.section_title}\n{chunk.content}"
        assert chunk.token_count == count_tokens(chunk.embedding_text)


# --------------------------------------------------------------------------------------------
# Vrai PDF INRS (local uniquement)
# --------------------------------------------------------------------------------------------
INRS = next(entry for entry in load_manifest() if entry.id == "inrs-ed6109")
PDF_PATH = ROOT / INRS.path
CONFIG = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
MAX_TOKENS: int = CONFIG["chunking"]["max_tokens"]
MIN_CHARS: int = CONFIG["guardrails"]["min_quote_chars"]
QUESTIONS = yaml.safe_load((ROOT / "evals" / "questions_dev.yaml").read_text(encoding="utf-8"))
VALID = [q for q in QUESTIONS["questions"] if not q["must_refuse"]]
real_pdf = pytest.mark.skipif(not PDF_PATH.exists(), reason="PDF INRS absent (non versionné)")

RERANK_MODEL = CONFIG["rerank"]["model"]
FLASHRANK_CACHE = ROOT / Path(CONFIG["rerank"]["cache_dir"])  # cache local du modèle FlashRank
BERT_LIMIT = 512
QUESTION_TOKENS = 30


@pytest.fixture(scope="module")
def blocks() -> list[Block]:
    from maintenance_rag.pdf_reader import read_pdf

    assert INRS.layout_profile is not None
    return read_pdf(PDF_PATH, INRS, load_layout_profile(INRS.layout_profile))


def _found(question: dict[str, Any], chunks: list[Chunk]) -> bool:
    page = question["expected_page"]
    return any(
        chunk.printed_page_start is not None
        and chunk.printed_page_end is not None
        and chunk.printed_page_start <= page <= chunk.printed_page_end
        and quote_matches(question["exact_quote"], chunk.content, MIN_CHARS)
        for chunk in chunks
    )


@real_pdf
@pytest.mark.parametrize("question", VALID, ids=[q["id"] for q in VALID])
def test_real_quote_is_in_one_chunk(question: dict[str, Any], blocks: list[Block]) -> None:
    chunks = chunk_blocks(blocks, MAX_TOKENS, document_slug=INRS.id)
    if not _found(question, chunks):
        page = question["expected_page"]
        on_page = [
            (chunk.chunk_index, chunk.block_types)
            for chunk in chunks
            if chunk.printed_page_start is not None
            and chunk.printed_page_end is not None
            and chunk.printed_page_start <= page <= chunk.printed_page_end
        ]
        pytest.fail(
            f"{question['id']} : citation dans aucun passage de la page imprimée {page} ; "
            f"{len(on_page)} passage(s) sur cette page (index, types) : {on_page}",
            pytrace=False,
        )


@real_pdf
def test_real_electrical_operations_in_one_chunk(blocks: list[Block]) -> None:
    chunks = chunk_blocks(blocks, MAX_TOKENS, document_slug=INRS.id)
    needles = [normalize_for_match(f"(opération {n})") for n in range(1, 6)]
    holders = [
        chunk.chunk_index
        for chunk in chunks
        if chunk.printed_page_start == 14
        and all(needle in normalize_for_match(chunk.content) for needle in needles)
    ]
    if len(holders) != 1:
        pytest.fail(f"opérations 1 à 5 : passages qui les contiennent toutes : {holders}")


@real_pdf
def test_real_red_colour_block_is_chunked(blocks: list[Block]) -> None:
    candidates = [b for b in blocks if b.printed_page == 29 and "couleur rouge" in b.text]
    if not candidates:
        pytest.fail("aucun bloc de la page imprimée 29 ne contient le mot-clé", pytrace=False)
    text = candidates[0].text
    start = max(0, text.index("couleur rouge") - 15)
    fragment = text[start : start + 40]  # choisi en mémoire, jamais écrit ni affiché
    chunks = chunk_blocks(blocks, MAX_TOKENS, document_slug=INRS.id)
    holders = [c.chunk_index for c in chunks if quote_matches(fragment, c.content, MIN_CHARS)]
    if not holders:
        pytest.fail("fragment « couleur rouge » de la p. 29 absent des passages", pytrace=False)


@real_pdf
def test_real_no_duplicate_chunk(blocks: list[Block]) -> None:
    chunks = chunk_blocks(blocks, MAX_TOKENS, document_slug=INRS.id)
    counts = Counter(normalize_for_match(chunk.content) for chunk in chunks)
    duplicated = [c.chunk_index for c in chunks if counts[normalize_for_match(c.content)] > 1]
    if duplicated:
        pytest.fail(f"passages en double (index) : {duplicated}", pytrace=False)


# --------------------------------------------------------------------------------------------
# Mesure (affichée avec -s) : uniquement des nombres
# --------------------------------------------------------------------------------------------
def _bert_tokenizer() -> Any | None:
    """Tokenizer du modèle FlashRank s'il est déjà en cache local ; jamais téléchargé ici."""
    path = FLASHRANK_CACHE / RERANK_MODEL / "tokenizer.json"
    if not path.exists():
        return None
    from tokenizers import Tokenizer

    return Tokenizer.from_str(json.dumps(json.loads(path.read_text(encoding="utf-8"))))


def _p95(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]


def measure(blocks: list[Block], sizes: tuple[int, ...] = (300, 350, 400, 500)) -> int | None:
    """Imprime les statistiques par max_tokens ; renvoie la valeur recommandée (ou None)."""
    tokenizer = _bert_tokenizer()
    print(f"\nMESURE (tokenizer BERT {RERANK_MODEL} : {'oui' if tokenizer else 'INDISPONIBLE'})")
    print("max_tokens chunks tik_min tik_med tik_p95 tik_max bert_med bert_p95 bert_max >512 cit")
    eligible: list[int] = []
    for size in sizes:
        chunks = chunk_blocks(blocks, size, document_slug=INRS.id)
        tik = [chunk.token_count for chunk in chunks]
        found = sum(_found(question, chunks) for question in VALID)
        bert_cols = "       -        -        -    -"
        over: int | None = None
        if tokenizer is not None:
            # [CLS] question [SEP] passage [SEP] : question type de 30 tokens + 3 spéciaux.
            bert = [
                QUESTION_TOKENS
                + 3
                + len(tokenizer.encode(c.embedding_text, add_special_tokens=False))
                for c in chunks
            ]
            over = sum(n > BERT_LIMIT for n in bert)
            bert_cols = f"{statistics.median(bert):8.0f} {_p95(bert):8d} {max(bert):8d} {over:4d}"
        print(
            f"{size:10d} {len(chunks):6d} {min(tik):7d} {statistics.median(tik):7.0f} "
            f"{_p95(tik):7d} {max(tik):7d} {bert_cols} {found:>2d}/{len(VALID)}"
        )
        if over == 0 and found == len(VALID):
            eligible.append(size)
    recommended = max(eligible) if eligible else None
    print(f"RECOMMANDATION max_tokens : {recommended if recommended else 'aucune valeur éligible'}")
    return recommended


@real_pdf
def test_real_measure(blocks: list[Block]) -> None:
    measure(blocks)


@real_pdf
def test_real_questions_fit_bert_window(blocks: list[Block]) -> None:
    """Pire cas : chaque question RÉELLE + [CLS]/[SEP]/[SEP] + le plus long embedding_text."""
    tokenizer = _bert_tokenizer()
    if tokenizer is None:
        pytest.skip(f"tokenizer FlashRank absent de {CONFIG['rerank']['cache_dir']}")
    chunks = chunk_blocks(blocks, MAX_TOKENS, document_slug=INRS.id)
    longest_chunk = max(
        len(tokenizer.encode(chunk.embedding_text, add_special_tokens=False)) for chunk in chunks
    )
    lengths = {
        q["id"]: len(tokenizer.encode(q["question"], add_special_tokens=False))
        for q in QUESTIONS["questions"]
    }
    longest_id = max(lengths, key=lambda qid: (lengths[qid], qid))
    print(f"\nquestion la plus longue : {longest_id} ({lengths[longest_id]} tokens BERT)")
    too_long = [qid for qid, n in lengths.items() if n + 3 + longest_chunk > BERT_LIMIT]
    if too_long:
        pytest.fail(
            f"dépassent {BERT_LIMIT} tokens BERT avec le plus long passage "
            f"({longest_chunk} tokens) : {too_long}",
            pytrace=False,
        )


@real_pdf
@pytest.mark.xfail(
    strict=True, reason="encadré p.22 rattaché au mauvais intertitre, voir docs/corpus.md"
)
def test_real_clp_box_section(blocks: list[Block]) -> None:
    from maintenance_rag.identifiers import extract_identifiers

    chunks = chunk_blocks(blocks, MAX_TOKENS, document_slug=INRS.id)
    holders = [chunk for chunk in chunks if "CLP" in extract_identifiers(chunk.content)]
    assert holders, "aucun passage ne contient l'identifiant CLP"
    for chunk in holders:
        assert "Principes de séparation de circuits" not in chunk.section_title, chunk.chunk_index
