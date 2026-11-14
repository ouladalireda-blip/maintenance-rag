"""Assistant consignation — interface Streamlit du technicien.

    uv run streamlit run ui/app.py        (l'API doit tourner : API_URL, défaut localhost:8000)

L'interface est un CLIENT de l'API (``ui.client``) : elle n'accède jamais à la base ni au LLM.
Elle n'affiche aucun texte du guide en dehors des extraits cités renvoyés par l'API. Les
exemples cliquables sont des questions du jeu DEV (``ui.examples`` de config.yaml), lues dans
evals/questions_dev.yaml ; le jeu test n'est jamais utilisé.
"""

import html
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # « streamlit run ui/app.py » : rendre le paquet ui importable
    sys.path.insert(0, str(ROOT))

from ui.client import AVERTISSEMENT, Answer, ApiClient, ApiError  # noqa: E402

TITLE = "Assistant consignation — guide INRS ED 6109"
BANNERS = {  # statut → (texte, couleur du texte, fond, bordure)
    "ANSWERED": ("Réponse trouvée dans le guide", "#14532d", "#dcfce7", "#16a34a"),
    "REFUSED_BY_LLM": ("Information précise absente du guide", "#7c2d12", "#ffedd5", "#ea580c"),
    "REFUSED_BY_GUARDRAIL": ("Pas de réponse vérifiable", "#1f2937", "#e5e7eb", "#6b7280"),
}
CATEGORY_LABELS = {"valid": "réponse dans le guide", "near_miss": "piège : info absente"}
# Libellés courts des boutons d'exemple (jeu DEV) ; la question complète est envoyée au clic.
SHORT_LABELS = {
    "INRS-DEV-007": "Qui déclare une conduite déconsignée ?",
    "INRS-DEV-028": "VAT : sur quels conducteurs ?",
    "INRS-DEV-037": "Attente après arrêt d'un extracteur ?",
    "INRS-DEV-041": "Oxygène mini après purge à l'azote ?",
}
SUMMARY_TITLE = "Résumé automatique — vérifiez dans l'extrait ci-dessus"


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    category: str


@dataclass(frozen=True)
class HistoryItem:
    question: str
    answer: Answer


def _ui_config() -> dict[str, Any]:
    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    return dict(config.get("ui") or {})


@st.cache_data
def load_examples() -> list[Example]:
    """Exemples du jeu DEV uniquement ; un id hors du jeu dev est refusé."""
    ids = [str(i) for i in _ui_config().get("examples", [])]
    if any(not i.startswith("INRS-DEV-") for i in ids):
        raise ValueError("ui.examples : seules des questions du jeu DEV sont autorisées")
    dev = yaml.safe_load((ROOT / "evals" / "questions_dev.yaml").read_text(encoding="utf-8"))
    by_id = {q["id"]: q for q in dev["questions"]}
    return [Example(i, by_id[i]["question"], by_id[i]["category"]) for i in ids]


@st.cache_resource
def api_client() -> ApiClient:
    return ApiClient()


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "question": "",
        "pending": None,  # question à envoyer au prochain affichage
        "answer": None,
        "asked": None,  # question de la réponse affichée
        "error": None,
        "history": [],
        "feedback": {},  # request_id → utile
        "feedback_error": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# ----------------------------------------------------------------------------- actions
def _submit() -> None:
    st.session_state.pending = st.session_state.question


def _use_example(question: str) -> None:
    st.session_state.question = question
    st.session_state.pending = question


def _show_history(index: int) -> None:
    item: HistoryItem = st.session_state.history[index]
    st.session_state.question = item.question
    st.session_state.answer = item.answer
    st.session_state.asked = item.question
    st.session_state.error = None
    st.session_state.feedback_error = None


def _send_feedback(request_id: str, utile: bool) -> None:
    try:
        api_client().feedback(request_id, utile)
    except ApiError as error:
        st.session_state.feedback_error = error.message
        return
    st.session_state.feedback[request_id] = utile
    st.session_state.feedback_error = None


def _ask(question: str) -> None:
    question = question.strip()
    st.session_state.error = None
    st.session_state.feedback_error = None
    try:
        with st.spinner("Recherche dans le guide…"):
            answer = api_client().ask(question)
    except ApiError as error:
        st.session_state.error = error.message
        st.session_state.answer = None
        return
    st.session_state.answer = answer
    st.session_state.asked = question
    size = int(_ui_config().get("history_size", 5))
    history = [item for item in st.session_state.history if item.question != question]
    st.session_state.history = [HistoryItem(question, answer), *history][:size]


# ----------------------------------------------------------------------------- affichage
def _box(text: str, color: str, background: str, border: str, size: str = "1rem") -> None:
    st.markdown(
        f'<div style="color:{color};background:{background};border-left:6px solid {border};'
        f'padding:0.6rem 0.9rem;border-radius:0.3rem;font-size:{size};font-weight:600;">'
        f"{html.escape(text)}</div>",
        unsafe_allow_html=True,
    )


def _render_quotes(answer: Answer) -> None:
    """Les extraits exacts, en grand : c'est la réponse principale (texte vérifié du guide)."""
    for citation in answer.citations:
        page = f"page {citation.page_imprimee}" if citation.page_imprimee else "page non numérotée"
        st.markdown(
            '<blockquote style="border-left:6px solid #1d4ed8;background:#f8fafc;'
            'margin:0 0 0.3rem 0;padding:0.7rem 1rem;font-size:1.3rem;line-height:1.5;">'
            f"« {html.escape(citation.extrait_exact)} »</blockquote>"
            '<div style="color:#334155;font-size:1rem;margin-bottom:1.2rem;">'
            f"<b>{html.escape(page.capitalize())}</b> — {html.escape(citation.section)}</div>",
            unsafe_allow_html=True,
        )


def _render_summary(text: str) -> None:
    """Le résumé du LLM, en plus petit : il n'est pas vérifié mot pour mot (ADR 0001)."""
    st.markdown(
        f'<div style="color:#475569;font-size:0.85rem;font-weight:600;margin-top:0.4rem;">'
        f"{html.escape(SUMMARY_TITLE)}</div>"
        f'<div style="color:#475569;font-size:0.95rem;line-height:1.45;margin:0.2rem 0 0.8rem;">'
        f"{html.escape(text)}</div>",
        unsafe_allow_html=True,
    )


def _render_feedback(answer: Answer) -> None:
    done = st.session_state.feedback.get(answer.request_id) is not None
    useful, useless, _ = st.columns([1, 1, 4])
    useful.button(
        "👍 Utile",
        key=f"utile-{answer.request_id}",
        disabled=done,
        on_click=_send_feedback,
        args=(answer.request_id, True),
    )
    useless.button(
        "👎 Pas utile",
        key=f"inutile-{answer.request_id}",
        disabled=done,
        on_click=_send_feedback,
        args=(answer.request_id, False),
    )
    if done:
        st.caption("Merci, votre avis est enregistré.")
    if st.session_state.feedback_error:
        st.warning(st.session_state.feedback_error)


def _render_answer(answer: Answer) -> None:
    label, color, background, border = BANNERS[answer.status]
    _box(label, color, background, border, size="1.1rem")
    st.write("")
    if answer.citations:
        if answer.status == "ANSWERED":
            st.subheader("Extrait du guide" if len(answer.citations) == 1 else "Extraits du guide")
        else:
            st.subheader("Ce que le guide dit sur ce sujet")
        _render_quotes(answer)
        _render_summary(answer.reponse)
    else:  # refus sans information liée vérifiée : message du pipeline
        st.markdown(
            f'<div style="font-size:1.15rem;line-height:1.5;margin:0.4rem 0 0.8rem;">'
            f"{html.escape(answer.reponse)}</div>",
            unsafe_allow_html=True,
        )
    cost = "-" if answer.cout_usd is None else f"{answer.cout_usd:.4f} $"
    st.caption(f"Temps de réponse : {answer.latence_ms / 1000:.1f} s · coût : {cost}")
    _render_feedback(answer)


def main() -> None:
    st.set_option("client.toolbarMode", "minimal")  # masque « Deploy » et le menu développeur
    st.set_page_config(page_title="Assistant consignation", page_icon="🔒", layout="wide")
    _init_state()

    st.title(TITLE)
    answer: Answer | None = st.session_state.answer
    _box(
        "⚠ " + (answer.avertissement if answer else AVERTISSEMENT),
        "#713f12",
        "#fef9c3",
        "#ca8a04",
        size="1.05rem",
    )
    st.write("")

    st.text_area(
        "Votre question",
        key="question",
        max_chars=500,
        height=100,
        placeholder="Ex. : qui peut déclarer une conduite déconsignée ?",
    )
    st.button("Demander", type="primary", on_click=_submit)

    st.caption("Exemples de questions")
    examples = load_examples()
    for column, example in zip(st.columns(len(examples)), examples, strict=True):
        column.button(
            SHORT_LABELS.get(example.id, example.question),  # la question complète est envoyée
            key=f"exemple-{example.id}",
            help=f"{example.question} ({example.id} — "
            f"{CATEGORY_LABELS.get(example.category, example.category)})",
            on_click=_use_example,
            args=(example.question,),
            use_container_width=True,
        )

    if st.session_state.pending is not None:
        question = st.session_state.pending
        st.session_state.pending = None
        _ask(question)
        answer = st.session_state.answer

    st.divider()
    if st.session_state.error:
        st.error(st.session_state.error)
    elif answer is not None:
        st.caption(f"Question : {st.session_state.asked}")
        _render_answer(answer)

    if st.session_state.history:
        st.divider()
        st.subheader("Historique de la session")
        for index, item in enumerate(st.session_state.history):
            st.button(
                item.question,
                key=f"historique-{index}-{item.answer.request_id}",
                on_click=_show_history,
                args=(index,),
            )


main()
