"""Funções puras de apresentação do dashboard (dono: I4), testáveis sem Streamlit."""

from __future__ import annotations

from typing import Any

import pandas as pd

# Caracteres que o Markdown do Streamlit interpretaria dentro do texto clínico.
_MD_SPECIAL = str.maketrans({c: f"\\{c}" for c in "\\`*_[]#<>|~$"})


def md_escape(text: str) -> str:
    return text.translate(_MD_SPECIAL)


def highlight_terms(text: str, terms: list[dict[str, Any]]) -> str:
    """Markdown com os termos críticos em vermelho.

    Usa ``offset``/``length`` do I2 quando batem com o texto; senão procura o
    termo por nome (formato dos mocks). O resto do texto é escapado.
    """
    spans: list[tuple[int, int]] = []
    for t in terms:
        off, length, term = t.get("offset"), t.get("length"), t.get("term", "")
        if isinstance(off, int) and isinstance(length, int) and text[off : off + length] == term and length > 0:
            spans.append((off, off + length))
        elif term and (i := text.find(term)) >= 0:
            spans.append((i, i + len(term)))
    out, pos = [], 0
    for start, end in sorted(set(spans)):
        if start < pos:  # sobreposição: mantém o primeiro
            continue
        out.append(md_escape(text[pos:start]))
        out.append(f"**:red[{md_escape(text[start:end])}]**")
        pos = end
    out.append(md_escape(text[pos:]))
    return "".join(out)


def series_layout(df: pd.DataFrame) -> tuple[str | None, list[str], list[str]]:
    """Coluna de tempo, colunas numéricas e colunas de marcação de anomalia.

    O formato do series_flagged.parquet ainda não foi fechado pelo I3, então a
    leitura é tolerante: tempo = ``timestamp`` (ou a primeira coluna de data);
    marcação = colunas booleanas com "anom" ou "flag" no nome.
    """
    time_col = "timestamp" if "timestamp" in df.columns else None
    if time_col is None:
        dates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        time_col = dates[0] if dates else None
    flags = [
        c
        for c in df.columns
        if pd.api.types.is_bool_dtype(df[c]) and any(k in c.lower() for k in ("anom", "flag"))
    ]
    numeric = [
        c
        for c in df.columns
        if c != time_col and c not in flags and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]
    return time_col, numeric, flags


LONG_COLUMNS = {"timestamp", "signal", "value", "flagged"}


def series_long(df: pd.DataFrame) -> pd.DataFrame | None:
    """Séries do I3 em formato longo: ``timestamp, serie, value, flagged, severity``.

    Aceita o formato do Integrante 3 (longo: ``source``, ``signal``, ``value``,
    ``flagged``, ``severity``...) e, por tolerância, um formato largo (uma
    coluna por sinal e colunas booleanas de marcação). ``None`` se não reconhecer.
    """
    if LONG_COLUMNS <= set(df.columns):
        out = pd.DataFrame(
            {
                "timestamp": df["timestamp"],
                "serie": (df["source"].astype(str) + " · " if "source" in df.columns else "") + df["signal"].astype(str),
                "value": pd.to_numeric(df["value"], errors="coerce"),
                "flagged": df["flagged"].fillna(False).astype(bool),
                "severity": df["severity"] if "severity" in df.columns else None,
            }
        )
        return out.dropna(subset=["value"])
    time_col, numeric, flags = series_layout(df)
    if time_col is None or not numeric:
        return None
    flagged = df[flags].any(axis=1) if flags else pd.Series(False, index=df.index)
    parts = [
        pd.DataFrame({"timestamp": df[time_col], "serie": c, "value": df[c], "flagged": flagged, "severity": None})
        for c in numeric
    ]
    return pd.concat(parts, ignore_index=True)


def note_changes(note: dict[str, Any]) -> str:
    """Resumo da comparação com a nota anterior feita pelo I2."""
    c = note.get("comparison") or {}
    if not c.get("previous_note"):
        return "Primeira nota da série (sem comparação)."
    parts = []
    if c.get("added_terms"):
        parts.append("novos: " + ", ".join(c["added_terms"]))
    if c.get("persisting_terms"):
        parts.append("persistem: " + ", ".join(c["persisting_terms"]))
    if c.get("removed_terms"):
        parts.append("saíram: " + ", ".join(c["removed_terms"]))
    detail = "; ".join(parts) if parts else "sem mudança nos termos críticos"
    return f"Comparada com `{c['previous_note']}`: {detail}."
