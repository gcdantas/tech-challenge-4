"""Leitura das saídas dos módulos em outputs/ (I4 só consome arquivos e a interface analyze)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from contracts.finding import Finding, read_findings
from contracts.patients import DEFAULT_PATIENTS_PATH, REPO_ROOT

# Pastas de cada módulo em outputs/ (plano v5, tabelas de saída).
MODULE_DIRS = ("video", "audio", "text", "anomaly")


def outputs_dir() -> Path:
    """Pasta de saídas; pode ser trocada com TC_OUTPUTS_DIR (usado nos testes)."""
    env = os.environ.get("TC_OUTPUTS_DIR")
    return Path(env) if env else REPO_ROOT / "outputs"


def patients_path() -> Path:
    env = os.environ.get("TC_PATIENTS_PATH")
    return Path(env) if env else DEFAULT_PATIENTS_PATH


@dataclass
class LoadResult:
    findings: list[Finding] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _module_files(out_dir: Path, pattern: str) -> list[Path]:
    """Arquivo nas saídas oficiais (outputs/<pattern>) e nos envios pelo Streamlit
    (outputs/uploads/<id>/<pattern>), nessa ordem."""
    return sorted(out_dir.glob(pattern)) + sorted(out_dir.glob(f"uploads/*/{pattern}"))


def load_findings(out_dir: Path, patient_id: str | None = None) -> LoadResult:
    """Lê <modulo>/<paciente>/findings.jsonl de todos os módulos, incluindo os envios."""
    result = LoadResult()
    pid = patient_id or "*"
    for module in MODULE_DIRS:
        for path in _module_files(out_dir, f"{module}/{pid}/findings.jsonl"):
            try:
                result.findings.extend(read_findings(path))
                result.files.append(path)
            except (ValueError, OSError) as exc:
                result.errors.append(str(exc))
    result.findings.sort(key=lambda f: (f.timestamp_start, f.source))
    return result


def findings_dataframe(findings: list[Finding]) -> pd.DataFrame:
    rows = []
    for f in findings:
        rows.append(
            {
                "inicio": f.timestamp_start,
                "fim": f.timestamp_end,
                "origem": f.source,
                "tipo": f.type,
                "severidade": f.severity,
                "score": f.score,
                "descricao": f.description,
                "modelo": f"{f.model.name} {f.model.version}",
                "mock": bool(f.evidence.get("mock")),
                "finding_id": str(f.finding_id),
            }
        )
    return pd.DataFrame(rows)


def load_movement(out_dir: Path, patient_id: str) -> pd.DataFrame | None:
    path = out_dir / "video" / patient_id / "movement.parquet"
    return pd.read_parquet(path) if path.exists() else None


def load_transcript(out_dir: Path, patient_id: str) -> dict[str, Any] | None:
    path = out_dir / "audio" / patient_id / "transcript.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_notes(out_dir: Path, patient_id: str) -> list[dict[str, Any]]:
    """Notas de evolução anotadas pelo I2 (notes.json), oficiais e enviadas, por horário."""
    notes: list[dict[str, Any]] = []
    for path in _module_files(out_dir, f"text/{patient_id}/notes.json"):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for note in doc.get("notes", []):
            notes.append({**note, "mock": bool(doc.get("mock")), "source_file": path})
    notes.sort(key=lambda n: n.get("timestamp_start", ""))
    return notes


def load_anomaly_series(out_dir: Path, patient_id: str) -> pd.DataFrame | None:
    """Séries com anomalias marcadas do I3 (series_flagged.parquet), se existirem."""
    path = out_dir / "anomaly" / patient_id / "series_flagged.parquet"
    return pd.read_parquet(path) if path.exists() else None


def resolve_artifact(ref: str, out_dir: Path) -> Path | None:
    """Acha no disco um caminho de ``Finding.artifacts``.

    Os módulos gravam caminhos relativos à raiz do repositório (``outputs/...``);
    quando a pasta de saídas é outra (TC_OUTPUTS_DIR), troca ``outputs/`` por ela.
    """
    p = Path(ref)
    candidates = [p] if p.is_absolute() else [REPO_ROOT / p]
    if not p.is_absolute() and p.parts and p.parts[0] == "outputs":
        candidates.append(out_dir.joinpath(*p.parts[1:]))
    for c in candidates:
        if c.is_file():
            return c
    return None


IMAGE_EXT = {".jpg", ".jpeg", ".png"}
VIDEO_EXT = {".mp4", ".webm"}


def video_media(findings: list[Finding], out_dir: Path) -> list[dict[str, Any]]:
    """Quadros e clipes anotados ligados a cada Finding de vídeo (I1)."""
    rows = []
    for f in findings:
        if f.source != "video":
            continue
        files = [resolve_artifact(a, out_dir) for a in f.artifacts]
        rows.append(
            {
                "finding": f,
                "frames": [x for x in files if x is not None and x.suffix.lower() in IMAGE_EXT],
                "clips": [x for x in files if x is not None and x.suffix.lower() in VIDEO_EXT],
            }
        )
    return rows


def video_frame_dirs(out_dir: Path, patient_id: str) -> dict[str, list[Path]]:
    """Quadros anotados por vídeo (frames/<video>/*.jpg), inclusive do leito, que não gera Findings."""
    result: dict[str, list[Path]] = {}
    for base in [out_dir / "video" / patient_id, *sorted(out_dir.glob(f"uploads/*/video/{patient_id}"))]:
        frames = base / "frames"
        if not frames.is_dir():
            continue
        for d in sorted(p for p in frames.iterdir() if p.is_dir()):
            imgs = sorted(x for x in d.iterdir() if x.suffix.lower() in IMAGE_EXT)
            if imgs:
                result.setdefault(d.name, []).extend(imgs)
    return result


def video_reports(out_dir: Path, patient_id: str) -> list[Path]:
    """Relatórios HTML automáticos do I1 (oficial e dos envios)."""
    return _module_files(out_dir, f"video/{patient_id}/report.html")


def load_transcripts(out_dir: Path, patient_id: str) -> list[tuple[Path, dict[str, Any]]]:
    """Transcrições do I2: a oficial e as dos áudios enviados pelo Streamlit."""
    items = []
    for path in _module_files(out_dir, f"audio/{patient_id}/transcript.json"):
        try:
            items.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            continue
    return items
