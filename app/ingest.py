"""Upload de mídia pelo Streamlit (dono: I4).

O I4 não analisa mídia: recebe o arquivo, chama a interface do módulo dono e
guarda os Findings para a fusão.

- Vídeo (fisioterapia ou leito): ``modules.video.analyze()`` do Integrante 1.
- Áudio (WAV) e nota (TXT): o pipeline do Integrante 2 (``AudioPipeline``, o
  mesmo que ``modules.audio.pipeline.analyze()`` usa). O ``analyze()`` dele só
  aceita arquivos já listados no patients.yaml; aqui o arquivo enviado é ligado
  ao artefato escolhido pelo ``artifact_name``, que define o ``demo_offset``.

Cada envio fica em ``outputs/uploads/<upload_id>/``, com a mesma estrutura das
saídas dos módulos (``<modulo>/<paciente>/findings.jsonl``), e nunca sobrescreve
as saídas oficiais em ``outputs/<modulo>/``. O ``upload_id`` depende do
conteúdo, do paciente e do artefato: reenviar o mesmo arquivo reaproveita a pasta.

O laço com o simulador do Integrante 3 ainda não existe (o simulador não foi
entregue); os Findings dele continuam vindo de ``outputs/anomaly/``.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from contracts.finding import Finding, write_findings
from contracts.patients import PatientsFile

BRT = timezone(timedelta(hours=-3))

# kind do patients.yaml -> pasta do módulo em outputs/ e extensões aceitas.
UPLOAD_KINDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "video_physio": ("video", ("mp4", "avi", "mov", "mkv")),
    "video_bed": ("video", ("mp4", "avi", "mov", "mkv")),
    "audio": ("audio", ("wav",)),
    "note": ("text", ("txt",)),
}
KIND_LABEL = {
    "video_physio": "vídeo de fisioterapia (I1)",
    "video_bed": "vídeo do leito (I1)",
    "audio": "áudio da consulta, WAV PCM 16 bits mono 16 kHz (I2)",
    "note": "nota de evolução, TXT (I2)",
}
META_FILE = "upload.json"


@dataclass
class UploadResult:
    upload_id: str
    patient_id: str
    artifact: str
    kind: str
    filename: str
    folder: Path
    findings: list[Finding] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def uploads_dir(out_dir: Path) -> Path:
    return out_dir / "uploads"


def uploadable_artifacts(pf: PatientsFile, patient_id: str) -> dict[str, str]:
    """Artefatos do paciente que aceitam upload: nome -> kind."""
    return {n: a.kind for n, a in pf.patient(patient_id).artifacts.items() if a.kind in UPLOAD_KINDS}


def upload_id_for(data: bytes, patient_id: str, artifact: str) -> str:
    h = hashlib.sha256()
    h.update(f"{patient_id}|{artifact}|".encode())
    h.update(data)
    return h.hexdigest()[:12]


def _safe_name(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^\w.\-]+", "_", name, flags=re.UNICODE).strip("._")
    return name or "arquivo"


def patients_with_upload(pf: PatientsFile, patient_id: str, artifact: str, path: Path) -> PatientsFile:
    """Cópia do patients.yaml em que o artefato aponta para o arquivo enviado."""
    copy = pf.model_copy(deep=True)
    copy.patient(patient_id).artifacts[artifact].path = str(path.resolve())
    return copy


def _default_video_analyze() -> Callable[..., list[Finding]]:
    from modules.video import analyze

    return analyze


def _default_audio_pipeline(out_dir: Path, allow_azure: bool):
    from modules.audio.pipeline import AudioPipeline

    # outputs=out_dir: o cache das respostas Azure é o mesmo das saídas oficiais.
    return AudioPipeline(allow_azure=allow_azure, outputs=out_dir)


def analyze_upload(
    data: bytes,
    filename: str,
    patient_id: str,
    artifact: str,
    out_dir: Path,
    pf: PatientsFile,
    *,
    allow_azure: bool = False,
    video_analyze: Callable[..., list[Finding]] | None = None,
    audio_pipeline: Any | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(BRT),
) -> UploadResult:
    """Grava o arquivo, chama o módulo dono e salva os Findings do envio.

    Erros do módulo (pesos ausentes, WAV fora do formato, Azure não autorizado)
    não derrubam a tela: voltam em ``UploadResult.error`` e ficam no upload.json.
    """
    kind = pf.artifact(patient_id, artifact).kind
    if kind not in UPLOAD_KINDS:
        raise ValueError(f"o artefato {artifact!r} é {kind!r}, que não aceita upload")
    module, exts = UPLOAD_KINDS[kind]
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in exts:
        raise ValueError(f"{filename}: extensão .{ext} não aceita para {kind} (use {', '.join(exts)})")
    if not data:
        raise ValueError(f"{filename}: arquivo vazio")

    uid = upload_id_for(data, patient_id, artifact)
    folder = uploads_dir(out_dir) / uid
    input_path = folder / "input" / _safe_name(filename)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(data)

    result = UploadResult(uid, patient_id, artifact, kind, filename, folder)
    findings_path = folder / module / patient_id / "findings.jsonl"
    try:
        if module == "video":
            analyze = video_analyze or _default_video_analyze()
            pf_up = patients_with_upload(pf, patient_id, artifact, input_path)
            result.findings = list(analyze(input_path, patient_id, patients=pf_up, outputs_root=folder))
            write_findings(findings_path, result.findings)
        else:
            pipe = audio_pipeline or _default_audio_pipeline(out_dir, allow_azure)
            if kind == "audio":
                res = pipe.audio(input_path, patient_id, artifact_name=artifact)
                document = "transcript.json"
            else:
                res = pipe.notes([input_path], patient_id, artifact_name=artifact)
                document = "notes.json"
            pipe.save(res, findings_path, document)
            result.findings = list(res.findings)
    except Exception as exc:  # noqa: BLE001 - a tela mostra o erro do módulo sem cair
        result.error = f"{type(exc).__name__}: {exc}"

    meta = {
        "upload_id": uid,
        "patient_id": patient_id,
        "artifact": artifact,
        "kind": kind,
        "filename": filename,
        "input": input_path.relative_to(folder).as_posix(),
        "created_at": now().isoformat(),
        "findings": len(result.findings),
        "error": result.error,
    }
    (folder / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def list_uploads(out_dir: Path, patient_id: str | None = None) -> list[dict[str, Any]]:
    """Envios gravados (mais recentes primeiro), lidos do upload.json de cada pasta."""
    items = []
    root = uploads_dir(out_dir)
    for meta_path in sorted(root.glob(f"*/{META_FILE}")) if root.exists() else []:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if patient_id is None or meta.get("patient_id") == patient_id:
            meta["folder"] = meta_path.parent
            items.append(meta)
    items.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return items


def remove_upload(out_dir: Path, upload_id: str) -> bool:
    """Apaga a pasta de um envio (os Findings dele saem da fusão)."""
    if not re.fullmatch(r"[0-9a-f]{12}", upload_id):
        raise ValueError(f"upload_id inválido: {upload_id!r}")
    folder = uploads_dir(out_dir) / upload_id
    if not folder.is_dir():
        return False
    shutil.rmtree(folder)
    return True
