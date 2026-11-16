"""Upload pelo Streamlit: grava o envio, chama o módulo dono e alimenta a fusão.

Nada aqui roda YOLOv8 nem Azure: o vídeo usa um analyze falso e as notas usam o
modo mock do Integrante 2 (MockLanguage, determinístico).
"""

from __future__ import annotations

import json
import shutil
from datetime import timedelta

import pytest

from app.fusion import fuse, load_fusion_config
from app.ingest import (
    analyze_upload,
    list_uploads,
    remove_upload,
    upload_id_for,
    uploadable_artifacts,
)
from app.loaders import load_findings, load_notes
from contracts.finding import Finding, ModelInfo, read_findings
from contracts.patients import REPO_ROOT, load_patients

NOTE = REPO_ROOT / "data" / "notes" / "P001_nota_2.txt"


@pytest.fixture
def pf():
    return load_patients()


class FakeVideo:
    """Faz o papel do modules.video.analyze: registra a chamada e devolve 1 Finding."""

    def __init__(self, fail: Exception | None = None):
        self.calls = []
        self.fail = fail

    def __call__(self, input, patient_id, *, patients, outputs_root):  # noqa: A002
        self.calls.append((input, patient_id, patients, outputs_root))
        if self.fail:
            raise self.fail
        name, art = patients.find_artifact_by_path(patient_id, input)
        start = patients.demo_time(art.demo_offset, 5.0)
        return [
            Finding(
                patient_id=patient_id, source="video", type="amplitude_incompleta",
                timestamp_start=start, timestamp_end=start + timedelta(seconds=2),
                score=0.6, severity="medium", description=f"teste {name}",
                model=ModelInfo(name="fake", version="0"),
            )
        ]


def test_artefatos_com_upload_do_p001(pf):
    assert uploadable_artifacts(pf, "P001") == {
        "video_fisio": "video_physio",
        "video_leito": "video_bed",
        "consulta": "audio",
        "nota_anterior": "note",
        "nota_recente": "note",
    }


def test_upload_de_video_chama_o_analyze_com_o_offset_do_artefato(tmp_path, pf):
    fake = FakeVideo()
    res = analyze_upload(b"conteudo-mp4", "meu video.mp4", "P001", "video_leito", tmp_path, pf, video_analyze=fake)
    assert res.ok and len(res.findings) == 1

    (input_path, pid, patients, root), = fake.calls
    assert pid == "P001" and root == res.folder
    assert input_path.read_bytes() == b"conteudo-mp4"
    assert input_path.name == "meu_video.mp4"
    # O vídeo enviado herda o demo_offset do artefato escolhido (leito: 600 s).
    assert res.findings[0].timestamp_start == pf.demo_time(600, 5.0)
    # O patients.yaml original não muda.
    assert pf.artifact("P001", "video_leito").path == "data/video/charades/SSP8C.mp4"

    saved = res.folder / "video" / "P001" / "findings.jsonl"
    assert read_findings(saved) == res.findings
    # Entra no loader e na fusão junto com as saídas oficiais.
    loaded = load_findings(tmp_path, "P001")
    assert [f.finding_id for f in loaded.findings] == [res.findings[0].finding_id]
    assert fuse(loaded.findings, load_fusion_config())


def test_upload_de_nota_usa_o_pipeline_do_integrante_2(tmp_path, pf):
    from modules.audio.pipeline import AudioPipeline

    pipe = AudioPipeline(mock=True, outputs=tmp_path)
    res = analyze_upload(NOTE.read_bytes(), NOTE.name, "P001", "nota_recente", tmp_path, pf, audio_pipeline=pipe)
    assert res.ok, res.error
    assert res.findings and all(f.source == "text" for f in res.findings)
    when = pf.artifact_time("P001", "nota_recente")
    assert {f.timestamp_start for f in res.findings} == {when}
    assert "termo_critico" in {f.type for f in res.findings}

    assert read_findings(res.folder / "text" / "P001" / "findings.jsonl")
    (note,) = load_notes(tmp_path, "P001")
    assert note["note"] == "nota_recente" and note["critical_terms"]


def test_erro_do_modulo_nao_derruba_e_fica_registrado(tmp_path, pf):
    fake = FakeVideo(fail=FileNotFoundError("pesos YOLOv8 não baixados"))
    res = analyze_upload(b"x", "a.mp4", "P001", "video_fisio", tmp_path, pf, video_analyze=fake)
    assert not res.ok and "pesos YOLOv8" in res.error
    (meta,) = list_uploads(tmp_path, "P001")
    assert meta["findings"] == 0 and "pesos YOLOv8" in meta["error"]
    assert not load_findings(tmp_path, "P001").findings


@pytest.mark.parametrize(
    ("filename", "artifact", "msg"),
    [
        ("a.wav", "video_fisio", "extensão"),
        ("a.mp4", "consulta", "extensão"),
        ("a.txt", "vitais", "não aceita upload"),
    ],
)
def test_entradas_recusadas(tmp_path, pf, filename, artifact, msg):
    with pytest.raises(ValueError, match=msg):
        analyze_upload(b"x", filename, "P001", artifact, tmp_path, pf, video_analyze=FakeVideo())
    assert not (tmp_path / "uploads").exists()


def test_arquivo_vazio_recusado(tmp_path, pf):
    with pytest.raises(ValueError, match="vazio"):
        analyze_upload(b"", "a.mp4", "P001", "video_fisio", tmp_path, pf, video_analyze=FakeVideo())


def test_mesmo_arquivo_reaproveita_a_pasta_e_remover_tira_da_fusao(tmp_path, pf):
    a = analyze_upload(b"v", "a.mp4", "P001", "video_fisio", tmp_path, pf, video_analyze=FakeVideo())
    b = analyze_upload(b"v", "b.mp4", "P001", "video_fisio", tmp_path, pf, video_analyze=FakeVideo())
    assert a.upload_id == b.upload_id == upload_id_for(b"v", "P001", "video_fisio")
    assert len(list_uploads(tmp_path)) == 1
    # O mesmo arquivo como outro artefato é outro envio.
    c = analyze_upload(b"v", "a.mp4", "P001", "video_leito", tmp_path, pf, video_analyze=FakeVideo())
    assert c.upload_id != a.upload_id

    assert remove_upload(tmp_path, a.upload_id)
    assert not remove_upload(tmp_path, a.upload_id)
    assert [m["upload_id"] for m in list_uploads(tmp_path, "P001")] == [c.upload_id]
    with pytest.raises(ValueError):
        remove_upload(tmp_path, "../..")


def test_list_uploads_ignora_pasta_corrompida(tmp_path, pf):
    analyze_upload(b"v", "a.mp4", "P001", "video_fisio", tmp_path, pf, video_analyze=FakeVideo())
    bad = tmp_path / "uploads" / "0123456789ab"
    bad.mkdir()
    (bad / "upload.json").write_text("{não é json", encoding="utf-8")
    assert len(list_uploads(tmp_path)) == 1


def test_upload_soma_as_saidas_oficiais(mocks_dir, tmp_path, pf):
    out = tmp_path / "outputs"
    shutil.copytree(mocks_dir, out)
    before = len(load_findings(out, "P001").findings)
    analyze_upload(b"v", "a.mp4", "P001", "video_fisio", out, pf, video_analyze=FakeVideo())
    assert len(load_findings(out, "P001").findings) == before + 1
    # As saídas oficiais não foram tocadas.
    assert json.loads((out / "audio" / "P001" / "transcript.json").read_text(encoding="utf-8"))["mock"] is True
