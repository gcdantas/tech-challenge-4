"""Dashboard: quadros e clipes do I1, notas do I2 e séries do I3 (dados sintéticos)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from app.dashboard import highlight_terms, note_changes, series_layout
from app.loaders import load_anomaly_series, load_findings, resolve_artifact, video_frame_dirs, video_media
from contracts.finding import read_findings, write_findings

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")

def _png() -> bytes:
    """Imagem SINTÉTICA de 8x8 px, só para o st.image ter o que mostrar."""
    import cv2

    ok, buf = cv2.imencode(".png", np.full((8, 8, 3), 128, np.uint8))
    assert ok
    return buf.tobytes()


PNG_1PX = _png()


# ------------------------------------------------------------------ funções puras


def test_destaque_usa_offset_do_integrante_2():
    text = "Refere falta de ar ao levantar."
    md = highlight_terms(text, [{"term": "falta de ar", "offset": 7, "length": 11}])
    assert md == "Refere **:red[falta de ar]** ao levantar."


def test_destaque_sem_offset_procura_o_termo_e_escapa_markdown():
    md = highlight_terms("Dor *forte*; dispneia.", [{"term": "dispneia"}])
    assert md == "Dor \\*forte\\*; **:red[dispneia]**."
    assert highlight_terms("nada", [{"term": "ausente"}]) == "nada"


def test_destaque_com_offset_errado_cai_na_busca():
    md = highlight_terms("abc sem ar", [{"term": "sem ar", "offset": 0, "length": 6}])
    assert md == "abc **:red[sem ar]**"


def test_series_layout_reconhece_tempo_valores_e_marcas():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-11-03 10:00", periods=3, freq="min", tz="-03:00"),
            "spo2": [97, 88, 96],
            "fc": [80, 110, 82],
            "is_anomaly": [False, True, False],
            "in_bed": [True, True, False],
        }
    )
    assert series_layout(df) == ("timestamp", ["spo2", "fc"], ["is_anomaly"])


def test_resumo_da_comparacao_das_notas():
    assert "Primeira nota" in note_changes({"comparison": {"previous_note": None}})
    msg = note_changes(
        {"comparison": {"previous_note": "nota_anterior", "added_terms": ["dispneia"], "persisting_terms": [],
                        "removed_terms": ["tosse"]}}
    )
    assert "nota_anterior" in msg and "novos: dispneia" in msg and "saíram: tosse" in msg


# ------------------------------------------------------------------ loaders


def _with_video_media(mocks_dir: Path, out: Path) -> Path:
    """Copia os mocks e liga quadro e clipe ao primeiro Finding de vídeo."""
    shutil.copytree(mocks_dir, out)
    vdir = out / "video" / "P001"
    frame = vdir / "frames" / "fisio" / "desvio_01_amplitude_incompleta.png"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(PNG_1PX)
    (vdir / "frames" / "leito").mkdir()
    (vdir / "frames" / "leito" / "amostra_01.png").write_bytes(PNG_1PX)
    (vdir / "report.html").write_text("<html>relatório</html>", encoding="utf-8")
    path = vdir / "findings.jsonl"
    fs = read_findings(path)
    fs[0] = fs[0].model_copy(update={"artifacts": ["outputs/video/P001/frames/fisio/desvio_01_amplitude_incompleta.png"]})
    write_findings(path, fs)
    return out


def test_caminho_outputs_e_remapeado_para_a_pasta_de_saidas(tmp_path):
    f = tmp_path / "video" / "P001" / "x.png"
    f.parent.mkdir(parents=True)
    f.write_bytes(PNG_1PX)
    assert resolve_artifact("outputs/video/P001/x.png", tmp_path) == f
    assert resolve_artifact("outputs/video/P001/nao_existe.png", tmp_path) is None


def test_quadros_ligados_aos_findings_e_por_video(mocks_dir, tmp_path):
    out = _with_video_media(mocks_dir, tmp_path / "out")
    media = video_media(load_findings(out, "P001").findings, out)
    assert [len(m["frames"]) for m in media] == [1, 0]
    assert set(video_frame_dirs(out, "P001")) == {"fisio", "leito"}


def test_serie_do_integrante_3_ausente_e_presente(tmp_path):
    assert load_anomaly_series(tmp_path, "P001") is None
    d = tmp_path / "anomaly" / "P001"
    d.mkdir(parents=True)
    pd.DataFrame({"timestamp": pd.date_range("2026-11-03 10:00", periods=2, freq="min"), "spo2": [97, 88]}).to_parquet(
        d / "series_flagged.parquet"
    )
    assert list(load_anomaly_series(tmp_path, "P001")["spo2"]) == [97, 88]


# ------------------------------------------------------------------ app completo


def test_abas_com_video_notas_e_series(mocks_dir, tmp_path, monkeypatch):
    out = _with_video_media(mocks_dir, tmp_path / "out")
    tdir = out / "text" / "P001"
    (tdir / "notes.json").write_text(
        json.dumps(
            {
                "mock": True,
                "notes": [
                    {"note": "nota_recente", "text": "Apresenta dispneia aos esforços.", "synthetic": True,
                     "timestamp_start": "2026-11-03T10:40:00-03:00",
                     "critical_terms": [{"term": "dispneia", "offset": 10, "length": 8}],
                     "comparison": {"previous_note": "nota_anterior", "added_terms": ["dispneia"]}}
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    adir = out / "anomaly" / "P001"
    n = 20
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-11-03 10:30", periods=n, freq="min", tz="-03:00"),
            "spo2": np.r_[np.full(15, 97), np.full(5, 87)],
            "is_anomaly": [False] * 15 + [True] * 5,
        }
    ).to_parquet(adir / "series_flagged.parquet")

    monkeypatch.setenv("TC_OUTPUTS_DIR", str(out))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    labels = [t.label for t in at.tabs]
    assert labels == ["Vídeo (Integrante 1)", "Áudio e notas (Integrante 2)", "Anomalias (Integrante 3)", "Enviar arquivo"]
    md = " ".join(m.value for m in at.markdown)
    assert "**:red[dispneia]**" in md
    assert any("5 ponto(s) marcado(s)" in c.value for c in at.caption)
    assert any("nota_anterior" in c.value for c in at.caption)
    assert {s.label for s in at.selectbox} == {"Paciente", "Vídeo", "Série", "Artefato"}


def test_aba_de_envio_lista_os_envios(mocks_dir, tmp_path, monkeypatch):
    from app.ingest import analyze_upload
    from contracts.patients import load_patients
    from tests.test_ingest import FakeVideo

    out = tmp_path / "out"
    shutil.copytree(mocks_dir, out)
    analyze_upload(b"v", "fisio_nova.mp4", "P001", "video_fisio", out, load_patients(), video_analyze=FakeVideo())
    monkeypatch.setenv("TC_OUTPUTS_DIR", str(out))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("fisio\\_nova.mp4" in m.value or "fisio_nova.mp4" in m.value for m in at.markdown)
    assert any(b.label == "Remover" for b in at.button)
    at.button(key=[b.key for b in at.button if b.label == "Remover"][0]).click().run()
    assert not at.exception, at.exception
    assert not list((out / "uploads").glob("*/upload.json"))
