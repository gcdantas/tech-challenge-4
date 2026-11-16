"""P001 com o módulo REAL de anomalias (Integrante 3) no lugar dos mocks dele.

Vídeo, áudio e texto continuam mock; vitais, prescrições e movimentação vêm do
simulador e dos detectores do I3 (cenário p001_integrado). Confere o contrato
entre o I3 e a fusão: a regra crítica dispara com a dessaturação real e o
series_flagged.parquet é lido pelo dashboard.
"""

from __future__ import annotations

import shutil

import pandas as pd
import pytest

from app.dashboard import series_long
from app.fusion import fuse, load_fusion_config
from app.loaders import load_anomaly_series, load_findings


@pytest.fixture(scope="module")
def p001_i3(mocks_dir, tmp_path_factory):
    from modules.anomaly.config import carregar_config, carregar_config_simulador
    from modules.anomaly.pipeline import executar, preparar_series, salvar

    out = tmp_path_factory.mktemp("integ") / "outputs"
    shutil.copytree(mocks_dir, out)
    shutil.rmtree(out / "anomaly")
    series = preparar_series(
        "P001", str(out / "video" / "P001" / "movement.parquet"), cfg_sim=carregar_config_simulador()
    )
    findings, monitor = executar(carregar_config(), series)
    salvar(series, findings, monitor, out / "anomaly" / "P001")
    return out


def test_regra_critica_dispara_com_a_dessaturacao_do_i3(p001_i3):
    loaded = load_findings(p001_i3, "P001")
    assert not loaded.errors
    by_id = {str(f.finding_id): f for f in loaded.findings}
    anomaly = [f for f in loaded.findings if f.source in ("vitals", "prescription", "movement")]
    assert all(not f.evidence.get("mock") or f.source == "movement" for f in anomaly)

    alerts = fuse(loaded.findings, load_fusion_config())
    (crit,) = [a for a in alerts if a.level == "critico"]
    assert crit.rule == "respiratorio_vocal_spo2"
    spo2 = [by_id[i] for i in crit.finding_ids if by_id[i].source == "vitals"]
    assert [f.type for f in spo2] == ["dessaturacao"]
    assert spo2[0].evidence.get("episodio_id")  # veio do detector do I3, não do mock
    # Todo Finding do I3 entra em algum alerta (emitido ou deduplicado).
    covered = {i for a in alerts for i in [*a.finding_ids, *a.suppressed_finding_ids]}
    assert {str(f.finding_id) for f in anomaly} <= covered


def test_series_flagged_do_i3_e_lida_pelo_dashboard(p001_i3):
    raw = load_anomaly_series(p001_i3, "P001")
    long = series_long(raw)
    assert long is not None and not long.empty
    spo2 = long[long["serie"] == "vitals · spo2"]
    assert spo2["flagged"].any()
    assert pd.api.types.is_datetime64_any_dtype(long["timestamp"])
