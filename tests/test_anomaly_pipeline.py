"""Testes do monitor, do pipeline (analyze), do fluxo e da linha de comando."""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from contracts.finding import DEMO_TZ, Finding, read_findings, write_findings
from modules.anomaly.config import SimuladorCfg, carregar_config, carregar_config_simulador
from modules.anomaly.fluxo import intercalar, reproduzir
from modules.anomaly.monitor import MonitorPaciente
from modules.anomaly.movement import COLUNAS, PontoMovimento
from modules.anomaly.pipeline import analyze, eventos_da_serie, executar, preparar_series, series_marcadas
from modules.anomaly.prescription import EventoPrescricao
from modules.anomaly.vitals import PontoVital

REFERENCIA = datetime(2026, 11, 3, 10, 0, tzinfo=DEMO_TZ)

PATIENTS_YAML = """
demo_reference: "2026-11-03T10:00:00-03:00"
patients:
  P001:
    artifacts:
      vitais: {kind: vitals, path: null, demo_offset: 0}
      prescricoes: {kind: prescriptions, path: null, demo_offset: 0}
  P002:
    artifacts:
      vitais: {kind: vitals, path: null, demo_offset: 600}
  P003:
    artifacts: {}
"""


@pytest.fixture(scope="module")
def cfg():
    return carregar_config()


@pytest.fixture(scope="module")
def sim():
    return carregar_config_simulador()


@pytest.fixture(scope="module")
def patients(tmp_path_factory):
    caminho = tmp_path_factory.mktemp("patients") / "patients.yaml"
    caminho.write_text(PATIENTS_YAML, encoding="utf-8")
    return caminho


def clipe_do_leito(patient_id="P001", inicio=REFERENCIA + timedelta(minutes=10), saida_s=18.0, duracao_s=30, hz=5):
    """Um clipe como o do vídeo SSP8C: 30 s a 5 Hz, a pessoa sai do leito aos 18 s."""
    n = duracao_s * hz
    segundos = np.arange(n) / hz
    deitada = segundos < saida_s
    return pd.DataFrame(
        {
            "patient_id": patient_id,
            "timestamp": [pd.Timestamp(inicio) + pd.Timedelta(milliseconds=int(k * 1000 / hz)) for k in range(n)],
            "centroid_x": np.where(deitada, 0.5, 0.9),
            "centroid_y": 0.55,
            "torso_angle": np.where(deitada, 83.0, 16.0),
            "velocidade": 0.0,
            "in_bed": deitada,
        },
        columns=COLUNAS,
    )


# ---------------------------------------------------------------- monitor


def test_monitor_recusa_evento_de_tipo_desconhecido(cfg):
    with pytest.raises(TypeError):
        MonitorPaciente(cfg, "P001").update("não é um evento")


def test_monitor_encaminha_cada_evento_ao_detector_certo(cfg):
    m = MonitorPaciente(cfg, "P001")
    m.update(PontoVital(REFERENCIA, "P001", {"fc": 76.0, "spo2": 97.0, "fr": 16.0, "pas": 122.0, "pad": 76.0, "temp": 36.7}))
    m.update(PontoMovimento(REFERENCIA, "P001", True, 85.0, 0.5, 0.55, 0.0))
    m.update(EventoPrescricao(REFERENCIA, "P001", "prescrever", "O1", "Dipirona", 500, 4))
    assert m.eventos_recebidos == 3
    assert m.vitais.sinais.descartes == {} and m.movimento.descartes == {} and m.prescricoes.descartes == {}
    assert "O1" in m.prescricoes.ordens


def test_suspensao_so_e_emitida_quando_o_tempo_passa_pelos_pontos_de_vitais(cfg):
    m = MonitorPaciente(cfg, "P001")
    m.update(EventoPrescricao(REFERENCIA - timedelta(hours=48), "P001", "prescrever", "O1", "Metoprolol", 50, 2))
    m.update(EventoPrescricao(REFERENCIA, "P001", "suspender", "O1"))
    ponto = lambda minutos: PontoMovimento(REFERENCIA + timedelta(minutes=minutos), "P001", True, 85.0, 0.5, 0.55, 0.0)  # noqa: E731
    assert m.update(ponto(10)) == []
    achados = m.update(ponto(31))
    assert [f.type for f in achados] == ["suspensao_abrupta"]


# ---------------------------------------------------------------- pipeline


def test_p001_de_ponta_a_ponta(cfg, sim, patients):
    findings = analyze(None, "P001", cfg=cfg, cfg_sim=sim, patients_path=patients)
    assert [(f.source, f.type) for f in findings] == [
        ("movement", "saida_do_leito"),
        ("prescription", "dose_duplicada"),
        ("vitals", "dessaturacao"),
    ]
    assert [f.severity for f in findings] == ["medium", "high", "high"]
    assert findings == sorted(findings, key=lambda f: f.timestamp_start)
    assert all(isinstance(f, Finding) and f.patient_id == "P001" for f in findings)
    # a regra crítica do grupo junta voz (10:30), nota (10:40) e queda de SpO2 na mesma janela de 30 min,
    # então a dessaturação do P001 tem que cair entre 10:30 e 10:48
    dessaturacao = findings[-1]
    assert REFERENCIA + timedelta(minutes=30) <= dessaturacao.timestamp_start <= REFERENCIA + timedelta(minutes=48)


def test_findings_viram_jsonl_e_voltam_iguais(cfg, sim, patients, tmp_path):
    findings = analyze(None, "P001", cfg=cfg, cfg_sim=sim, patients_path=patients)
    caminho = write_findings(tmp_path / "findings.jsonl", findings)
    assert read_findings(caminho) == findings


def test_paciente_sem_cenario_roda_estavel(cfg, sim, patients):
    findings = analyze(None, "P002", cfg=cfg, cfg_sim=sim, patients_path=patients)
    # limites, z-score, prescrições e movimentação não acusam nada. O Isolation Forest erra de vez em quando
    # (uns 5 % dos pacientes estáveis na calibração), então ele fica de fora desta conta
    assert [f for f in findings if f.type != "padrao_multivariado"] == []


def test_serie_comeca_no_horario_do_artefato_vitais(sim, patients):
    p2 = preparar_series("P002", cfg_sim=sim, patients_path=patients)
    assert p2.inicio == REFERENCIA + timedelta(seconds=600)
    assert p2.vitais["timestamp"].iloc[0] == p2.inicio and p2.movimento["timestamp"].iloc[0] == p2.inicio
    p3 = preparar_series("P003", cfg_sim=sim, patients_path=patients)  # sem artefato: usa a referência da demo
    assert p3.inicio == REFERENCIA


def test_paciente_desconhecido_levanta_erro(sim, patients):
    with pytest.raises(KeyError):
        preparar_series("P999", cfg_sim=sim, patients_path=patients)


def test_cenario_pode_ser_trocado_na_chamada(cfg, sim, patients):
    findings = analyze(None, "P001", cenario="distratores_demo", cfg=cfg, cfg_sim=sim, patients_path=patients)
    assert findings == []  # distratores não geram Finding
    with pytest.raises(KeyError):
        analyze(None, "P001", cenario="nao_existe", cfg=cfg, cfg_sim=sim, patients_path=patients)


def test_cenario_do_paciente_que_nao_existe_e_recusado_na_config(sim):
    dados = sim.model_dump()
    dados["pacientes"] = {"P001": "nao_existe"}
    with pytest.raises(ValueError, match="nao_existe"):
        SimuladorCfg.model_validate(dados)


# ---------------------------------------------------------------- movement.parquet real estendido


def test_clipe_real_e_estendido_com_serie_simulada(cfg, sim, patients):
    clipe = clipe_do_leito()
    series = preparar_series("P001", clipe, cfg_sim=sim, patients_path=patients)
    mov = series.movimento

    assert mov["timestamp"].iloc[0] == REFERENCIA  # começa junto com os vitais
    assert mov["timestamp"].is_monotonic_increasing and mov["timestamp"].is_unique
    assert len(mov) == 10 * 60 * 5 + len(clipe)
    # as amostras do clipe real ficam como vieram
    pd.testing.assert_frame_equal(mov.tail(len(clipe)).reset_index(drop=True), clipe, check_dtype=False)
    assert series.composicao["clipe"]["amostras"] == len(clipe)
    assert series.composicao["base_simulada"]["amostras"] == 10 * 60 * 5


def test_saida_do_leito_do_clipe_real(cfg, sim, patients):
    findings = analyze(clipe_do_leito(), "P001", cfg=cfg, cfg_sim=sim, patients_path=patients)
    [saida] = [f for f in findings if f.source == "movement"]
    assert saida.type == "saida_do_leito"
    assert saida.timestamp_start == REFERENCIA + timedelta(minutes=10, seconds=18)  # 10:10:18, como no plano
    assert saida.evidence["torso_angle"] == 16.0
    assert "mock" not in saida.evidence


def test_clipe_que_ja_comeca_com_a_serie_nao_ganha_base(sim, patients):
    clipe = clipe_do_leito(inicio=REFERENCIA)
    series = preparar_series("P001", clipe, cfg_sim=sim, patients_path=patients)
    assert len(series.movimento) == len(clipe)
    assert series.composicao["base_simulada"] is None


def test_clipe_longo_nao_e_estendido(sim, patients):
    clipe = clipe_do_leito(duracao_s=11 * 60, hz=1, saida_s=300)
    series = preparar_series("P001", clipe, cfg_sim=sim, patients_path=patients)
    assert series.composicao is None and len(series.movimento) == len(clipe)


def test_clipe_de_outro_paciente_e_recusado(sim, patients):
    with pytest.raises(ValueError, match="P009"):
        preparar_series("P001", clipe_do_leito("P009"), cfg_sim=sim, patients_path=patients)


def test_clipe_sem_fuso_e_sem_colunas_sao_recusados(sim, patients):
    sem_fuso = clipe_do_leito()
    sem_fuso["timestamp"] = sem_fuso["timestamp"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="fuso"):
        preparar_series("P001", sem_fuso, cfg_sim=sim, patients_path=patients)
    with pytest.raises(ValueError, match="faltam colunas"):
        preparar_series("P001", clipe_do_leito().drop(columns=["in_bed"]), cfg_sim=sim, patients_path=patients)


def test_arquivo_inexistente_levanta_erro_claro(sim, patients, tmp_path):
    with pytest.raises(FileNotFoundError, match="movement.parquet"):
        preparar_series("P001", tmp_path / "nao_tem.parquet", cfg_sim=sim, patients_path=patients)


# ---------------------------------------------------------------- séries marcadas


def test_series_marcadas_marcam_so_o_que_esta_dentro_de_episodio(cfg, sim, patients):
    series = preparar_series("P001", cfg_sim=sim, patients_path=patients)
    findings, monitor = executar(cfg, series)
    tabela = series_marcadas(series, findings, monitor)

    assert list(tabela.columns) == ["patient_id", "timestamp", "source", "signal", "value", "flagged", "finding_id", "severity"]
    assert set(tabela["source"]) == {"vitals", "movement"}
    assert set(tabela["signal"]) == {"fc", "spo2", "fr", "pas", "pad", "temp", "torso_angle", "centroid_x", "centroid_y", "velocidade", "in_bed"}

    marcadas = tabela[tabela["flagged"]]
    assert set(marcadas["finding_id"]) <= {str(f.finding_id) for f in findings}
    assert (marcadas["finding_id"].notna()).all() and (tabela[~tabela["flagged"]]["finding_id"].isna()).all()

    spo2 = tabela[(tabela["signal"] == "spo2") & tabela["flagged"]]
    assert not spo2.empty
    assert spo2["timestamp"].min() >= REFERENCIA + timedelta(minutes=44)  # só em volta da dessaturação (10:45)
    assert spo2["timestamp"].max() <= REFERENCIA + timedelta(minutes=50)
    # temperatura não saiu do normal
    assert tabela[(tabela["signal"] == "temp") & tabela["flagged"]].empty
    # a saída do leito marca os sinais de movimento
    assert tabela[(tabela["signal"] == "in_bed") & tabela["flagged"]]["severity"].eq("medium").all()


# ---------------------------------------------------------------- fluxo


class Relogio:
    """Relógio falso: esperar() só adianta o tempo."""

    def __init__(self):
        self.t = 0.0
        self.esperas = []

    def agora(self):
        return self.t

    def esperar(self, s):
        self.esperas.append(s)
        self.t += s


def test_intercalar_ordena_por_tempo_e_desempata_pela_ordem_das_fontes():
    a = [PontoMovimento(REFERENCIA + timedelta(seconds=s), "P001", True) for s in (0, 10, 20)]
    b = [PontoMovimento(REFERENCIA + timedelta(seconds=s), "P001", False) for s in (5, 10, 25)]
    fila = list(intercalar(a, b))
    assert [e.timestamp.second for e in fila] == [0, 5, 10, 10, 20, 25]
    assert [e.in_bed for e in fila if e.timestamp.second == 10] == [True, False]


def test_reproduzir_respeita_a_velocidade():
    eventos = [PontoMovimento(REFERENCIA + timedelta(seconds=s), "P001", True) for s in (0, 30, 60)]
    r = Relogio()
    saida = list(reproduzir(eventos, 30, REFERENCIA, esperar=r.esperar, agora=r.agora))
    assert saida == eventos
    assert sum(r.esperas) == pytest.approx(2.0)  # 60 s de série a 30x = 2 s


def test_reproduzir_nao_espera_pelo_historico_nem_com_velocidade_zero():
    historico = EventoPrescricao(REFERENCIA - timedelta(hours=72), "P001", "prescrever", "O1", "Dipirona", 500, 4)
    agora = PontoMovimento(REFERENCIA + timedelta(seconds=10), "P001", True)
    r = Relogio()
    list(reproduzir([historico, agora], 10, REFERENCIA, esperar=r.esperar, agora=r.agora))
    assert sum(r.esperas) == pytest.approx(1.0)  # só os 10 s depois da referência
    r0 = Relogio()
    list(reproduzir([historico, agora], 0, REFERENCIA, esperar=r0.esperar, agora=r0.agora))
    assert r0.esperas == []


def test_fila_da_serie_comeca_pelo_historico_das_prescricoes(sim, patients):
    series = preparar_series("P001", cfg_sim=sim, patients_path=patients)
    fila = eventos_da_serie(series)
    assert isinstance(fila[0], EventoPrescricao) and fila[0].timestamp < REFERENCIA
    assert [e.timestamp for e in fila] == sorted(e.timestamp for e in fila)


# ---------------------------------------------------------------- arquivos (precisam do pyarrow)


def test_salvar_grava_os_tres_arquivos(cfg, sim, patients, tmp_path):
    pytest.importorskip("pyarrow")
    from modules.anomaly.pipeline import salvar

    series = preparar_series("P001", clipe_do_leito(), cfg_sim=sim, patients_path=patients)
    findings, monitor = executar(cfg, series)
    caminhos = salvar(series, findings, monitor, tmp_path / "P001")

    assert read_findings(caminhos["findings"]) == findings
    tabela = pd.read_parquet(caminhos["series"])
    assert tabela["timestamp"].dt.tz is not None and tabela["flagged"].dtype == bool
    assert caminhos["composicao"].read_text(encoding="utf-8").count("base_simulada") == 1


def test_analyze_le_o_parquet_do_i1(cfg, sim, patients, tmp_path):
    pytest.importorskip("pyarrow")
    caminho = tmp_path / "movement.parquet"
    clipe_do_leito().to_parquet(caminho, index=False)
    findings = analyze(caminho, "P001", cfg=cfg, cfg_sim=sim, patients_path=patients)
    assert [f.type for f in findings if f.source == "movement"] == ["saida_do_leito"]


def test_parquet_marcado_como_mock_marca_a_evidencia(cfg, sim, patients, tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    tabela = pa.Table.from_pandas(clipe_do_leito(), preserve_index=False)
    caminho = tmp_path / "movement.parquet"
    pq.write_table(tabela.replace_schema_metadata({"mock": "true"}), caminho)
    findings = analyze(caminho, "P001", cfg=cfg, cfg_sim=sim, patients_path=patients)
    [saida] = [f for f in findings if f.source == "movement"]
    assert saida.evidence["mock"] is True


def test_linha_de_comando(tmp_path, monkeypatch, capsys, patients):
    pytest.importorskip("pyarrow")
    from modules.anomaly import pipeline, run

    original = pipeline.preparar_series
    monkeypatch.setattr(pipeline, "preparar_series", lambda *a, **k: original(*a, **k, patients_path=patients))
    monkeypatch.setattr(run, "preparar_series", pipeline.preparar_series)
    caminho = tmp_path / "movement.parquet"
    clipe_do_leito().to_parquet(caminho, index=False)
    saida = tmp_path / "saida" / "findings.jsonl"
    assert run.main(["--input", str(caminho), "--patient", "P001", "--out", str(saida)]) == 0
    assert len(read_findings(saida)) == 3
    assert (saida.parent / "series_flagged.parquet").exists()
    assert "3 Findings" in capsys.readouterr().out
