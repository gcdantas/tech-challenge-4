"""Testes do detector de sinais vitais em fluxo."""
from datetime import datetime, timedelta

import pytest

from contracts.finding import DEMO_TZ, Finding
from modules.anomaly.config import InjecaoCfg, carregar_config, carregar_config_simulador
from modules.anomaly.simulator import gerar_cenario_vitais, gerar_vitais
from modules.anomaly.vitals import DetectorVitais, PontoVital, analisar_quadro, pontos_do_quadro

INICIO = datetime(2026, 11, 3, 8, 30, tzinfo=DEMO_TZ)


@pytest.fixture(scope="module")
def cfg():
    return carregar_config()


@pytest.fixture(scope="module")
def sim():
    return carregar_config_simulador()


def com_injecoes(sim, *injecoes, paciente="P001"):
    return gerar_vitais(sim, paciente, INICIO, injecoes=list(injecoes))


def test_paciente_estavel_nao_gera_nenhum_finding(cfg, sim):
    for i in range(100):
        quadro = gerar_vitais(sim, f"T{i:03d}", INICIO, variar_paciente=True)
        assert analisar_quadro(cfg, quadro) == [], f"falso alarme no paciente T{i:03d}"


def test_dessaturacao_do_p001_gera_um_finding_so(cfg, sim):
    quadro, [rotulo] = gerar_cenario_vitais(sim, "p001_demo", "P001", INICIO)
    findings = analisar_quadro(cfg, quadro)

    assert len(findings) == 1  # a FC e a FR sobem junto, mas é o mesmo evento
    f = findings[0]
    assert (f.source, f.type, f.severity) == ("vitals", "dessaturacao", "high")
    assert rotulo.inicio <= f.timestamp_start and f.timestamp_end <= rotulo.fim
    assert (f.timestamp_end - rotulo.inicio) < timedelta(seconds=75)  # atraso de detecção
    assert f.evidence["sinal"] == "spo2" and f.evidence["limite_cruzado"] == 90


def test_distratores_nao_geram_finding(cfg, sim):
    quadro, _ = gerar_cenario_vitais(sim, "distratores_demo", "P001", INICIO)
    assert analisar_quadro(cfg, quadro) == []


@pytest.mark.parametrize(
    "injecao, tipo_esperado",
    [
        (InjecaoCfg(tipo="taquicardia", inicio_min=30, duracao_min=5, magnitude=50, rampa_s=30), "taquicardia"),
        (InjecaoCfg(tipo="hipotensao", inicio_min=30, duracao_min=5, magnitude=30, rampa_s=30), "hipotensao"),
        (InjecaoCfg(tipo="febre", inicio_min=30, duracao_min=5, magnitude=1.5, rampa_s=60), "febre"),
        (InjecaoCfg(tipo="taquipneia", inicio_min=30, duracao_min=5, magnitude=10, rampa_s=30), "taquipneia"),
    ],
)
def test_cada_anomalia_vira_o_tipo_certo(cfg, sim, injecao, tipo_esperado):
    findings = analisar_quadro(cfg, com_injecoes(sim, injecao))
    assert tipo_esperado in [f.type for f in findings]
    assert all(f.timestamp_start >= INICIO + timedelta(minutes=30) for f in findings)


def test_desvio_so_estatistico_e_marcado_como_tal(cfg, sim):
    # FC de 76 para 96: não passa de nenhum limite clínico, mas foge muito do padrão do paciente
    quadro = com_injecoes(sim, InjecaoCfg(tipo="taquicardia", inicio_min=30, duracao_min=5, magnitude=20, rampa_s=30))
    [f] = analisar_quadro(cfg, quadro)
    assert f.type == "desvio_fc"
    assert f.evidence["estatistico_apenas"] is True and f.evidence["limite_cruzado"] is None
    assert f.evidence["mediana_base"] == pytest.approx(76, abs=3)


def test_piora_gera_novo_finding_com_severidade_maior(cfg, sim):
    # queda lenta (2 min) e funda: passa por high e termina em critical
    quadro = com_injecoes(sim, InjecaoCfg(tipo="dessaturacao", inicio_min=30, duracao_min=5, magnitude=14, rampa_s=120))
    desat = [f for f in analisar_quadro(cfg, quadro) if f.type == "dessaturacao"]

    assert [f.severity for f in desat] == ["high", "critical"]
    assert desat[0].score < desat[1].score
    assert {f.evidence["episodio_id"] for f in desat} == {"spo2-1"}  # é o mesmo episódio
    assert [f.evidence["evento"] for f in desat] == ["confirmado", "escalonado"]


def test_perda_de_sinal_no_meio_encerra_o_episodio(cfg, sim):
    quadro = com_injecoes(
        sim,
        InjecaoCfg(tipo="dessaturacao", inicio_min=30, duracao_min=10, magnitude=10, rampa_s=20),
        InjecaoCfg(tipo="perda_sinal", inicio_min=33, duracao_min=2),
    )
    desat = [f for f in analisar_quadro(cfg, quadro) if f.type == "dessaturacao"]
    # o sinal voltou e a dessaturação continua: é um episódio novo
    assert [f.evidence["episodio_id"] for f in desat] == ["spo2-1", "spo2-2"]
    assert desat[1].timestamp_start >= INICIO + timedelta(minutes=35)


def test_anomalia_nao_contamina_a_linha_de_base(cfg, sim):
    # a dessaturação começa no minuto 20; no minuto 24 o SpO2 já está há 4 min em ~89 %
    quadro = com_injecoes(sim, InjecaoCfg(tipo="dessaturacao", inicio_min=20, duracao_min=20, magnitude=8, rampa_s=10))
    detector = DetectorVitais(cfg, "P001")
    for ponto in pontos_do_quadro(quadro):
        if ponto.timestamp >= INICIO + timedelta(minutes=24):
            break
        detector.update(ponto)

    linha = detector.linhas["spo2"]
    assert linha.pronta()
    assert linha.mediana_e_escala()[0] >= 96  # a mediana continua sendo a do paciente estável


def test_findings_seguem_o_contrato(cfg, sim):
    quadro, _ = gerar_cenario_vitais(sim, "p001_demo", "P001", INICIO)
    for f in analisar_quadro(cfg, quadro):
        assert Finding.model_validate_json(f.to_json_line()) == f
        assert f.timestamp_start.utcoffset() == timedelta(hours=-3)
        assert 0 <= f.score <= 1 and f.model.name and f.description


def test_leitura_fora_da_faixa_fisica_e_ignorada_e_contada(cfg):
    detector = DetectorVitais(cfg, "P001")
    normal = {"fc": 76, "spo2": 97, "fr": 16, "pas": 122, "pad": 76, "temp": 36.7}
    for k in range(10):
        valores = dict(normal, spo2=0 if k >= 5 else 97)
        assert detector.update(PontoVital(INICIO + timedelta(seconds=10 * k), "P001", valores)) == []
    assert detector.descartes["spo2_fora_da_faixa_fisica"] == 5


def test_timestamp_repetido_ou_atrasado_e_descartado(cfg):
    detector = DetectorVitais(cfg, "P001")
    normal = {"fc": 76, "spo2": 97, "fr": 16, "pas": 122, "pad": 76, "temp": 36.7}
    detector.update(PontoVital(INICIO + timedelta(seconds=10), "P001", normal))
    assert detector.update(PontoVital(INICIO + timedelta(seconds=10), "P001", normal)) == []
    assert detector.update(PontoVital(INICIO, "P001", normal)) == []
    assert detector.descartes["fora_de_ordem"] == 2


def test_paciente_errado_e_fuso_ausente_dao_erro(cfg):
    detector = DetectorVitais(cfg, "P001")
    normal = {"fc": 76, "spo2": 97, "fr": 16, "pas": 122, "pad": 76, "temp": 36.7}
    with pytest.raises(ValueError):
        detector.update(PontoVital(INICIO, "P002", normal))
    with pytest.raises(ValueError):
        detector.update(PontoVital(datetime(2026, 11, 3, 8, 30), "P001", normal))