"""Testes das regras de prescrição."""
from datetime import datetime, timedelta

import pytest

from contracts.finding import DEMO_TZ, Finding
from modules.anomaly.config import carregar_config, carregar_config_simulador
from modules.anomaly.prescription import (
    DetectorPrescricoes,
    EventoPrescricao,
    analisar_eventos,
    normalizar_nome,
)
from modules.anomaly.simulator import gerar_prescricoes

T0 = datetime(2026, 11, 3, 8, 30, tzinfo=DEMO_TZ)


@pytest.fixture(scope="module")
def cfg():
    return carregar_config()


@pytest.fixture(scope="module")
def sim():
    return carregar_config_simulador()


def pedido(minutos, ordem, med, dose, vezes):
    return EventoPrescricao(T0 + timedelta(minutes=minutos), "P001", "prescrever", ordem, med, dose, vezes)


def suspende(minutos, ordem):
    return EventoPrescricao(T0 + timedelta(minutes=minutos), "P001", "suspender", ordem)


def tipos(findings):
    return [f.type for f in findings]


def test_normalizar_nome():
    assert normalizar_nome("  Dipirona ") == "dipirona"
    assert normalizar_nome("Ácido Acetilsalicílico") == "acido acetilsalicilico"


def test_dose_normal_nao_gera_finding(cfg):
    assert analisar_eventos(cfg, [pedido(0, "O1", "Dipirona", 500, 4)]) == []


def test_dose_acima_do_maximo_por_tomada(cfg):
    [f] = analisar_eventos(cfg, [pedido(0, "O1", "Paracetamol", 10000, 1)])
    assert isinstance(f, Finding)
    assert (f.source, f.type, f.severity) == ("prescription", "dose_fora_da_faixa", "critical")
    assert f.evidence["medicamento"] == "Paracetamol" and f.evidence["direcao"] == "acima"


def test_dose_diaria_acima_do_maximo(cfg):
    # 1000 mg por tomada é o máximo; 6 tomadas passam da dose diária
    [f] = analisar_eventos(cfg, [pedido(0, "O1", "Paracetamol", 1000, 6)])
    assert f.type == "dose_fora_da_faixa" and f.severity == "medium"
    assert "dose diária" in f.description


def test_dose_muito_abaixo_do_minimo_e_so_low(cfg):
    [f] = analisar_eventos(cfg, [pedido(0, "O1", "Paracetamol", 50, 4)])
    assert f.severity == "low" and f.evidence["direcao"] == "abaixo"


def test_dose_duplicada(cfg):
    findings = analisar_eventos(cfg, [pedido(0, "O1", "Dipirona", 500, 4), pedido(20, "O2", "Dipirona", 500, 4)])
    assert tipos(findings) == ["dose_duplicada"]
    assert findings[0].severity == "medium"  # a soma (4000 mg) ainda não passa do máximo


def test_dose_duplicada_que_passa_do_maximo_e_high(cfg):
    findings = analisar_eventos(cfg, [pedido(0, "O1", "Dipirona", 500, 4), pedido(20, "O2", "Dipirona", 1000, 4)])
    assert findings[0].severity == "high"


def test_troca_de_pedido_nao_e_duplicidade(cfg):
    eventos = [pedido(0, "O1", "Dipirona", 500, 4), suspende(30, "O1"), pedido(30, "O2", "Dipirona", 500, 4)]
    # a suspensão e o pedido novo na mesma hora: a suspensão vem primeiro
    assert analisar_eventos(cfg, list(reversed(eventos))) == []


def test_aumento_abrupto_de_dose(cfg):
    eventos = [
        pedido(0, "O1", "Ibuprofeno", 300, 4),
        suspende(60, "O1"),
        pedido(60, "O2", "Ibuprofeno", 600, 4),
    ]
    [f] = analisar_eventos(cfg, eventos)
    assert f.type == "aumento_abrupto_dose" and f.severity == "high"
    assert f.evidence["aumento_pct"] == 100.0 and f.evidence["ordem_anterior"] == "O1"


def test_aumento_pequeno_nao_gera_finding(cfg):
    eventos = [pedido(0, "O1", "Ibuprofeno", 300, 4), suspende(60, "O1"), pedido(60, "O2", "Ibuprofeno", 350, 4)]
    assert analisar_eventos(cfg, eventos) == []


def test_aumento_ignora_pedido_muito_antigo(cfg):
    eventos = [
        pedido(0, "O1", "Ibuprofeno", 300, 4),
        suspende(60, "O1"),
        pedido(60 + 73 * 60, "O2", "Ibuprofeno", 600, 4),  # 73 h depois
    ]
    assert analisar_eventos(cfg, eventos) == []


def test_suspensao_abrupta_so_depois_da_janela(cfg):
    uso_longo = T0 - timedelta(hours=48)
    ev = [
        EventoPrescricao(uso_longo, "P001", "prescrever", "O1", "Metoprolol", 50, 2),
        suspende(0, "O1"),
    ]
    detector = DetectorPrescricoes(cfg, "P001")
    saida = []
    for e in ev:
        saida += detector.update(e)
    assert saida == []  # ainda dentro da janela de substituição
    assert detector.avancar_relogio(T0 + timedelta(minutes=10)) == []
    [f] = detector.avancar_relogio(T0 + timedelta(minutes=31))
    assert f.type == "suspensao_abrupta"
    assert detector.avancar_relogio(T0 + timedelta(hours=5)) == []  # só emite uma vez


def test_suspensao_seguida_de_novo_pedido_e_troca(cfg):
    ev = [
        EventoPrescricao(T0 - timedelta(hours=48), "P001", "prescrever", "O1", "Metoprolol", 50, 2),
        suspende(0, "O1"),
        pedido(10, "O2", "Metoprolol", 50, 2),
    ]
    findings = analisar_eventos(cfg, ev, ate=T0 + timedelta(hours=3))
    assert findings == []


def test_medicamento_de_uso_curto_suspenso_nao_conta(cfg):
    ev = [pedido(-60, "O1", "Metoprolol", 50, 2), suspende(0, "O1")]  # só 1 h de uso
    assert analisar_eventos(cfg, ev, ate=T0 + timedelta(hours=3)) == []


def test_eventos_invalidos_sao_descartados_e_contados(cfg):
    d = DetectorPrescricoes(cfg, "P001")
    assert d.update(pedido(0, "O1", "Remedio Inventado", 10, 1)) == []
    assert d.update(pedido(1, "O2", "Dipirona", -5, 1)) == []
    assert d.update(pedido(2, "O3", "Dipirona", float("nan"), 1)) == []
    assert d.update(suspende(3, "O999")) == []
    assert d.update(pedido(-10, "O4", "Dipirona", 500, 4)) == []  # antes do relógio
    assert d.descartes["medicamento_fora_do_catalogo"] == 1
    assert d.descartes["dose_invalida"] == 2
    assert d.descartes["ordem_desconhecida"] == 1
    assert d.descartes["fora_de_ordem"] == 1


def test_evento_de_outro_paciente_e_recusado(cfg):
    d = DetectorPrescricoes(cfg, "P002")
    with pytest.raises(ValueError):
        d.update(pedido(0, "O1", "Dipirona", 500, 4))


def test_timestamp_sem_fuso_e_recusado(cfg):
    d = DetectorPrescricoes(cfg, "P001")
    with pytest.raises(ValueError):
        d.update(EventoPrescricao(datetime(2026, 11, 3, 8, 30), "P001", "prescrever", "O1", "Dipirona", 500, 4))


def test_cenarios_do_simulador_batem_com_os_rotulos(cfg, sim):
    for nome in ("p001_demo", "prescricoes_demo"):
        eventos, rotulos = gerar_prescricoes(sim, "P001", T0, sim.cenarios[nome].prescricoes)
        findings = analisar_eventos(cfg, eventos, ate=T0 + timedelta(hours=3))
        achados = sorted((f.type, f.timestamp_start) for f in findings)
        esperados = sorted((r.tipo, r.inicio) for r in rotulos)
        assert [t for t, _ in achados] == [t for t, _ in esperados]


def test_sem_anomalia_a_prescricao_normal_nao_gera_finding(cfg, sim):
    eventos, rotulos = gerar_prescricoes(sim, "P001", T0, sim.cenarios["p001_demo"].prescricoes)
    normais = [e for e in eventos if e.timestamp < T0]  # só a admissão
    assert analisar_eventos(cfg, normais, ate=T0 + timedelta(hours=3)) == []
