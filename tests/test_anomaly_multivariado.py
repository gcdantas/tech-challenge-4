"""Testes do Isolation Forest e do monitor de vitais completo."""
from datetime import datetime

import numpy as np
import pytest

from contracts.finding import DEMO_TZ
from modules.anomaly.config import SINAIS, carregar_config, carregar_config_simulador
from modules.anomaly.multivariado import MonitorVitais, ModeloIF, analisar_quadro_completo
from modules.anomaly.simulator import gerar_cenario_vitais, gerar_vitais
from modules.anomaly.vitals import PontoVital, pontos_do_quadro

INICIO = datetime(2026, 11, 3, 8, 30, tzinfo=DEMO_TZ)


@pytest.fixture(scope="module")
def cfg():
    return carregar_config()


@pytest.fixture(scope="module")
def sim():
    return carregar_config_simulador()


def modelo_treinado(cfg, n=200, semente=1):
    rng = np.random.default_rng(semente)
    medias = np.array([76, 97, 16, 122, 76, 36.7])
    desvios = np.array([2, 0.4, 0.8, 3, 2, 0.04])
    modelo = ModeloIF(cfg.isolation_forest, cfg.vitais.zscore.escala_min)
    for k in range(n):
        modelo.guardar_normal(k * 10.0, list(medias + rng.normal(0, 1, 6) * desvios))
    modelo.treinar_se_preciso(n * 10.0)
    return modelo, medias


def test_nao_treina_antes_do_minimo(cfg):
    modelo = ModeloIF(cfg.isolation_forest, cfg.vitais.zscore.escala_min)
    for k in range(cfg.isolation_forest.min_treino - 1):
        modelo.guardar_normal(k * 10.0, [76, 97, 16, 122, 76, 36.7])
        modelo.treinar_se_preciso(k * 10.0)
    assert not modelo.treinado
    assert modelo.pontuar([76, 97, 16, 122, 76, 36.7]) is None


def test_ponto_normal_fica_abaixo_do_limiar_e_ponto_estranho_passa(cfg):
    modelo, medias = modelo_treinado(cfg)
    normal = modelo.pontuar(list(medias))
    assert normal.excesso == 0

    # FC e FR altas e SpO2 baixa ao mesmo tempo, cada uma ainda "pequena"
    estranho = modelo.pontuar([medias[0] + 12, medias[1] - 3, medias[2] + 6, medias[3], medias[4], medias[5]])
    assert estranho.excesso > 0
    assert estranho.escore > estranho.limiar


def test_contribuintes_apontam_o_sinal_que_se_afastou(cfg):
    modelo, medias = modelo_treinado(cfg)
    vetor = list(medias)
    vetor[SINAIS.index("pas")] += 40
    r = modelo.pontuar(vetor)
    assert r.contribuintes[0]["sinal"] == "pas"
    assert r.contribuintes[0]["desvio_robusto"] > 5


def test_reiniciar_esquece_o_modelo(cfg):
    modelo, _ = modelo_treinado(cfg)
    assert modelo.treinado
    modelo.reiniciar()
    assert not modelo.treinado and len(modelo.buffer) == 0


def test_resultado_e_reproduzivel(cfg):
    a, medias = modelo_treinado(cfg)
    b, _ = modelo_treinado(cfg)
    vetor = list(medias + 3)
    assert a.pontuar(vetor).escore == b.pontuar(vetor).escore


def test_pacientes_estaveis_quase_nunca_geram_finding_do_if(cfg, sim):
    com_if = 0
    for i in range(10):
        quadro = gerar_vitais(sim, f"M{i:03d}", INICIO, variar_paciente=True)
        findings = analisar_quadro_completo(cfg, quadro)
        assert all(f.type == "padrao_multivariado" for f in findings)  # nada dos outros detectores
        com_if += bool(findings)
    assert com_if <= 1


def test_monitor_mantem_os_findings_dos_sinais(cfg, sim):
    quadro, _ = gerar_cenario_vitais(sim, "p001_demo", "P001", INICIO)
    tipos = [f.type for f in analisar_quadro_completo(cfg, quadro)]
    assert tipos.count("dessaturacao") == 1
    assert tipos.count("padrao_multivariado") == 0  # a dessaturação já está explicada pelos sinais


def test_distratores_continuam_sem_finding(cfg, sim):
    quadro, _ = gerar_cenario_vitais(sim, "distratores_demo", "P001", INICIO)
    assert analisar_quadro_completo(cfg, quadro) == []


def test_ponto_sem_um_dos_sinais_nao_quebra_nem_treina(cfg):
    monitor = MonitorVitais(cfg, "P001")
    ponto = PontoVital(INICIO, "P001", {"fc": 76.0, "spo2": 97.0})
    assert monitor.update(ponto) == []
    assert len(monitor.modelo.buffer) == 0


def test_if_desligado_no_yaml_nao_treina(cfg, sim):
    desligado = cfg.model_copy(deep=True)
    desligado.isolation_forest.ativo = False
    quadro = gerar_vitais(sim, "P001", INICIO)
    monitor = MonitorVitais(desligado, "P001")
    for ponto in pontos_do_quadro(quadro):
        monitor.update(ponto)
    assert not monitor.modelo.treinado
