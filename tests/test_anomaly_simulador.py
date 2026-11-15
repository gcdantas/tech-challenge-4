"""Testes do simulador de sinais vitais."""
from datetime import datetime

import numpy as np
import pytest

from contracts.finding import DEMO_TZ
from modules.anomaly.config import SINAIS, carregar_config, carregar_config_simulador
from modules.anomaly.simulator import gerar_vitais

INICIO = datetime(2026, 11, 3, 8, 30, tzinfo=DEMO_TZ)


@pytest.fixture
def cfg():
    return carregar_config_simulador()


def test_formato_da_serie(cfg):
    quadro = gerar_vitais(cfg, "P001", INICIO)
    # 90 min com uma amostra a cada 10s
    assert len(quadro) == 540
    assert list(quadro.columns) == ["patient_id", "timestamp", *SINAIS]
    assert not quadro[list(SINAIS)].isna().any().any()
    assert (quadro["timestamp"].diff().dropna().dt.total_seconds() == 10).all()
    assert str(quadro["timestamp"].iloc[0].utcoffset()) == "-1 day, 21:00:00"  # -03:00


def test_mesma_semente_gera_a_mesma_serie(cfg):
    a = gerar_vitais(cfg, "P001", INICIO)
    b = gerar_vitais(cfg, "P001", INICIO)
    assert a.equals(b)


def test_pacientes_diferentes_geram_series_diferentes(cfg):
    a = gerar_vitais(cfg, "P001", INICIO)
    b = gerar_vitais(cfg, "P002", INICIO)
    assert not a["fc"].equals(b["fc"])


def test_valores_sao_realistas(cfg):
    quadro = gerar_vitais(cfg, "P001", INICIO)
    assert quadro["spo2"].max() <= 100
    assert (quadro["fc"] % 1 == 0).all()  # arredondado para inteiro
    assert np.allclose(quadro["temp"] * 10, np.round(quadro["temp"] * 10))  # 1 casa decimal


def test_ruido_tem_memoria(cfg):
    # com AR(1) um ponto se parece com o anterior; ruído branco ficaria perto de 0
    fc = gerar_vitais(cfg, "P001", INICIO)["fc"].to_numpy()
    autocorrelacao = np.corrcoef(fc[:-1], fc[1:])[0, 1]
    assert autocorrelacao > 0.5


def test_paciente_estavel_fica_dentro_do_normal(cfg):
    # 200 pacientes diferentes, nenhum pode passar do limite "medium" do anomaly.yaml.
    # Se passar, o simulador geraria falso alarme sem nenhuma anomalia injetada
    limites = carregar_config().vitais.sinais
    for i in range(200):
        quadro = gerar_vitais(cfg, f"T{i:03d}", INICIO, variar_paciente=True)
        for sinal, lim in limites.items():
            if lim.baixo:
                assert quadro[sinal].min() > lim.baixo.medium, (i, sinal)
            if lim.alto:
                assert quadro[sinal].max() < lim.alto.medium, (i, sinal)


def test_inicio_sem_fuso_e_rejeitado(cfg):
    with pytest.raises(ValueError):
        gerar_vitais(cfg, "P001", datetime(2026, 11, 3, 8, 30))