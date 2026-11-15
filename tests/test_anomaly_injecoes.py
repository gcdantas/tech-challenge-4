"""Testes das anomalias e dos distratores injetados nos sinais vitais."""
from datetime import datetime

import numpy as np
import pytest
from pydantic import ValidationError

from contracts.finding import DEMO_TZ
from modules.anomaly.config import InjecaoCfg, carregar_config_simulador
from modules.anomaly.simulator import gerar_cenario_vitais, gerar_vitais, rotulos_das_injecoes

INICIO = datetime(2026, 11, 3, 8, 30, tzinfo=DEMO_TZ)


@pytest.fixture
def cfg():
    return carregar_config_simulador()


def minutos(quadro):
    """Minutos desde o início de cada amostra, para filtrar janelas fácil."""
    return (quadro["timestamp"] - INICIO).dt.total_seconds().to_numpy() / 60


def test_dessaturacao_derruba_spo2_e_sobe_a_fc(cfg):
    inj = InjecaoCfg(tipo="dessaturacao", inicio_min=45, duracao_min=3, magnitude=10, rampa_s=40)
    base = gerar_vitais(cfg, "P001", INICIO)
    quadro = gerar_vitais(cfg, "P001", INICIO, injecoes=[inj])
    dentro = (minutos(quadro) >= 45.7) & (minutos(quadro) < 48)  # platô, depois da rampa de subida

    assert quadro.loc[dentro, "spo2"].max() < 89  # caiu uns 10 pontos de 97
    assert quadro.loc[dentro, "fc"].mean() - base.loc[dentro, "fc"].mean() > 8  # resposta fisiológica


def test_serie_fica_igual_fora_do_evento(cfg):
    # a injeção não pode mexer em nada antes nem depois do evento
    inj = InjecaoCfg(tipo="dessaturacao", inicio_min=45, duracao_min=3, magnitude=10, rampa_s=40)
    base = gerar_vitais(cfg, "P001", INICIO)
    quadro = gerar_vitais(cfg, "P001", INICIO, injecoes=[inj])
    fora = (minutos(quadro) < 45) | (minutos(quadro) > 48 + 40 / 60)
    assert quadro.loc[fora].equals(base.loc[fora])


def test_rampa_sobe_aos_poucos(cfg):
    inj = InjecaoCfg(tipo="taquicardia", inicio_min=20, duracao_min=5, magnitude=40, rampa_s=60)
    base = gerar_vitais(cfg, "P001", INICIO)
    quadro = gerar_vitais(cfg, "P001", INICIO, injecoes=[inj])
    diferenca = (quadro["fc"] - base["fc"]).to_numpy()
    m = minutos(quadro)

    assert abs(diferenca[np.argmin(abs(m - 20.5))] - 20) <= 1  # na metade da rampa, metade da magnitude
    assert abs(diferenca[np.argmin(abs(m - 22))] - 40) <= 1  # no platô, magnitude inteira


@pytest.mark.parametrize(
    "tipo, sinal, sentido",
    [
        ("taquicardia", "fc", +1),
        ("hipotensao", "pas", -1),
        ("febre", "temp", +1),
        ("taquipneia", "fr", +1),
        ("deterioracao_multivariada", "spo2", -1),
    ],
)
def test_cada_anomalia_mexe_no_sinal_certo(cfg, tipo, sinal, sentido):
    inj = InjecaoCfg(tipo=tipo, inicio_min=30, duracao_min=5, magnitude=3, rampa_s=30)
    base = gerar_vitais(cfg, "P001", INICIO)
    quadro = gerar_vitais(cfg, "P001", INICIO, injecoes=[inj])
    dentro = (minutos(quadro) >= 31) & (minutos(quadro) < 35)
    diferenca = (quadro.loc[dentro, sinal] - base.loc[dentro, sinal]).mean()
    assert sentido * diferenca > 0


def test_artefato_pico_e_curto_e_so_no_spo2(cfg):
    inj = InjecaoCfg(tipo="artefato_pico", inicio_min=30, duracao_min=0.5, magnitude=12)
    base = gerar_vitais(cfg, "P001", INICIO)
    quadro = gerar_vitais(cfg, "P001", INICIO, injecoes=[inj])
    dentro = (minutos(quadro) >= 30) & (minutos(quadro) < 30.5)

    assert dentro.sum() == 3  # 30 s com amostras a cada 10 s
    assert (quadro.loc[dentro, "spo2"] < base.loc[dentro, "spo2"] - 10).all()
    assert quadro["fc"].equals(base["fc"])  # o resto do corpo está normal


def test_perda_de_sinal_zera_tudo_com_nan(cfg):
    inj = InjecaoCfg(tipo="perda_sinal", inicio_min=30, duracao_min=2)
    quadro = gerar_vitais(cfg, "P001", INICIO, injecoes=[inj])
    dentro = (minutos(quadro) >= 30) & (minutos(quadro) < 32)

    assert quadro.loc[dentro, ["fc", "spo2", "fr", "pas", "pad", "temp"]].isna().all().all()
    assert quadro.loc[~dentro, ["fc", "spo2"]].notna().all().all()


def test_leitura_invalida_marca_zero_em_spo2_e_fc(cfg):
    inj = InjecaoCfg(tipo="leitura_invalida", inicio_min=30, duracao_min=1)
    quadro = gerar_vitais(cfg, "P001", INICIO, injecoes=[inj])
    dentro = (minutos(quadro) >= 30) & (minutos(quadro) < 31)

    assert (quadro.loc[dentro, ["spo2", "fc"]] == 0).all().all()
    assert (quadro.loc[dentro, "pas"] > 100).all()  # a pressão segue normal


def test_rotulos_marcam_anomalia_e_distrator():
    injecoes = [
        InjecaoCfg(tipo="dessaturacao", inicio_min=45, duracao_min=3, magnitude=10, rampa_s=40),
        InjecaoCfg(tipo="artefato_pico", inicio_min=10, duracao_min=0.5, magnitude=12),
    ]
    anomalia, distrator = rotulos_das_injecoes(INICIO, injecoes)

    assert anomalia.anomalia and not distrator.anomalia
    assert (anomalia.inicio - INICIO).total_seconds() == 45 * 60
    # o fim da anomalia inclui a rampa de descida (3 min + 40 s)
    assert (anomalia.fim - anomalia.inicio).total_seconds() == 3 * 60 + 40
    assert (distrator.fim - distrator.inicio).total_seconds() == 30


def test_injecao_depois_do_fim_da_serie_e_rejeitada(cfg):
    inj = InjecaoCfg(tipo="taquicardia", inicio_min=200, duracao_min=2, magnitude=30)
    with pytest.raises(ValueError):
        gerar_vitais(cfg, "P001", INICIO, injecoes=[inj])


def test_tipo_desconhecido_e_magnitude_zero_sao_rejeitados():
    with pytest.raises(ValidationError):
        InjecaoCfg(tipo="explosao", inicio_min=1, duracao_min=1, magnitude=5)
    with pytest.raises(ValidationError):
        InjecaoCfg(tipo="dessaturacao", inicio_min=1, duracao_min=1, magnitude=0)


def test_cenario_p001_demo_do_yaml(cfg):
    quadro, rotulos = gerar_cenario_vitais(cfg, "p001_demo", "P001", INICIO)
    assert len(rotulos) == 1 and rotulos[0].tipo == "dessaturacao" and rotulos[0].anomalia
    assert quadro["spo2"].min() < 89


def test_cenario_inexistente_da_erro_claro(cfg):
    with pytest.raises(KeyError):
        gerar_cenario_vitais(cfg, "nao_existe", "P001", INICIO)