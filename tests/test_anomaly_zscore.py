"""Testes do z-score com janela."""
from datetime import datetime

import numpy as np
import pytest

from contracts.finding import DEMO_TZ
from modules.anomaly.config import SINAIS, carregar_config, carregar_config_simulador
from modules.anomaly.simulator import gerar_cenario_vitais, gerar_vitais
from modules.anomaly.zscore import FATOR_MAD, LinhaBase, z_da_serie

INICIO = datetime(2026, 11, 3, 8, 30, tzinfo=DEMO_TZ)


def linha_com(valores, escala_min=0.1, janela_s=900, min_pontos=60):
    """LinhaBase já com `valores` guardados, uma amostra a cada 10 s."""
    linha = LinhaBase(janela_s, min_pontos, escala_min)
    for k, v in enumerate(valores):
        linha.guardar(k * 10.0, v)
    return linha


def test_sem_pontos_suficientes_nao_pontua():
    linha = linha_com([10.0] * 59)
    assert linha.pontuar(600.0, 10.0) is None


def test_z_robusto_com_mediana_e_mad_conhecidos():
    # ciclo 9, 10, 11: mediana = 10 e MAD = 1
    linha = linha_com([9, 10, 11] * 30)
    z = linha.pontuar(900.0, 20.0)
    assert z == pytest.approx(10 / FATOR_MAD)


def test_piso_da_escala_evita_z_infinito():
    # sinal parado em 97: o MAD é zero e, sem o piso, qualquer mudança daria z infinito
    linha = linha_com([97.0] * 60, escala_min=0.6)
    assert linha.pontuar(600.0, 96.0) == pytest.approx(-1 / 0.6)
    assert linha.pontuar(600.0, 90.0) == pytest.approx(-7 / 0.6)


def test_mediana_resiste_a_anomalia_que_o_desvio_padrao_nao_resiste():
    # 48 valores em volta de 10 e 12 leituras de 60 (20% da janela já anômala)
    rng = np.random.default_rng(1)
    normais = list(10 + rng.normal(0, 1, 48))
    valores = normais + [60.0] * 12

    z_classico = (60 - np.mean(valores)) / np.std(valores)
    z_robusto = linha_com(valores).pontuar(900.0, 60.0)

    assert z_classico < 3.5  # o z-score das aulas deixa passar: a anomalia inflou o desvio padrão
    assert z_robusto > 20  # o robusto continua enxergando


def test_nan_e_infinito_sao_ignorados():
    linha = linha_com([10.0] * 60)
    assert linha.pontuar(700.0, float("nan")) is None
    assert linha.pontuar(700.0, float("inf")) is None
    linha.guardar(700.0, float("nan"))
    assert len(linha.pontos) == 60


def test_pontuar_nao_guarda_a_amostra():
    linha = linha_com([10.0] * 60)
    linha.pontuar(700.0, 50.0)
    assert len(linha.pontos) == 60


def test_janela_e_por_tempo_e_esquece_o_passado():
    linha = linha_com([10.0] * 60)
    assert linha.pontuar(600.0, 10.0) is not None
    # uma hora depois (a janela é de 15 min) as amostras antigas já saíram
    assert linha.pontuar(4200.0, 10.0) is None
    assert len(linha.pontos) == 0


def test_timestamp_fora_de_ordem_da_erro():
    linha = linha_com([10.0] * 60)  # última amostra em t = 590
    with pytest.raises(ValueError):
        linha.guardar(590.0, 10.0)
    with pytest.raises(ValueError):
        linha.pontuar(300.0, 10.0)


def test_dessaturacao_aparece_no_z_do_spo2():
    cfg = carregar_config()
    quadro, [rotulo] = gerar_cenario_vitais(carregar_config_simulador(), "p001_demo", "P001", INICIO)
    z = z_da_serie(quadro, "spo2", cfg)

    dentro = (quadro["timestamp"] >= rotulo.inicio) & (quadro["timestamp"] <= rotulo.fim)
    assert z[dentro].abs().max() > 10
    fora = z[~dentro].dropna().abs()  # os primeiros minutos não têm z (a linha de base ainda está enchendo)
    assert len(fora) > 400
    assert (fora < cfg.vitais.zscore.limiar).all()


def test_paciente_estavel_quase_nao_gera_desvio():
    cfg = carregar_config()
    sim = carregar_config_simulador()
    acima, total = 0, 0
    for i in range(20):
        quadro = gerar_vitais(sim, f"T{i:03d}", INICIO, variar_paciente=True)
        for sinal in SINAIS:
            z = z_da_serie(quadro, sinal, cfg).abs()
            acima += int((z > cfg.vitais.zscore.limiar).sum())
            total += int(z.notna().sum())
    assert acima / total < 0.005  # menos de 0,5% das amostras


def test_distratores_so_o_artefato_chama_atencao():
    # O z-score sozinho acusa o artefato (queda de 12 pontos no SpO2). Por isso o detector de vitais
    # ainda precisa exigir uma duração mínima antes de gerar um Finding.
    cfg = carregar_config()
    quadro, _ = gerar_cenario_vitais(carregar_config_simulador(), "distratores_demo", "P001", INICIO)

    z_spo2 = z_da_serie(quadro, "spo2", cfg)
    assert int((z_spo2.abs() > cfg.vitais.zscore.limiar).sum()) == 3  # as 3 amostras do artefato

    # perda de sinal (NaN) e leitura inválida (0 fora da faixa física) nem chegam a ser pontuadas
    minutos = (quadro["timestamp"] - INICIO).dt.total_seconds() / 60
    perda = (minutos >= 50) & (minutos < 52)
    invalida = (minutos >= 70) & (minutos < 71)
    assert z_spo2[perda | invalida].isna().all()
    assert z_da_serie(quadro, "fc", cfg)[invalida].isna().all()