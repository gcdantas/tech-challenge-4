"""Testes do detector de movimentação e do simulador de movimentação."""
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from contracts.finding import DEMO_TZ, Finding
from modules.anomaly.config import EventoMovimentoCfg, carregar_config, carregar_config_simulador
from modules.anomaly.movement import (
    COLUNAS,
    DetectorMovimento,
    PontoMovimento,
    analisar_quadro,
    pontos_do_quadro,
)
from modules.anomaly.simulator import gerar_cenario_movimento, gerar_movimento

INICIO = datetime(2026, 11, 3, 8, 30, tzinfo=DEMO_TZ)


@pytest.fixture(scope="module")
def cfg():
    return carregar_config()


@pytest.fixture(scope="module")
def sim():
    return carregar_config_simulador()


def ev(tipo, inicio_min, duracao_s=0, **kw):
    return EventoMovimentoCfg(tipo=tipo, inicio_min=inicio_min, duracao_s=duracao_s, **kw)


def serie_leito(trechos, hz=5):
    """Monta uma série a mão. Cada trecho é (segundos, in_bed). Pose de quem está deitado ou de pé."""
    linhas = []
    k = 0  # número da amostra: com inteiros o tempo sai exato (somar 0.2 s acumula erro)
    for segundos, in_bed in trechos:
        for _ in range(int(segundos * hz)):
            linhas.append(
                {
                    "patient_id": "P001",
                    "timestamp": pd.Timestamp(INICIO) + pd.Timedelta(milliseconds=k * 1000 // hz),
                    "centroid_x": 0.5 if in_bed else 0.9,
                    "centroid_y": 0.55,
                    "torso_angle": 85.0 if in_bed else 10.0,
                    "velocidade": 0.0,
                    "in_bed": in_bed,
                }
            )
            k += 1
    return pd.DataFrame(linhas, columns=COLUNAS)


# ---------------------------------------------------------------- saída do leito


def test_saida_do_leito_gera_um_finding(cfg):
    quadro = serie_leito([(20, True), (30, False), (20, True)])
    [f] = analisar_quadro(cfg, quadro)
    assert isinstance(f, Finding)
    assert (f.source, f.type, f.severity, f.score) == ("movement", "saida_do_leito", "medium", 0.7)
    assert f.timestamp_start == INICIO + timedelta(seconds=20)
    assert f.timestamp_end - f.timestamp_start == timedelta(seconds=3)  # confirmou em 3 s
    assert f.evidence["pessoa_detectada"] is True


def test_quadro_falso_curto_nao_gera_finding(cfg):
    quadro = serie_leito([(20, True), (2, False), (20, True)])
    assert analisar_quadro(cfg, quadro) == []


def test_quadro_falso_antes_da_saida_nao_antecipa_o_inicio(cfg):
    # como no SSP8C: um quadro falso aos 11,8 s, de volta ao leito por 6 s e só então a saída de verdade (18 s)
    quadro = serie_leito([(12, True), (0.2, False), (6, True), (10, False), (10, True)])
    [f] = analisar_quadro(cfg, quadro)
    assert f.timestamp_start == INICIO + timedelta(seconds=18.2)
    assert f.timestamp_end - f.timestamp_start == timedelta(seconds=3)


def test_pessoa_se_debrucando_nao_quebra_a_saida_em_duas(cfg):
    # como no vídeo SSP8C: fora, volta por 4 s, fora de novo. 10 s de retorno é que encerra
    quadro = serie_leito([(20, True), (10, False), (4, True), (10, False), (4, True), (10, False), (20, True)])
    findings = analisar_quadro(cfg, quadro)
    assert [f.type for f in findings] == ["saida_do_leito"]


def test_voltar_ao_leito_e_sair_de_novo_gera_dois_findings(cfg):
    quadro = serie_leito([(20, True), (10, False), (30, True), (10, False), (20, True)])
    assert [f.type for f in analisar_quadro(cfg, quadro)] == ["saida_do_leito"] * 2


def test_serie_que_comeca_fora_do_leito_nao_e_saida(cfg):
    quadro = serie_leito([(60, False)])
    assert analisar_quadro(cfg, quadro) == []


def test_ausencia_longa_sobe_para_high(cfg):
    quadro = serie_leito([(20, True), (700, False)], hz=1)
    findings = analisar_quadro(cfg, quadro)
    assert [f.severity for f in findings] == ["medium", "high"]
    assert findings[1].score == 0.85
    assert findings[0].evidence["episodio_id"] == findings[1].evidence["episodio_id"]


def test_pessoa_nao_detectada_com_in_bed_falso_conta_como_fora(cfg):
    # contrato do I1: sem detecção, in_bed fica falso depois de 1 s
    quadro = serie_leito([(20, True), (10, False)])
    quadro.loc[quadro["in_bed"] == False, ["torso_angle", "centroid_x", "centroid_y"]] = np.nan  # noqa: E712
    [f] = analisar_quadro(cfg, quadro)
    assert f.evidence["pessoa_detectada"] is False
    assert f.evidence["torso_angle"] is None  # NaN não vai para o JSON
    f.model_dump_json()


def test_in_bed_ausente_e_descartado(cfg):
    d = DetectorMovimento(cfg, "P001")
    assert d.update(PontoMovimento(INICIO, "P001", None)) == []
    assert d.descartes["in_bed_ausente"] == 1


def test_ponto_fora_de_ordem_e_descartado(cfg):
    d = DetectorMovimento(cfg, "P001")
    d.update(PontoMovimento(INICIO + timedelta(seconds=5), "P001", True))
    assert d.update(PontoMovimento(INICIO, "P001", True)) == []
    assert d.descartes["fora_de_ordem"] == 1


def test_paciente_errado_e_fuso_ausente_levantam_erro(cfg):
    d = DetectorMovimento(cfg, "P001")
    with pytest.raises(ValueError):
        d.update(PontoMovimento(INICIO, "P002", True))
    with pytest.raises(ValueError):
        d.update(PontoMovimento(datetime(2026, 11, 3, 8, 30), "P001", True))


def test_lacuna_grande_zera_o_que_estava_em_andamento(cfg):
    d = DetectorMovimento(cfg, "P001")
    for k in range(10):
        d.update(PontoMovimento(INICIO + timedelta(seconds=k), "P001", True, 85.0, 0.5, 0.55))
    # 10 min sem dados; ao voltar, a pessoa aparece fora do leito. Não dá para chamar de saída
    saida = d.update(PontoMovimento(INICIO + timedelta(minutes=10), "P001", False, 10.0, 0.9, 0.55))
    saida += d.update(PontoMovimento(INICIO + timedelta(minutes=10, seconds=5), "P001", False, 10.0, 0.9, 0.55))
    assert saida == []


# ---------------------------------------------------------------- imobilidade


def test_paciente_que_se_reposiciona_nunca_gera_finding(cfg, sim):
    for i in range(8):
        quadro, _ = gerar_movimento(sim, f"M{i:03d}", INICIO, duracao_min=300)
        assert analisar_quadro(cfg, quadro) == [], f"falso alarme no paciente M{i:03d}"


def test_imobilidade_e_detectada_depois_do_limiar(cfg, sim):
    quadro, rotulos = gerar_movimento(sim, "P001", INICIO, duracao_min=300, eventos=[ev("imobilidade", 100, 10800)])
    [rotulo] = rotulos
    [f, *resto] = analisar_quadro(cfg, quadro)
    assert (f.source, f.type, f.severity) == ("movement", "imobilidade_prolongada", "low")
    # a contagem começa no primeiro bloco de 1 min já na pose nova
    assert timedelta(0) <= f.timestamp_start - rotulo.inicio <= timedelta(seconds=60)
    limiar = timedelta(seconds=cfg.movimento.imobilidade.limiar_s)
    assert limiar <= f.timestamp_end - rotulo.inicio <= limiar + timedelta(seconds=180)
    assert rotulo.inicio <= f.timestamp_end <= rotulo.fim
    assert 55 <= f.evidence["pose"]["torso_angle"] <= 120  # ângulo do tronco na pose em que ficou parado


def test_imobilidade_sobe_de_severidade_com_o_tempo(cfg, sim):
    quadro, _ = gerar_movimento(sim, "P001", INICIO, duracao_min=420, eventos=[ev("imobilidade", 30, 6 * 3600)])
    findings = analisar_quadro(cfg, quadro)
    assert [f.severity for f in findings] == ["low", "medium", "high"]
    assert len({f.evidence["episodio_id"] for f in findings}) == 1
    assert [f.score for f in findings] == sorted(f.score for f in findings)


def test_virar_o_paciente_zera_a_contagem(cfg, sim):
    # parado 100 min, vira, e mais 100 min parado: nenhuma das partes chega a 2 h
    quadro, _ = gerar_movimento(
        sim, "P001", INICIO, duracao_min=250, eventos=[ev("imobilidade", 5, 95 * 60), ev("imobilidade", 105, 95 * 60)]
    )
    assert analisar_quadro(cfg, quadro) == []


def test_sair_do_leito_zera_a_imobilidade(cfg, sim):
    quadro, _ = gerar_movimento(
        sim, "P001", INICIO, duracao_min=250, eventos=[ev("imobilidade", 5, 200 * 60), ev("saida_do_leito", 100, 120)]
    )
    tipos = [f.type for f in analisar_quadro(cfg, quadro)]
    assert tipos.count("saida_do_leito") == 1
    assert "imobilidade_prolongada" not in tipos  # 95 min + 100 min, cada um abaixo do limiar


def test_pessoa_nao_detectada_por_muito_tempo_nao_conta_como_imovel(cfg, sim):
    quadro, _ = gerar_movimento(sim, "P001", INICIO, duracao_min=200, eventos=[ev("imobilidade", 5, 190 * 60)])
    # 20 min sem pose no meio da imobilidade: a contagem zera e cada metade fica abaixo de 2 h
    sem_pose = (quadro["timestamp"] >= INICIO + timedelta(minutes=60)) & (quadro["timestamp"] < INICIO + timedelta(minutes=80))
    quadro.loc[sem_pose, ["torso_angle", "centroid_x", "centroid_y", "velocidade"]] = np.nan
    assert [f.type for f in analisar_quadro(cfg, quadro) if f.type == "imobilidade_prolongada"] == []


# ---------------------------------------------------------------- cenários e fluxo


def test_cenario_movimento_demo(cfg, sim):
    quadro, rotulos = gerar_cenario_movimento(sim, "movimento_demo", "P001", INICIO)
    findings = analisar_quadro(cfg, quadro)
    saida = [f for f in findings if f.type == "saida_do_leito"]
    imob = [f for f in findings if f.type == "imobilidade_prolongada"]

    assert len(saida) == 1 and len(imob) >= 1  # os três distratores não geram nada
    [r_saida] = [r for r in rotulos if r.tipo == "saida_do_leito"]
    assert saida[0].timestamp_start >= r_saida.inicio
    assert saida[0].timestamp_start - r_saida.inicio < timedelta(seconds=20)  # sobe em rampa antes do in_bed cair
    assert {f.severity for f in imob} <= {"low", "medium"}


def test_p001_demo_tem_saida_do_leito(cfg, sim):
    quadro, [rotulo] = gerar_cenario_movimento(sim, "p001_demo", "P001", INICIO)
    [f] = analisar_quadro(cfg, quadro)
    assert f.type == "saida_do_leito" and rotulo.inicio <= f.timestamp_start <= rotulo.fim


def test_fluxo_ponto_a_ponto_igual_ao_em_lote(cfg, sim):
    quadro, _ = gerar_cenario_movimento(sim, "p001_demo", "P001", INICIO)
    detector = DetectorMovimento(cfg, "P001")
    fluxo = []
    for ponto in pontos_do_quadro(quadro):
        fluxo += detector.update(ponto)
    fluxo += detector.finalizar()
    assert [f.model_dump(exclude={"finding_id"}) for f in fluxo] == [
        f.model_dump(exclude={"finding_id"}) for f in analisar_quadro(cfg, quadro)
    ]
    assert [e.tipo for e in detector.episodios()] == ["saida_do_leito"]


def test_funciona_com_5_hz_como_o_parquet_do_i1(cfg, sim):
    quadro, _ = gerar_movimento(sim, "P001", INICIO, duracao_min=20, hz=5, eventos=[ev("saida_do_leito", 5, 120)])
    assert quadro["timestamp"].diff().dropna().eq(pd.Timedelta(milliseconds=200)).all()
    assert [f.type for f in analisar_quadro(cfg, quadro)] == ["saida_do_leito"]


def test_quadro_sem_colunas_levanta_erro(cfg):
    with pytest.raises(ValueError, match="faltam colunas"):
        list(pontos_do_quadro(pd.DataFrame({"patient_id": ["P001"]})))


# ---------------------------------------------------------------- simulador


def test_simulador_gera_as_colunas_do_i1(sim):
    quadro, _ = gerar_movimento(sim, "P001", INICIO, duracao_min=10)
    assert list(quadro.columns) == COLUNAS
    assert quadro["in_bed"].dtype == bool
    assert quadro["centroid_x"].between(0, 1).all() and quadro["centroid_y"].between(0, 1).all()
    assert quadro["timestamp"].iloc[0] == INICIO


def test_simulador_e_reproduzivel_e_depende_do_paciente(sim):
    a, _ = gerar_movimento(sim, "P001", INICIO, duracao_min=30)
    b, _ = gerar_movimento(sim, "P001", INICIO, duracao_min=30)
    c, _ = gerar_movimento(sim, "P002", INICIO, duracao_min=30)
    pd.testing.assert_frame_equal(a, b)
    assert not a["torso_angle"].equals(c["torso_angle"])


def test_perda_de_deteccao_deixa_a_pose_vazia_e_in_bed_cai_depois_de_1_s(sim):
    quadro, _ = gerar_movimento(sim, "P001", INICIO, duracao_min=5, eventos=[ev("perda_deteccao", 2, 3)])
    janela = quadro[(quadro["timestamp"] >= INICIO + timedelta(minutes=2)) & (quadro["timestamp"] < INICIO + timedelta(minutes=2, seconds=3))]
    assert janela["torso_angle"].isna().all()
    assert janela["in_bed"].tolist() == [True, False, False]


def test_evento_depois_do_fim_da_serie_levanta_erro(sim):
    with pytest.raises(ValueError, match="depois do fim"):
        gerar_movimento(sim, "P001", INICIO, duracao_min=10, eventos=[ev("saida_do_leito", 30, 60)])


def test_evento_invalido_no_yaml():
    with pytest.raises(ValueError):
        EventoMovimentoCfg(tipo="voar", inicio_min=1, duracao_s=1)
    with pytest.raises(ValueError):
        EventoMovimentoCfg(tipo="saida_do_leito", inicio_min=1, duracao_s=0)
    with pytest.raises(ValueError):
        EventoMovimentoCfg(tipo="falso_quadro", inicio_min=1, duracao_s=2, oscila=True)


def test_rotulos_marcam_anomalia_e_distrator(sim):
    _, rotulos = gerar_cenario_movimento(sim, "movimento_demo", "P001", INICIO)
    assert {(r.tipo, r.anomalia) for r in rotulos} == {
        ("saida_do_leito", True),
        ("falso_quadro", False),
        ("perda_deteccao", False),
        ("virar_no_leito", False),
        ("imobilidade_prolongada", True),
    }
