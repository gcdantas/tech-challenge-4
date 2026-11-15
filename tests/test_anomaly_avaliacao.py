"""Testes da avaliação (casamento de Findings com rótulos, métricas, sorteio de cenários) e dos gráficos."""
from datetime import datetime, timedelta

import pytest

from contracts.finding import DEMO_TZ
from modules.anomaly.avaliacao import (
    _tarefa,
    atraso_minimo_s,
    calcular_metricas,
    casar,
    montar_tarefas,
    reduzir,
    relatorio_markdown,
    resumir,
    sortear_cenario_prescricoes,
    sortear_eventos_movimento,
    sortear_injecoes_vitais,
    wilson,
    _rng,
)
from modules.anomaly.common import novo_finding
from modules.anomaly.config import carregar_config, carregar_config_avaliacao, carregar_config_simulador
from modules.anomaly.prescription import analisar_eventos
from modules.anomaly.simulator import Rotulo, gerar_prescricoes

T0 = datetime(2026, 11, 3, 10, 0, tzinfo=DEMO_TZ)


@pytest.fixture(scope="module")
def cfg():
    return carregar_config()


@pytest.fixture(scope="module")
def sim():
    return carregar_config_simulador()


@pytest.fixture(scope="module")
def av():
    return carregar_config_avaliacao()


def finding(fonte, tipo, inicio_s, fim_s=None):
    return novo_finding(
        patient_id="P001", source=fonte, tipo=tipo, inicio=T0 + timedelta(seconds=inicio_s),
        fim=T0 + timedelta(seconds=inicio_s if fim_s is None else fim_s), score=0.5, severidade="medium",
        descricao="teste", evidencia={}, modelo="teste", versao="1",
    )


def rotulo(fonte, tipo, inicio_s, fim_s, anomalia=True, **detalhes):
    return Rotulo(fonte, tipo, T0 + timedelta(seconds=inicio_s), T0 + timedelta(seconds=fim_s), anomalia, detalhes)


# ---------------------------------------------------------------- estatística


def test_wilson():
    assert wilson(0, 0) is None
    baixo, alto = wilson(0, 10)
    assert baixo == 0 and 0.2 < alto < 0.35
    baixo, alto = wilson(10, 10)
    assert alto == 1 and 0.65 < baixo < 0.8
    baixo, alto = wilson(50, 100)
    assert baixo < 0.5 < alto and alto - baixo < 0.21  # mais pacientes, intervalo mais estreito


# ---------------------------------------------------------------- casamento


def test_vitais_casa_por_tempo_em_qualquer_tipo(cfg):
    r = rotulo("vitals", "dessaturacao", 600, 780, magnitude=8)
    eventos, achados = casar(cfg, [finding("vitals", "desvio_fc", 640)], [r], 120, "P001")
    assert eventos[0]["detectado"] and eventos[0]["tipo_correto"] is False
    assert eventos[0]["atraso_s"] == 40
    assert achados[0]["verdadeiro"]


def test_finding_fora_da_janela_e_falso_e_distrator_e_contado_a_parte(cfg):
    anomalia = rotulo("vitals", "dessaturacao", 600, 780)
    distrator = rotulo("vitals", "artefato_pico", 3000, 3020, anomalia=False)
    fs = [finding("vitals", "dessaturacao", 2000), finding("vitals", "dessaturacao", 3010)]
    eventos, achados = casar(cfg, fs, [anomalia, distrator], 120, "P001")
    assert not eventos[0]["detectado"] and eventos[0]["atraso_s"] is None
    assert [a["verdadeiro"] for a in achados] == [False, False]
    assert [a["em_distrator"] for a in achados] == [False, True]


def test_tolerancia_depois_do_fim_do_rotulo(cfg):
    r = rotulo("vitals", "dessaturacao", 600, 780)
    assert casar(cfg, [finding("vitals", "dessaturacao", 880)], [r], 120, "P")[0][0]["detectado"]
    assert not casar(cfg, [finding("vitals", "dessaturacao", 920)], [r], 120, "P")[0][0]["detectado"]


def test_outra_fonte_nao_casa(cfg):
    r = rotulo("vitals", "dessaturacao", 600, 780)
    assert not casar(cfg, [finding("movement", "saida_do_leito", 650)], [r], 120, "P")[0][0]["detectado"]


def test_movimentacao_e_prescricao_exigem_o_mesmo_tipo(cfg):
    rm = rotulo("movement", "saida_do_leito", 600, 700)
    assert not casar(cfg, [finding("movement", "imobilidade_prolongada", 610)], [rm], 120, "P")[0][0]["detectado"]
    assert casar(cfg, [finding("movement", "saida_do_leito", 600, 603)], [rm], 120, "P")[0][0]["detectado"]
    rp = rotulo("prescription", "dose_duplicada", 1200, 1200)
    eventos, achados = casar(cfg, [finding("prescription", "dose_fora_da_faixa", 1200)], [rp], 5, "P")
    assert not eventos[0]["detectado"] and not achados[0]["verdadeiro"]


def test_atraso_desconta_a_espera_da_propria_regra(cfg):
    limiar = cfg.movimento.imobilidade.limiar_s
    r = rotulo("movement", "imobilidade_prolongada", 0, 4 * 3600)
    eventos, _ = casar(cfg, [finding("movement", "imobilidade_prolongada", 0, limiar + 90)], [r], 120, "P")
    assert eventos[0]["atraso_s"] == 90
    assert atraso_minimo_s(cfg, "suspensao_abrupta") == 1800
    assert atraso_minimo_s(cfg, "dessaturacao") == 0


def test_dois_findings_do_mesmo_evento_contam_um_evento_detectado(cfg):
    r = rotulo("vitals", "dessaturacao", 600, 780)
    eventos, achados = casar(cfg, [finding("vitals", "dessaturacao", 640), finding("vitals", "dessaturacao", 700)], [r], 120, "P")
    assert len(eventos) == 1 and eventos[0]["atraso_s"] == 40 and all(a["verdadeiro"] for a in achados)


# ---------------------------------------------------------------- métricas


def test_resumir_calcula_recall_precisao_e_f1():
    eventos = [
        {"tipo": "a", "detectado": True, "atraso_s": 10.0, "tipo_correto": True},
        {"tipo": "a", "detectado": True, "atraso_s": 30.0, "tipo_correto": True},
        {"tipo": "b", "detectado": False, "atraso_s": None, "tipo_correto": None},
        {"tipo": "b", "detectado": False, "atraso_s": None, "tipo_correto": None},
    ]
    achados = [
        {"verdadeiro": True, "em_distrator": False}, {"verdadeiro": True, "em_distrator": False},
        {"verdadeiro": False, "em_distrator": True}, {"verdadeiro": False, "em_distrator": False},
    ]
    m = resumir(eventos, achados, horas=2)
    assert (m["recall"], m["precisao"], m["f1"]) == (0.5, 0.5, 0.5)
    assert m["falsos_alarmes_por_hora"] == 1.0
    assert (m["falsos_em_distrator"], m["falsos_espontaneos"]) == (1, 1)
    assert m["atraso"]["mediana_s"] == 20.0
    assert m["por_tipo"]["a"]["recall"] == 1.0 and m["por_tipo"]["b"]["atraso"] is None


def test_resumir_sem_eventos_nem_findings_nao_quebra():
    m = resumir([], [], horas=1)
    assert m["recall"] is None and m["precisao"] is None and m["f1"] is None and m["atraso"] is None


# ---------------------------------------------------------------- sorteio dos cenários


def test_injecoes_vitais_nao_se_sobrepoem_e_sao_reproduziveis(av):
    for i in range(30):
        a = sortear_injecoes_vitais(av, _rng(av.semente, f"x{i}"), com_eventos=True)
        b = sortear_injecoes_vitais(av, _rng(av.semente, f"x{i}"), com_eventos=True)
        assert a == b
        fim_anterior = -1.0
        for inj in a:
            assert inj.inicio_min >= av.vitais.inicio_min
            assert inj.inicio_min + inj.duracao_min <= av.vitais.duracao_min
            assert inj.inicio_min >= fim_anterior
            fim_anterior = inj.inicio_min + inj.duracao_min + av.vitais.folga_min


def test_paciente_estavel_so_recebe_distratores(av):
    for i in range(30):
        injecoes = sortear_injecoes_vitais(av, _rng(av.semente, f"s{i}"), com_eventos=False)
        assert injecoes and all(inj.tipo in av.vitais.distratores for inj in injecoes)


def test_cenario_de_prescricoes_sorteado_e_coerente_com_o_detector(cfg, sim, av):
    """Em 40 pacientes sorteados, o detector acha exatamente os eventos marcados como anomalia."""
    total = 0
    for i in range(40):
        cenario = sortear_cenario_prescricoes(cfg, av, _rng(av.semente, f"p{i}"))
        eventos, rotulos = gerar_prescricoes(sim, f"p{i}", T0, cenario)
        achados = analisar_eventos(cfg, eventos, ate=T0 + timedelta(minutes=av.prescricoes.duracao_min))
        assert sorted(f.type for f in achados) == sorted(r.tipo for r in rotulos), f"paciente p{i}"
        total += len(rotulos)
    assert total > 20  # o sorteio gera anomalias de verdade


def test_eventos_de_movimento_respeitam_a_duracao_e_o_limiar(cfg, av):
    for tipo in ("saida", "imobilidade", "estavel"):
        for i in range(20):
            eventos, duracao = sortear_eventos_movimento(av, _rng(av.semente, f"{tipo}{i}"), tipo)
            assert all(e.inicio_min * 60 + e.duracao_s <= duracao * 60 for e in eventos)
            assert all(e.tipo != "imobilidade" or e.duracao_s >= cfg.movimento.imobilidade.limiar_s for e in eventos)
            if tipo == "estavel":
                assert all(e.tipo not in ("saida_do_leito", "imobilidade") for e in eventos)


def test_reduzir_diminui_as_quantidades(av):
    pequeno = reduzir(av, 5)
    assert len(montar_tarefas(pequeno)) < len(montar_tarefas(av)) / 3
    assert pequeno.semente == av.semente


# ---------------------------------------------------------------- tarefas e relatório


def test_tarefas_de_prescricao_e_movimento():
    for tarefa, saida in [
        (("prescricoes", "AV-PR-T01", None), "prescricoes"),
        (("movimento", "AV-MS-T01", "saida"), "movimentacao"),
        (("movimento", "AV-MI-T01", "imobilidade"), "movimentacao"),
        (("movimento", "AV-ME-T01", "estavel"), "movimentacao"),
    ]:
        r = _tarefa(tarefa)
        eventos, achados = r["saidas"][saida]
        assert r["horas"] > 0
        assert all(e["detectado"] for e in eventos)  # as regras acham o que foi desenhado
        assert all(a["verdadeiro"] for a in achados)  # e não inventam nada


def test_tarefa_de_vitais_devolve_as_tres_variantes():
    r = _tarefa(("vitais_eventos", "AV-VE-T01", None))
    assert set(r["saidas"]) == {"vitais_sinais", "vitais_if", "vitais_completo"}
    assert len(r["saidas"]["vitais_completo"][1]) >= len(r["saidas"]["vitais_sinais"][1])


def test_metricas_relatorio_e_graficos(av, tmp_path):
    pytest.importorskip("matplotlib")
    from modules.anomaly.graficos import gerar_todos

    tarefas = [
        ("vitais_eventos", "AV-VE-T02", None),
        ("vitais_estavel", "AV-VS-T02", None),
        ("varredura", "AV-SW-dessaturacao-4-T", ("dessaturacao", 4)),
        ("varredura", "AV-SW-dessaturacao-10-T", ("dessaturacao", 10)),
        ("prescricoes", "AV-PR-T02", None),
        ("movimento", "AV-MS-T02", "saida"),
        ("movimento", "AV-MI-T02", "imobilidade"),
    ]
    metricas = calcular_metricas([_tarefa(t) for t in tarefas], av)
    assert set(metricas["modulos"]) == {"vitais_sinais", "vitais_if", "vitais_completo", "prescricoes", "movimentacao"}
    assert metricas["modulos"]["movimentacao"]["recall"] == 1.0

    texto = relatorio_markdown(metricas)
    assert "Dados sintéticos" in texto and "| Prescrições (regras) |" in texto

    # os gráficos de métricas e de sensibilidade esperam todos os tipos; completa os que a amostra pequena não cobriu
    for variante in metricas["sensibilidade_vitais"].values():
        for tipo in av.vitais.varredura.magnitudes:
            variante.setdefault(tipo, [{"magnitude": 1, "pacientes": 1, "recall": 0.0}])
    caminhos = gerar_todos(metricas, tmp_path)
    assert len(caminhos) == 6 and all(c.exists() and c.stat().st_size > 5000 for c in caminhos)
