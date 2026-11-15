"""Testes da classificação por limites e do rastreador de episódios."""
import pytest

from modules.anomaly.common import classificar
from modules.anomaly.episodios import RastreadorEpisodio

SPO2 = [(94.0, "medium"), (90.0, "high"), (85.0, "critical")]
FC_ALTA = [(110.0, "medium"), (130.0, "high"), (150.0, "critical")]


def test_classificar_valor_normal_nao_cruza_limite():
    assert classificar(97, SPO2, "menor") is None
    assert classificar(80, FC_ALTA, "maior") is None


def test_classificar_limite_vale_com_igualdade():
    severidade, score, limite = classificar(94, SPO2, "menor")
    assert (severidade, limite) == ("medium", 94.0)
    assert score == pytest.approx(0.50)


def test_classificar_score_sobe_entre_os_limites():
    # 87 % está a 60% do caminho entre o limite high (90) e o critical (85)
    severidade, score, limite = classificar(87, SPO2, "menor")
    assert (severidade, limite) == ("high", 90.0)
    assert score == pytest.approx(0.75 + (0.95 - 0.75) * 0.6)


def test_classificar_alem_do_ultimo_limite_fica_critico():
    assert classificar(70, SPO2, "menor")[:2] == ("critical", 0.95)
    assert classificar(200, FC_ALTA, "maior")[:2] == ("critical", 0.95)


def test_classificar_nan_nao_classifica():
    assert classificar(float("nan"), SPO2, "menor") is None


def novo(clinico_s=30, estatistico_s=90, recuperacao_s=30):
    return RastreadorEpisodio("spo2", clinico_s, estatistico_s, recuperacao_s)


def rodar(rastreador, trechos, **kw):
    """`trechos` é uma lista de (t, ativo, rank). Devolve {t: eventos} só dos que tiveram evento."""
    saida = {}
    for t, ativo, rank in trechos:
        eventos = rastreador.atualizar(t, ativo, rank=rank, score=0.5, tipo="dessaturacao", clinico=True, **kw)
        if eventos:
            saida[t] = eventos
    return saida


def test_episodio_curto_e_descartado_como_artefato():
    r = novo()
    eventos = rodar(r, [(0, True, 3), (10, True, 3), (20, True, 3)] + [(30 + 10 * k, False, 0) for k in range(5)])
    assert "emitir" not in [e for ev in eventos.values() for e in ev]
    assert r.todos_emitidos() == []


def test_episodio_longo_emite_uma_vez_so():
    r = novo()
    eventos = rodar(r, [(10 * k, True, 2) for k in range(20)])
    assert eventos == {30: ["emitir"]}  # só quando completa 30 s, e nunca mais


def test_piora_da_severidade_escala():
    r = novo()
    trechos = [(10 * k, True, 1) for k in range(6)] + [(10 * k, True, 3) for k in range(6, 12)]
    assert rodar(r, trechos) == {30: ["emitir"], 60: ["escalar"]}


def test_uma_amostra_normal_no_meio_nao_parte_o_episodio():
    r = novo()
    trechos = [(0, True, 2), (10, True, 2), (20, False, 0), (30, True, 2), (40, True, 2)]
    assert rodar(r, trechos) == {30: ["emitir"]}
    assert r.atual.inicio == 0


def test_recuperacao_encerra_o_episodio_e_o_proximo_e_novo():
    r = novo()
    trechos = [(10 * k, True, 2) for k in range(5)]  # 0 a 40 s
    trechos += [(50, False, 0), (60, False, 0), (70, False, 0), (80, False, 0)]  # 30 s normais seguidos
    trechos += [(90 + 10 * k, True, 2) for k in range(5)]  # outro episódio
    eventos = rodar(r, trechos)
    assert eventos[80] == ["encerrar"]
    assert eventos[120] == ["emitir"]  # 90 + 30 s
    assert len(r.encerrados) == 1 and len(r.todos_emitidos()) == 2


def test_so_estatistico_exige_mais_tempo():
    r = novo()
    eventos = {}
    for k in range(12):
        ev = r.atualizar(10 * k, True, rank=0, score=0.3, tipo="desvio_spo2", clinico=False)
        if ev:
            eventos[10 * k] = ev
    assert eventos == {90: ["emitir"]}


def test_pode_emitir_falso_segura_o_finding_sem_perder_o_episodio():
    r = novo()
    for k in range(10):
        assert r.atualizar(10 * k, True, rank=1, score=0.5, tipo="x", clinico=True, pode_emitir=False) == []
    # liberou: emite na hora, porque o episódio já passou da duração mínima
    assert r.atualizar(100, True, rank=1, score=0.5, tipo="x", clinico=True) == ["emitir"]


def test_tipo_clinico_vale_mais_que_o_estatistico():
    r = novo()
    r.atualizar(0, True, rank=0, score=0.3, tipo="desvio_spo2", clinico=False)
    r.atualizar(10, True, rank=1, score=0.5, tipo="dessaturacao", clinico=True)
    r.atualizar(20, True, rank=0, score=0.3, tipo="desvio_spo2", clinico=False)
    assert r.atual.tipo == "dessaturacao"