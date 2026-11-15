"""Avaliação dos detectores de anomalias: precisão, recall, F1 e atraso de detecção.

Para cada fonte o script sorteia séries sintéticas com anomalias injetadas e
distratores, roda o detector e compara os Findings com os rótulos de verdade.

  recall     dos eventos injetados, quantos o detector achou
  precisão   dos Findings emitidos, quantos caem em cima de um evento injetado
  atraso     do início do evento até o Finding sair (para imobilidade e suspensão de
             medicamento, depois do tempo que a própria regra precisa esperar)

Os dados são sintéticos e quem gera é o mesmo projeto que mede, então o resultado
mostra se os detectores fazem o que foi desenhado, não o desempenho clínico real.
As sementes daqui são diferentes das usadas para calibrar os limiares.

    python -m modules.anomaly.avaliacao              # avaliação completa
    python -m modules.anomaly.avaliacao --rapido     # versão pequena, para conferir se roda
"""
import argparse
import json
import math
import os
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

from contracts.finding import DEMO_TZ
from contracts.patients import REPO_ROOT
from modules.anomaly import movement
from modules.anomaly.config import (
    AnomaliaCfg,
    AvaliacaoCfg,
    CenarioPrescricoesCfg,
    EventoCenarioCfg,
    EventoMovimentoCfg,
    InjecaoCfg,
    PedidoCfg,
    carregar_config,
    carregar_config_avaliacao,
    carregar_config_simulador,
)
from modules.anomaly.multivariado import analisar_quadro_completo
from modules.anomaly.prescription import analisar_eventos, normalizar_nome
from modules.anomaly.simulator import (
    Rotulo,
    criar_rng,
    gerar_movimento,
    gerar_prescricoes,
    gerar_vitais,
    rotulos_das_injecoes,
)

REFERENCIA = datetime(2026, 11, 3, 10, 0, tzinfo=DEMO_TZ)
SAIDA_PADRAO = REPO_ROOT / "outputs" / "anomaly" / "avaliacao"
TIPO_IF = "padrao_multivariado"

# Os grupos de vitais: o que o detector de sinais acha sozinho, só o Isolation Forest, e os dois juntos
VARIANTES_VITAIS = {
    "vitais_sinais": "Vitais: limites + z-score",
    "vitais_if": "Vitais: só Isolation Forest",
    "vitais_completo": "Vitais: tudo junto",
}


# ---------------------------------------------------------------------- estatística


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Intervalo de confiança de 95 % de uma proporção k/n (Wilson). Com n pequeno ele é largo, e é para ser."""
    if n == 0:
        return None
    p = k / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    meia = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centro - meia), min(1.0, centro + meia)


def _razao(k: int, n: int) -> float | None:
    return None if n == 0 else k / n


def _f1(precisao: float | None, recall: float | None) -> float | None:
    if precisao is None or recall is None or precisao + recall == 0:
        return None if precisao is None or recall is None else 0.0
    return 2 * precisao * recall / (precisao + recall)


# ---------------------------------------------------------------------- casamento Finding x rótulo


def atraso_minimo_s(cfg: AnomaliaCfg, tipo: str) -> float:
    """Tempo que a regra espera por desenho. Não conta como atraso do detector."""
    if tipo == "imobilidade_prolongada":
        return cfg.movimento.imobilidade.limiar_s
    if tipo == "suspensao_abrupta":
        return cfg.prescricoes.suspensao.janela_substituicao_h * 3600
    return 0.0


def _casa(f, r: Rotulo, tolerancia: timedelta) -> bool:
    """O Finding `f` é sobre o evento `r`? Mesma fonte e janelas que se tocam. Fora dos vitais, o tipo também precisa bater."""
    if f.source != r.fonte:
        return False
    if f.timestamp_start > r.fim + tolerancia or f.timestamp_end < r.inicio:
        return False
    return r.fonte == "vitals" or f.type == r.tipo


def casar(cfg: AnomaliaCfg, findings: list, rotulos: list[Rotulo], tolerancia_s: float, paciente: str):
    """Compara Findings e rótulos de um paciente. Devolve (eventos, achados) como dicionários simples."""
    tol = timedelta(seconds=tolerancia_s)
    anomalias = [r for r in rotulos if r.anomalia]
    distratores = [r for r in rotulos if not r.anomalia]

    eventos = []
    for r in anomalias:
        cand = sorted((f for f in findings if _casa(f, r, tol)), key=lambda f: f.timestamp_end)
        atraso = None
        if cand:
            atraso = (cand[0].timestamp_end - r.inicio).total_seconds() - atraso_minimo_s(cfg, r.tipo)
        eventos.append(
            {
                "paciente": paciente,
                "fonte": r.fonte,
                "tipo": r.tipo,
                "magnitude": r.detalhes.get("magnitude"),
                "detectado": bool(cand),
                "atraso_s": None if atraso is None else round(max(atraso, 0.0), 1),
                "tipo_correto": any(f.type == r.tipo for f in cand) if cand else None,
            }
        )

    achados = []
    for f in findings:
        verdadeiro = any(_casa(f, r, tol) for r in anomalias)
        achados.append(
            {
                "paciente": paciente,
                "fonte": f.source,
                "tipo": f.type,
                "severidade": f.severity,
                "verdadeiro": verdadeiro,
                "em_distrator": (not verdadeiro) and any(_casa(f, r, tol) for r in distratores),
            }
        )
    return eventos, achados


# ---------------------------------------------------------------------- sorteio dos cenários


def _rng(semente: int, nome: str) -> np.random.Generator:
    return np.random.default_rng([semente, zlib.crc32(nome.encode("utf-8"))])


def _sortear_posicoes(rng, duracoes_min: list[float], inicio_min: float, total_min: float, folga_min: float) -> list[float | None]:
    """Início de cada evento (em minutos) sem sobreposição, com `folga_min` de espaço. None se não coube."""
    ocupados: list[tuple[float, float]] = []
    inicios: list[float | None] = []
    for dur in duracoes_min:
        escolhido = None
        for _ in range(80):
            ini = rng.uniform(inicio_min, max(inicio_min, total_min - dur - folga_min))
            if all(ini + dur + folga_min <= a or ini >= b + folga_min for a, b in ocupados):
                escolhido = ini
                break
        if escolhido is not None:
            ocupados.append((escolhido, escolhido + dur))
        inicios.append(escolhido)
    return inicios


def _uniforme(rng, intervalo) -> float:
    return float(rng.uniform(intervalo[0], intervalo[1]))


def sortear_injecoes_vitais(av: AvaliacaoCfg, rng, com_eventos: bool) -> list[InjecaoCfg]:
    v = av.vitais
    n_ev = int(rng.integers(v.eventos_por_paciente[0], v.eventos_por_paciente[1] + 1)) if com_eventos else 0
    n_di = int(rng.integers(v.distratores_por_paciente[0], v.distratores_por_paciente[1] + 1))
    modelos: list[tuple[str, object]] = [
        (nome, v.tipos[nome]) for nome in rng.choice(list(v.tipos), size=n_ev, replace=True)
    ] + [(nome, v.distratores[nome]) for nome in rng.choice(list(v.distratores), size=n_di, replace=True)]
    duracoes = [_uniforme(rng, m.duracao_min) for _, m in modelos]
    inicios = _sortear_posicoes(rng, duracoes, v.inicio_min, v.duracao_min, v.folga_min)
    injecoes = []
    for (nome, m), dur, ini in zip(modelos, duracoes, inicios):
        if ini is None:
            continue
        injecoes.append(
            InjecaoCfg(
                tipo=nome,
                inicio_min=round(ini, 2),
                duracao_min=round(dur, 2),
                magnitude=round(_uniforme(rng, m.magnitude), 2) if nome not in ("perda_sinal", "leitura_invalida") else 0,
                rampa_s=m.rampa_s,
            )
        )
    return sorted(injecoes, key=lambda i: i.inicio_min)


def _dose_normal(cat, rng) -> tuple[float, float]:
    """Dose por tomada e número de tomadas dentro da faixa do catálogo, com folga na dose diária."""
    for vezes in (int(x) for x in rng.permutation([1, 2, 3, 4])):
        dose = round(_uniforme(rng, (cat.dose_min_mg, cat.dose_max_mg)) / 5) * 5 or cat.dose_min_mg
        if cat.dose_min_mg <= dose <= cat.dose_max_mg and dose * vezes <= 0.9 * cat.dose_diaria_max_mg:
            return float(dose), float(vezes)
    return float(cat.dose_min_mg), 1.0


def sortear_cenario_prescricoes(cfg: AnomaliaCfg, av: AvaliacaoCfg, rng) -> CenarioPrescricoesCfg:
    """Pedidos de admissão normais e uma mistura de eventos normais e anômalos, cada um num medicamento diferente."""
    catalogo = cfg.prescricoes.medicamentos
    chaves = list(catalogo)
    iniciais = [str(k) for k in rng.choice(chaves, size=4, replace=False)]
    fora = [k for k in chaves if k not in iniciais]
    doses = {k: _dose_normal(catalogo[k], rng) for k in iniciais}
    pedidos = [PedidoCfg(medicamento=catalogo[k].nome, dose_mg=doses[k][0], doses_por_dia=doses[k][1]) for k in iniciais]

    n_ev = int(rng.integers(av.prescricoes.eventos_por_paciente[0], av.prescricoes.eventos_por_paciente[1] + 1))
    tipos = [str(t) for t in rng.choice(
        ["acima", "abaixo", "duplicada", "aumento", "suspensao", "renovacao", "nova", "suspende_pontual", "ajuste_pequeno"],
        size=n_ev,
    )]
    minutos = np.sort(rng.uniform(15, av.prescricoes.duracao_min - 40, size=n_ev))
    # um medicamento só por evento, para um não mexer no resultado do outro
    livres_iniciais, livres_fora = list(iniciais), list(fora)
    rng.shuffle(livres_iniciais)
    rng.shuffle(livres_fora)
    eventos: list[EventoCenarioCfg] = []

    def prescrever(minuto, k, dose, vezes, tipo_esperado=None):
        return EventoCenarioCfg(
            acao="prescrever", minuto=round(float(minuto), 1), medicamento=catalogo[k].nome, dose_mg=float(dose),
            doses_por_dia=float(vezes), anomalia=tipo_esperado is not None, tipo_esperado=tipo_esperado,
        )

    def suspender(minuto, k, tipo_esperado=None):
        return EventoCenarioCfg(
            acao="suspender", minuto=round(float(minuto), 1), medicamento=catalogo[k].nome,
            anomalia=tipo_esperado is not None, tipo_esperado=tipo_esperado,
        )

    for minuto, tipo in zip(minutos, tipos):
        if tipo in ("acima", "abaixo", "nova"):
            if not livres_fora:
                continue
            k = livres_fora.pop()
            cat = catalogo[k]
            if tipo == "acima":
                eventos.append(prescrever(minuto, k, round(cat.dose_max_mg * _uniforme(rng, (1.2, 10))), 1, "dose_fora_da_faixa"))
            elif tipo == "abaixo":
                eventos.append(prescrever(minuto, k, max(1, round(cat.dose_min_mg * _uniforme(rng, (0.15, 0.45)))), 1, "dose_fora_da_faixa"))
            else:
                dose, vezes = _dose_normal(cat, rng)
                eventos.append(prescrever(minuto, k, dose, vezes))
            continue

        if not livres_iniciais:
            continue
        if tipo == "suspensao":
            continuos = [k for k in livres_iniciais if catalogo[k].continuo]
            if not continuos:
                continue
            k = continuos[0]
            livres_iniciais.remove(k)
            eventos.append(suspender(minuto, k, "suspensao_abrupta"))
        elif tipo == "suspende_pontual":
            nao_continuos = [k for k in livres_iniciais if not catalogo[k].continuo]
            if not nao_continuos:
                continue
            k = nao_continuos[0]
            livres_iniciais.remove(k)
            eventos.append(suspender(minuto, k))  # medicamento que não é contínuo: suspender é normal
        else:
            k = livres_iniciais.pop()
            cat, (dose0, vezes0) = catalogo[k], doses[k]
            if tipo == "duplicada":
                dose, vezes = _dose_normal(cat, rng)
                eventos.append(prescrever(minuto, k, dose, vezes, "dose_duplicada"))
            elif tipo == "renovacao":
                eventos += [suspender(minuto, k), prescrever(minuto, k, dose0, vezes0)]
            else:  # aumento (anômalo, acima de 50 %) ou ajuste pequeno (normal, abaixo de 50 %)
                faixa = (1.7, 4.0) if tipo == "aumento" else (1.05, 1.3)
                novo = None
                for _ in range(30):
                    f = _uniforme(rng, faixa)
                    for dose, vezes in ((round(dose0 * f), vezes0), (dose0, math.ceil(vezes0 * f))):
                        pct = (dose * vezes / (dose0 * vezes0) - 1) * 100
                        dentro = dose <= cat.dose_max_mg and dose * vezes <= cat.dose_diaria_max_mg and vezes <= 6
                        alvo = pct >= 55 if tipo == "aumento" else 0 < pct <= 35
                        if dentro and alvo:
                            novo = (dose, vezes)
                            break
                    if novo:
                        break
                if novo is None:
                    continue  # esta dose não deixa espaço para o aumento: não sorteia o evento
                eventos += [
                    suspender(minuto, k),
                    prescrever(minuto, k, novo[0], novo[1], "aumento_abrupto_dose" if tipo == "aumento" else None),
                ]
    return CenarioPrescricoesCfg(pedidos_iniciais=pedidos, eventos=eventos)


def sortear_eventos_movimento(av: AvaliacaoCfg, rng, tipo_paciente: str) -> tuple[list[EventoMovimentoCfg], float]:
    """Eventos de movimentação de um paciente e a duração da série em minutos."""
    m = av.movimento
    duracao = m.duracao_imobilidade_min if tipo_paciente == "imobilidade" else m.duracao_saida_min
    n_di = int(rng.integers(m.distratores_por_paciente[0], m.distratores_por_paciente[1] + 1))
    modelos: list[tuple[str, float, bool]] = []  # tipo, duração em min, oscila
    if tipo_paciente == "saida":
        n_sa = int(rng.integers(m.saidas_por_paciente[0], m.saidas_por_paciente[1] + 1))
        for _ in range(n_sa):
            dur_s = _uniforme(rng, m.saida_duracao_s)
            modelos.append(("saida_do_leito", dur_s / 60, bool(dur_s > 60 and rng.random() < 0.5)))
    elif tipo_paciente == "imobilidade":
        modelos.append(("imobilidade", _uniforme(rng, m.imobilidade_duracao_min), False))
    for nome in rng.choice(["falso_quadro", "perda_deteccao", "virar_no_leito"], size=n_di):
        modelos.append((str(nome), 0.5 if nome == "virar_no_leito" else 2 / 60, False))

    # a imobilidade é sorteada primeiro e ocupa o seu lugar, os distratores se arrumam em volta
    inicios = _sortear_posicoes(rng, [d for _, d, _ in modelos], 15, duracao, m.folga_min)
    eventos = []
    for (nome, dur, oscila), ini in zip(modelos, inicios):
        if ini is None:
            continue
        duracao_s = 0 if nome == "virar_no_leito" else (dur * 60 if nome in ("saida_do_leito", "imobilidade") else float(rng.uniform(1, 2.5)))
        eventos.append(EventoMovimentoCfg(tipo=nome, inicio_min=round(ini, 2), duracao_s=round(duracao_s, 1), oscila=oscila))
    return sorted(eventos, key=lambda e: e.inicio_min), duracao


# ---------------------------------------------------------------------- uma tarefa = um paciente


@lru_cache(maxsize=1)
def _configs() -> tuple[AnomaliaCfg, object, AvaliacaoCfg]:
    return carregar_config(), carregar_config_simulador(), carregar_config_avaliacao()


def _dividir_vitais(findings: list) -> dict[str, list]:
    return {
        "vitais_sinais": [f for f in findings if f.type != TIPO_IF],
        "vitais_if": [f for f in findings if f.type == TIPO_IF],
        "vitais_completo": findings,
    }


def _tarefa(args: tuple) -> dict:
    """Roda um paciente e devolve só números e dicionários simples (precisa ser enviado entre processos)."""
    cfg, sim, av = _configs()
    tipo, nome, extra = args
    resultado: dict = {"grupo": tipo, "paciente": nome, "horas": 0.0, "saidas": {}}

    if tipo in ("vitais_eventos", "vitais_estavel", "varredura"):
        rng = _rng(av.semente, nome)
        if tipo == "varredura":
            tipo_evento, magnitude = extra
            m = av.vitais.tipos[tipo_evento]
            injecoes = [
                InjecaoCfg(
                    tipo=tipo_evento, inicio_min=av.vitais.varredura.evento_inicio_min,
                    duracao_min=sum(m.duracao_min) / 2, magnitude=magnitude, rampa_s=m.rampa_s,
                )
            ]
            duracao = av.vitais.varredura.duracao_min
        else:
            injecoes = sortear_injecoes_vitais(av, rng, com_eventos=tipo == "vitais_eventos")
            duracao = av.vitais.duracao_min
        quadro = gerar_vitais(
            sim, nome, REFERENCIA, rng=criar_rng(av.semente, nome), duracao_min=duracao, variar_paciente=True, injecoes=injecoes
        )
        rotulos = rotulos_das_injecoes(REFERENCIA, injecoes)
        achados = analisar_quadro_completo(cfg, quadro)
        resultado["horas"] = duracao / 60
        for variante, fs in _dividir_vitais(achados).items():
            resultado["saidas"][variante] = casar(cfg, fs, rotulos, av.tolerancia_s["vitals"], nome)

    elif tipo == "prescricoes":
        rng = _rng(av.semente, nome)
        cenario = sortear_cenario_prescricoes(cfg, av, rng)
        eventos, rotulos = gerar_prescricoes(sim, nome, REFERENCIA, cenario)
        ate = REFERENCIA + timedelta(minutes=av.prescricoes.duracao_min)
        achados = analisar_eventos(cfg, eventos, ate=ate)
        resultado["horas"] = av.prescricoes.duracao_min / 60
        resultado["saidas"]["prescricoes"] = casar(cfg, achados, rotulos, av.tolerancia_s["prescription"], nome)

    elif tipo == "movimento":
        rng = _rng(av.semente, nome)
        tipo_paciente = extra
        eventos_mov, duracao = sortear_eventos_movimento(av, rng, tipo_paciente)
        quadro, rotulos = gerar_movimento(
            sim, nome, REFERENCIA, rng=criar_rng(av.semente, nome), duracao_min=duracao, eventos=eventos_mov
        )
        achados = movement.analisar_quadro(cfg, quadro)
        resultado["horas"] = duracao / 60
        resultado["saidas"]["movimentacao"] = casar(cfg, achados, rotulos, av.tolerancia_s["movement"], nome)
    else:
        raise ValueError(f"tarefa desconhecida: {tipo}")
    return resultado


def montar_tarefas(av: AvaliacaoCfg) -> list[tuple]:
    t: list[tuple] = []
    t += [("vitais_eventos", f"AV-VE-{i:03d}", None) for i in range(av.vitais.pacientes_com_eventos)]
    t += [("vitais_estavel", f"AV-VS-{i:03d}", None) for i in range(av.vitais.pacientes_estaveis)]
    for tipo, magnitudes in av.vitais.varredura.magnitudes.items():
        for m in magnitudes:
            t += [("varredura", f"AV-SW-{tipo}-{m}-{i:02d}", (tipo, m)) for i in range(av.vitais.varredura.pacientes_por_ponto)]
    t += [("prescricoes", f"AV-PR-{i:03d}", None) for i in range(av.prescricoes.pacientes)]
    t += [("movimento", f"AV-MS-{i:03d}", "saida") for i in range(av.movimento.pacientes_saida)]
    t += [("movimento", f"AV-MI-{i:03d}", "imobilidade") for i in range(av.movimento.pacientes_imobilidade)]
    t += [("movimento", f"AV-ME-{i:03d}", "estavel") for i in range(av.movimento.pacientes_estaveis)]
    return t


def reduzir(av: AvaliacaoCfg, divisor: int) -> AvaliacaoCfg:
    """Versão pequena da avaliação (para conferir se tudo roda). Divide as quantidades de pacientes."""
    d = av.model_dump()
    pequeno = lambda n: max(1, n // divisor)  # noqa: E731
    for chave in ("pacientes_com_eventos", "pacientes_estaveis"):
        d["vitais"][chave] = pequeno(d["vitais"][chave])
    d["vitais"]["varredura"]["pacientes_por_ponto"] = pequeno(d["vitais"]["varredura"]["pacientes_por_ponto"])
    d["vitais"]["varredura"]["magnitudes"] = {k: [v[0], v[-1]] for k, v in d["vitais"]["varredura"]["magnitudes"].items()}
    d["prescricoes"]["pacientes"] = pequeno(d["prescricoes"]["pacientes"])
    for chave in ("pacientes_saida", "pacientes_imobilidade", "pacientes_estaveis"):
        d["movimento"][chave] = pequeno(d["movimento"][chave])
    return AvaliacaoCfg.model_validate(d)


def executar_tarefas(tarefas: list[tuple], processos: int, avisar=print) -> list[dict]:
    if processos <= 1:
        saidas = []
        for k, t in enumerate(tarefas, 1):
            saidas.append(_tarefa(t))
            if k % 20 == 0:
                avisar(f"  {k}/{len(tarefas)} pacientes")
        return saidas
    saidas = []
    with ProcessPoolExecutor(max_workers=processos) as pool:
        for k, r in enumerate(pool.map(_tarefa, tarefas, chunksize=2), 1):
            saidas.append(r)
            if k % 40 == 0:
                avisar(f"  {k}/{len(tarefas)} pacientes")
    return saidas


# ---------------------------------------------------------------------- métricas


def _atraso(valores: list[float]) -> dict | None:
    if not valores:
        return None
    a = np.array(valores)
    return {
        "mediana_s": round(float(np.median(a)), 1),
        "p90_s": round(float(np.percentile(a, 90)), 1),
        "maximo_s": round(float(a.max()), 1),
    }


def resumir(eventos: list[dict], achados: list[dict], horas: float) -> dict:
    """Métricas de um grupo de eventos e Findings."""
    detectados = sum(e["detectado"] for e in eventos)
    verdadeiros = sum(a["verdadeiro"] for a in achados)
    recall, precisao = _razao(detectados, len(eventos)), _razao(verdadeiros, len(achados))
    por_tipo = {}
    tipos = defaultdict(list)
    for e in eventos:
        tipos[e["tipo"]].append(e)
    for tipo, lista in sorted(tipos.items()):
        det = [e for e in lista if e["detectado"]]
        por_tipo[tipo] = {
            "eventos": len(lista),
            "detectados": len(det),
            "recall": _razao(len(det), len(lista)),
            "atraso": _atraso([e["atraso_s"] for e in det if e["atraso_s"] is not None]),
            "tipo_correto": sum(bool(e["tipo_correto"]) for e in det),
        }
    ic_r, ic_p = wilson(detectados, len(eventos)), wilson(verdadeiros, len(achados))
    return {
        "eventos": len(eventos),
        "detectados": detectados,
        "recall": recall,
        "recall_ic95": ic_r,
        "findings": len(achados),
        "verdadeiros": verdadeiros,
        "falsos_em_distrator": sum(a["em_distrator"] for a in achados),
        "falsos_espontaneos": sum((not a["verdadeiro"]) and (not a["em_distrator"]) for a in achados),
        "precisao": precisao,
        "precisao_ic95": ic_p,
        "f1": _f1(precisao, recall),
        "horas_monitoradas": round(horas, 1),
        "falsos_alarmes_por_hora": sum(not a["verdadeiro"] for a in achados) / horas if horas else None,
        "atraso": _atraso([e["atraso_s"] for e in eventos if e["detectado"] and e["atraso_s"] is not None]),
        "por_tipo": por_tipo,
    }


def agregar(saidas: list[dict], grupos: tuple[str, ...], saida: str) -> dict:
    """Junta os resultados dos pacientes de `grupos` para a saída `saida` (um detector ou variante)."""
    eventos, achados, horas = [], [], 0.0
    for r in saidas:
        if r["grupo"] in grupos and saida in r["saidas"]:
            ev, ach = r["saidas"][saida]
            eventos += ev
            achados += ach
            horas += r["horas"]
    return resumir(eventos, achados, horas)


def sensibilidade(saidas: list[dict]) -> dict:
    """Recall por tipo e magnitude (os pacientes da varredura têm um evento cada)."""
    pontos: dict[str, dict[str, dict[float, list[bool]]]] = {v: defaultdict(lambda: defaultdict(list)) for v in VARIANTES_VITAIS}
    for r in saidas:
        if r["grupo"] != "varredura":
            continue
        for variante in VARIANTES_VITAIS:
            for e in r["saidas"][variante][0]:
                pontos[variante][e["tipo"]][e["magnitude"]].append(e["detectado"])
    return {
        variante: {
            tipo: [
                {"magnitude": m, "pacientes": len(v), "recall": sum(v) / len(v)}
                for m, v in sorted(por_mag.items())
            ]
            for tipo, por_mag in por_tipo.items()
        }
        for variante, por_tipo in pontos.items()
    }


def calcular_metricas(saidas: list[dict], av: AvaliacaoCfg) -> dict:
    vit = ("vitais_eventos", "vitais_estavel")
    modulos = {
        "vitais_sinais": agregar(saidas, vit, "vitais_sinais"),
        "vitais_if": agregar(saidas, vit, "vitais_if"),
        "vitais_completo": agregar(saidas, vit, "vitais_completo"),
        "prescricoes": agregar(saidas, ("prescricoes",), "prescricoes"),
        "movimentacao": agregar(saidas, ("movimento",), "movimentacao"),
    }
    return {
        "aviso": "Dados sintéticos. Quem gera e quem mede é o mesmo projeto: o resultado mostra se os detectores fazem o que foi desenhado, não o desempenho clínico.",
        "semente": av.semente,
        "pacientes": {g: len({r["paciente"] for r in saidas if r["grupo"] == g}) for g in sorted({r["grupo"] for r in saidas})},
        "modulos": modulos,
        "sensibilidade_vitais": sensibilidade(saidas),
    }


# ---------------------------------------------------------------------- relatório em texto


NOMES = {**VARIANTES_VITAIS, "prescricoes": "Prescrições (regras)", "movimentacao": "Movimentação (regras)"}


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f} %".replace(".", ",")


def _ic(x) -> str:
    return "" if x is None else f" ({100 * x[0]:.0f}–{100 * x[1]:.0f})"


def _tempo(segundos: float | None) -> str:
    if segundos is None:
        return "–"
    if segundos < 120:
        return f"{segundos:.0f} s"
    return f"{segundos / 60:.1f} min".replace(".", ",")


def tabela_markdown(metricas: dict) -> str:
    linhas = [
        "| Módulo | Eventos | Recall (IC 95 %) | Findings | Precisão (IC 95 %) | F1 | Falsos alarmes por hora | Atraso mediano (p90) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for chave, m in metricas["modulos"].items():
        atraso = m["atraso"]
        fa = m["falsos_alarmes_por_hora"]
        linhas.append(
            f"| {NOMES[chave]} | {m['detectados']}/{m['eventos']} | {_pct(m['recall'])}{_ic(m['recall_ic95'])} "
            f"| {m['verdadeiros']}/{m['findings']} | {_pct(m['precisao'])}{_ic(m['precisao_ic95'])} "
            f"| {_pct(m['f1'])} | {'–' if fa is None else f'{fa:.2f}'.replace('.', ',')} "
            f"| {_tempo(atraso['mediana_s']) + ' (' + _tempo(atraso['p90_s']) + ')' if atraso else '–'} |"
        )
    linhas += ["", "Por tipo de evento:", "", "| Módulo | Tipo | Eventos | Recall | Atraso mediano |", "|---|---|---:|---:|---:|"]
    for chave, m in metricas["modulos"].items():
        for tipo, t in m["por_tipo"].items():
            linhas.append(
                f"| {NOMES[chave]} | {tipo} | {t['detectados']}/{t['eventos']} | {_pct(t['recall'])} "
                f"| {_tempo(t['atraso']['mediana_s']) if t['atraso'] else '–'} |"
            )
    return "\n".join(linhas) + "\n"


def relatorio_markdown(metricas: dict) -> str:
    m = metricas
    por_fonte = " | ".join(f"{k}: {v}" for k, v in m["pacientes"].items())
    falsos = []
    for chave, mod in m["modulos"].items():
        falsos.append(
            f"- {NOMES[chave]}: {mod['falsos_espontaneos']} sem causa aparente, {mod['falsos_em_distrator']} em cima de distrator"
        )
    return (
        "# Avaliação dos detectores de anomalias\n\n"
        f"> {m['aviso']}\n\n"
        f"Semente de teste: {m['semente']} (diferente da usada para calibrar os limiares). Pacientes simulados — {por_fonte}.\n\n"
        "## Métricas por módulo\n\n"
        + tabela_markdown(m)
        + "\n## Findings que não casaram com nenhum evento\n\n"
        + "\n".join(falsos)
        + "\n\nLeitura: o recall conta eventos injetados e a precisão conta Findings. Um evento pode gerar mais de um Finding "
        "(o episódio que piora de severidade emite de novo), então a precisão é por Finding. O atraso do Finding sai da hora de início do evento "
        "e, na imobilidade e na suspensão de medicamento, desconta o tempo que a própria regra espera (2 h e 30 min).\n"
    )


# ---------------------------------------------------------------------- linha de comando


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Avalia os detectores de anomalias com dados sintéticos")
    parser.add_argument("--rapido", action="store_true", help="versão pequena (um quinto dos pacientes), só para conferir se roda")
    parser.add_argument("--processos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", default=str(SAIDA_PADRAO), help="pasta de saída")
    parser.add_argument("--sem-graficos", action="store_true")
    args = parser.parse_args(argv)

    av = carregar_config_avaliacao()
    if args.rapido:
        av = reduzir(av, 5)
    tarefas = montar_tarefas(av)
    print(f"{len(tarefas)} pacientes simulados, {args.processos} processo(s)")
    saidas = executar_tarefas(tarefas, args.processos)
    metricas = calcular_metricas(saidas, av)

    pasta = Path(args.out)
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (pasta / "metricas.md").write_text(relatorio_markdown(metricas), encoding="utf-8")
    print(tabela_markdown(metricas))
    print(f"métricas gravadas em {pasta}")

    if not args.sem_graficos:
        from modules.anomaly.graficos import gerar_todos

        for caminho in gerar_todos(metricas, pasta / "graficos"):
            print(f"gráfico: {caminho}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
