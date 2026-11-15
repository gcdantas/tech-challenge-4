"""Peças comuns dos detectores: severidade, score e criação do Finding."""
import math
from datetime import datetime
from typing import Any, Literal

from contracts.finding import DEMO_TZ, Finding, ModelInfo

# Da menos para a mais grave. O rank é a posição nesta tupla.
SEVERIDADES = ("low", "medium", "high", "critical")

# Score (0 a 1) de cada severidade no exato momento em que o limite é cruzado
SCORE_NO_LIMITE = {"low": 0.30, "medium": 0.50, "high": 0.75, "critical": 0.95}


def para_demo(ts: datetime) -> datetime:
    """Converte para o fuso da demonstração (-03:00) sem mudar o instante."""
    if ts.tzinfo is None:
        raise ValueError("timestamp sem fuso horário")
    return ts.astimezone(DEMO_TZ)


def classificar(
    valor: float, limites: list[tuple[float, str]], sentido: Literal["maior", "menor"]
) -> tuple[str, float, float] | None:
    """Severidade, score e limite cruzado de `valor`, ou None se nenhum limite foi cruzado.

    `limites` vem em ordem crescente de gravidade, ex.: [(94, "medium"), (90, "high"), (85, "critical")].
    Com sentido "maior" o limite vale quando valor >= limite; com "menor", quando valor <= limite.
    A severidade é a do último limite cruzado. O score sobe em linha reta entre o
    limite cruzado e o próximo, então 87 % de SpO2 pontua mais que 89 %.
    """
    if not math.isfinite(valor):
        return None
    sinal = 1.0 if sentido == "maior" else -1.0
    v = sinal * valor
    lims = [sinal * limite for limite, _ in limites]

    idx = -1
    for i, lim in enumerate(lims):
        if v >= lim:
            idx = i
    if idx < 0:
        return None

    limite_cruzado, severidade = limites[idx]
    base = SCORE_NO_LIMITE[severidade]
    if idx == len(limites) - 1:
        return severidade, base, limite_cruzado
    proximo = SCORE_NO_LIMITE[limites[idx + 1][1]]
    fracao = (v - lims[idx]) / (lims[idx + 1] - lims[idx])
    return severidade, base + (proximo - base) * fracao, limite_cruzado


def novo_finding(
    *,
    patient_id: str,
    source: str,
    tipo: str,
    inicio: datetime,
    fim: datetime,
    score: float,
    severidade: str,
    descricao: str,
    evidencia: dict[str, Any],
    modelo: str,
    versao: str,
) -> Finding:
    """Cria um Finding válido: score entre 0 e 1, fuso -03:00 e fim nunca antes do início."""
    inicio, fim = para_demo(inicio), para_demo(fim)
    return Finding(
        patient_id=patient_id,
        source=source,
        type=tipo,
        timestamp_start=inicio,
        timestamp_end=max(fim, inicio),
        score=round(min(max(score, 0.0), 1.0), 4),
        severity=severidade,
        description=descricao,
        evidence=evidencia,
        model=ModelInfo(name=modelo, version=versao),
    )