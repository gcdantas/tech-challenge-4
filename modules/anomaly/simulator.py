"""Simulador de sinais vitais. Os dados são SINTÉTICOS.

Cada sinal é: valor basal + oscilação lenta (deriva) + ruído AR(1).
Em cima disso o simulador injeta anomalias (que o sistema deve sinalizar) e
distratores (que parecem anomalia, mas o sistema NÃO deve sinalizar).
"""
import math
import zlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from modules.anomaly.config import (
    SINAIS,
    TIPOS_ANOMALIA,
    CenarioPrescricoesCfg,
    EventoMovimentoCfg,
    InjecaoCfg,
    SimuladorCfg,
)
from modules.anomaly.prescription import EventoPrescricao, normalizar_nome, ordenar_eventos


@dataclass(frozen=True)
class Rotulo:
    """Verdade de terreno: o que foi injetado e quando. Usado na avaliação."""

    fonte: str  # vitals, prescription ou movement
    tipo: str
    inicio: datetime
    fim: datetime
    anomalia: bool  # True = o sistema deve sinalizar, False = distrator
    detalhes: dict[str, Any] = field(default_factory=dict)


def criar_rng(semente: int, patient_id: str) -> np.random.Generator:
    # crc32 em vez de hash(): o hash() do Python muda a cada execução e a
    # série deixaria de ser reproduzível
    return np.random.default_rng([semente, zlib.crc32(patient_id.encode("utf-8"))])


def ruido_ar1(n: int, phi: float, desvio: float, rng: np.random.Generator) -> np.ndarray:
    """Ruído em que cada ponto depende do anterior: x[k] = phi * x[k-1] + novidade.

    A novidade é escalada com sqrt(1 - phi²) para o desvio padrão final do
    ruído ficar igual a `desvio`, não importa o valor de phi.
    """
    ruido = np.zeros(n)
    sorteio = rng.standard_normal(n)
    ruido[0] = sorteio[0] * desvio
    novidade = math.sqrt(1 - phi**2) * desvio
    for k in range(1, n):
        ruido[k] = phi * ruido[k - 1] + novidade * sorteio[k]
    return ruido


def forma_trapezio(t: np.ndarray, inicio: float, duracao: float, rampa: float) -> np.ndarray:
    """Vale 0 antes do evento, sobe até 1 em `rampa` segundos, fica em 1 e volta a 0.

    A descida começa em inicio + duracao e também leva `rampa` segundos. Com
    rampa = 0 vira um degrau. Multiplicar isto pela magnitude dá a anomalia.
    """
    if rampa <= 0:
        return ((t >= inicio) & (t < inicio + duracao)).astype(float)
    sobe = np.clip((t - inicio) / rampa, 0, 1)
    desce = np.clip((inicio + duracao + rampa - t) / rampa, 0, 1)
    return np.minimum(sobe, desce)


def _aplicar_injecao(valores: dict[str, np.ndarray], t: np.ndarray, inj: InjecaoCfg, resposta_fisiologica: bool):
    """Altera `valores` (antes do arredondamento) e devolve as máscaras de falha do sensor."""
    inicio = inj.inicio_min * 60
    duracao = inj.duracao_min * 60
    f = forma_trapezio(t, inicio, duracao, inj.rampa_s)
    m = inj.magnitude
    janela = (t >= inicio) & (t < inicio + duracao)
    ausente = np.zeros(len(t), dtype=bool)
    invalido = np.zeros(len(t), dtype=bool)

    if inj.tipo == "dessaturacao":
        valores["spo2"] -= m * f
        if resposta_fisiologica:
            valores["fc"] += 1.0 * m * f
            valores["fr"] += 0.4 * m * f
    elif inj.tipo == "taquicardia":
        valores["fc"] += m * f
    elif inj.tipo == "hipotensao":
        valores["pas"] -= m * f
        valores["pad"] -= 0.6 * m * f
        valores["fc"] += 0.3 * m * f  # o coração acelera para compensar
    elif inj.tipo == "febre":
        valores["temp"] += m * f
        valores["fc"] += 8.0 * m * f
    elif inj.tipo == "taquipneia":
        valores["fr"] += m * f
    elif inj.tipo == "deterioracao_multivariada":
        # vários sinais pioram um pouco ao mesmo tempo; nenhum passa do limite
        # sozinho, mas o conjunto foge do padrão do paciente
        valores["fc"] += 10.0 * m * f
        valores["fr"] += 3.0 * m * f
        valores["spo2"] -= 2.0 * m * f
        valores["pas"] -= 8.0 * m * f
    elif inj.tipo == "artefato_pico":
        # queda brusca e curta do SpO2, típica de sensor mexendo no dedo
        valores["spo2"] -= m * janela
    elif inj.tipo == "perda_sinal":
        ausente = janela
    elif inj.tipo == "leitura_invalida":
        invalido = janela
    return ausente, invalido


def rotulos_das_injecoes(inicio: datetime, injecoes: Iterable[InjecaoCfg]) -> list[Rotulo]:
    """Rótulos de verdade das injeções, na mesma ordem em que foram dadas."""
    rotulos = []
    for inj in injecoes:
        anomalia = inj.tipo in TIPOS_ANOMALIA
        # a anomalia só acaba de verdade depois da rampa de descida
        fim_s = inj.inicio_min * 60 + inj.duracao_min * 60 + (inj.rampa_s if anomalia else 0)
        rotulos.append(
            Rotulo(
                fonte="vitals",
                tipo=inj.tipo,
                inicio=inicio + timedelta(minutes=inj.inicio_min),
                fim=inicio + timedelta(seconds=fim_s),
                anomalia=anomalia,
                detalhes={"magnitude": inj.magnitude},
            )
        )
    return rotulos


def gerar_vitais(
    cfg: SimuladorCfg,
    patient_id: str,
    inicio: datetime,
    rng: np.random.Generator | None = None,
    duracao_min: float | None = None,
    variar_paciente: bool = False,
    injecoes: Iterable[InjecaoCfg] = (),
) -> pd.DataFrame:
    """Série de sinais vitais simulada, com as injeções (se houver).

    `inicio` precisa ter fuso (o projeto usa -03:00). Com `variar_paciente`
    o basal de cada sinal é sorteado em torno do valor do yaml.
    Sem injeções a série é só o paciente estável.
    """
    if inicio.tzinfo is None:
        raise ValueError("inicio precisa ter fuso horário (ex.: -03:00)")

    sv = cfg.vitais
    rng = rng if rng is not None else criar_rng(cfg.semente, patient_id)
    duracao = sv.duracao_min if duracao_min is None else duracao_min

    n = int(round(duracao * 60 / sv.periodo_s))
    t = np.arange(n) * sv.periodo_s  # segundos desde o início

    valores = {}
    for sinal in SINAIS:
        basal = sv.basal[sinal]
        if variar_paciente:
            basal += rng.uniform(-1, 1) * sv.variacao_entre_pacientes[sinal]
        # fase aleatória para os sinais não oscilarem todos juntos
        fase = rng.uniform(0, 2 * math.pi)
        deriva = sv.deriva[sinal] * np.sin(2 * math.pi * t / 7200 + fase)
        valores[sinal] = basal + deriva + ruido_ar1(n, sv.phi, sv.ruido[sinal], rng)

    # As injeções entram DEPOIS de sortear o ruído. Assim a série com e sem
    # injeção é igual fora do evento, e dá para comparar uma com a outra
    ausente = np.zeros(n, dtype=bool)
    invalido = np.zeros(n, dtype=bool)
    for inj in injecoes:
        if inj.inicio_min * 60 >= n * sv.periodo_s:
            raise ValueError(f"{inj.tipo} começa depois do fim da série ({duracao:g} min)")
        aus, inv = _aplicar_injecao(valores, t, inj, sv.resposta_fisiologica)
        ausente |= aus
        invalido |= inv

    for sinal in SINAIS:
        if sv.arredondar:
            valores[sinal] = np.round(valores[sinal], 1 if sinal == "temp" else 0)
    valores["spo2"] = np.minimum(valores["spo2"], 100.0)  # saturação não passa de 100

    # Falhas de sensor entram por último, senão o arredondamento as desfaria
    for sinal in SINAIS:
        if sinal in ("spo2", "fc"):
            valores[sinal][invalido] = 0.0  # sensor desconectado marca zero
        valores[sinal][ausente] = np.nan  # sem sinal nenhum

    quadro = pd.DataFrame(valores)
    quadro.insert(0, "timestamp", pd.date_range(inicio, periods=n, freq=f"{sv.periodo_s:g}s"))
    quadro.insert(0, "patient_id", patient_id)
    return quadro


def gerar_cenario_vitais(
    cfg: SimuladorCfg, nome: str, patient_id: str, inicio: datetime, **kwargs
) -> tuple[pd.DataFrame, list[Rotulo]]:
    """Gera a série de um cenário do yaml e devolve também os rótulos de verdade."""
    if nome not in cfg.cenarios:
        raise KeyError(f"cenário {nome!r} não existe em simulator.yaml (tem: {', '.join(cfg.cenarios)})")
    injecoes = cfg.cenarios[nome].vitais.injecoes
    quadro = gerar_vitais(cfg, patient_id, inicio, injecoes=injecoes, **kwargs)
    return quadro, rotulos_das_injecoes(inicio, injecoes)


def gerar_prescricoes(
    cfg: SimuladorCfg, patient_id: str, referencia: datetime, cenario: CenarioPrescricoesCfg
) -> tuple[list[EventoPrescricao], list[Rotulo]]:
    """Eventos de prescrição de um cenário, em ordem de tempo, e os rótulos de verdade.

    Os pedidos iniciais entram na admissão (`historico_h` antes da referência) e cada
    evento do cenário acontece `minuto` minutos depois da referência.
    """
    eventos: list[EventoPrescricao] = []
    rotulos: list[Rotulo] = []
    ativas: dict[str, list[str]] = {}  # medicamento -> ids das ordens ativas
    contador = 0

    def novo_id() -> str:
        nonlocal contador
        contador += 1
        return f"{patient_id}-O{contador:03d}"

    admissao = referencia - timedelta(hours=cfg.prescricoes.historico_h)
    for k, pedido in enumerate(cenario.pedidos_iniciais):
        ordem = novo_id()
        ativas.setdefault(normalizar_nome(pedido.medicamento), []).append(ordem)
        # um pedido a cada 5 min, para não ficarem todos no mesmo segundo
        eventos.append(
            EventoPrescricao(
                admissao + timedelta(minutes=5 * k), patient_id, "prescrever", ordem,
                pedido.medicamento, pedido.dose_mg, pedido.doses_por_dia,
            )
        )

    for ev in cenario.eventos:
        quando = referencia + timedelta(minutes=ev.minuto)
        chave = normalizar_nome(ev.medicamento)
        if ev.acao == "prescrever":
            ordem = novo_id()
            ativas.setdefault(chave, []).append(ordem)
            eventos.append(EventoPrescricao(quando, patient_id, "prescrever", ordem, ev.medicamento, ev.dose_mg, ev.doses_por_dia))
        else:
            if not ativas.get(chave):
                raise ValueError(f"cenário suspende {ev.medicamento!r} sem pedido ativo")
            ordem = ativas[chave].pop(0)  # suspende o pedido mais antigo
            eventos.append(EventoPrescricao(quando, patient_id, "suspender", ordem))
        if ev.anomalia:
            rotulos.append(
                Rotulo("prescription", ev.tipo_esperado, quando, quando, True, {"medicamento": ev.medicamento, "ordem_id": ordem})
            )
    return ordenar_eventos(eventos), rotulos


def gerar_movimento(
    cfg: SimuladorCfg,
    patient_id: str,
    inicio: datetime,
    rng: np.random.Generator | None = None,
    duracao_min: float | None = None,
    hz: float | None = None,
    eventos: Iterable[EventoMovimentoCfg] = (),
) -> tuple[pd.DataFrame, list[Rotulo]]:
    """Série de movimentação do leito (formato do movement.parquet) e os rótulos de verdade.

    O paciente fica deitado e se reposiciona de tempos em tempos (a equipe vira
    o paciente). Sem eventos a série é só isso, e não pode gerar alerta.
    """
    if inicio.tzinfo is None:
        raise ValueError("inicio precisa ter fuso horário (ex.: -03:00)")
    sm = cfg.movimento
    eventos = sorted(eventos, key=lambda e: e.inicio_min)
    rng = rng if rng is not None else criar_rng(cfg.semente + 1, patient_id)
    hz = sm.hz if hz is None else hz
    duracao = sm.duracao_min if duracao_min is None else duracao_min
    n = int(round(duracao * 60 * hz))
    t = np.arange(n) / hz
    total_s = n / hz

    for ev in eventos:
        if ev.inicio_min * 60 >= total_s:
            raise ValueError(f"{ev.tipo} começa depois do fim da série ({duracao:g} min)")

    # --- reposicionamentos: os naturais, mais os pedidos pelo cenário, menos os que caem numa imobilidade
    imobilidades = [(e.inicio_min * 60, e.inicio_min * 60 + e.duracao_s) for e in eventos if e.tipo == "imobilidade"]
    reposicoes = []
    instante = 0.0
    while True:
        instante += rng.uniform(0.7, 1.3) * sm.reposicionamento_min * 60
        if instante >= total_s:
            break
        reposicoes.append(instante)
    reposicoes = [r for r in reposicoes if not any(a <= r < b for a, b in imobilidades)]
    reposicoes += [e.inicio_min * 60 for e in eventos if e.tipo == "virar_no_leito"]
    reposicoes += [b for _, b in imobilidades if b < total_s]  # no fim da imobilidade alguém vira o paciente
    reposicoes = sorted(reposicoes)

    # a pose alterna entre a base, um lado, a base e o outro lado: cada virada muda mais que a tolerância
    amplitude = rng.uniform(0.8, 1.2, size=len(reposicoes))
    ciclo = [0.0, 1.0, 0.0, -1.0]
    angulo = np.full(n, sm.deitado.torso_angle)
    cx = np.full(n, sm.deitado.centroid_x)
    cy = np.full(n, sm.deitado.centroid_y)
    anterior = 0.0
    for k, r in enumerate(reposicoes):
        atual = ciclo[(k + 1) % 4]
        degrau = np.clip((t - r) / sm.rampa_s, 0, 1) * (atual - anterior)
        angulo += degrau * sm.deslocamento_angulo * amplitude[k]
        cx += degrau * sm.deslocamento_centro * amplitude[k]
        anterior = atual

    angulo += ruido_ar1(n, sm.phi, sm.ruido_angulo, rng)
    cx += ruido_ar1(n, sm.phi, sm.ruido_centro, rng)
    cy += ruido_ar1(n, sm.phi, sm.ruido_centro, rng)

    # --- eventos de saída do leito e distratores
    em_leito = np.ones(n, dtype=bool)
    sem_pessoa = np.zeros(n, dtype=bool)
    rotulos: list[Rotulo] = []
    pose_fora = (sm.fora.torso_angle, sm.fora.centroid_x, sm.fora.centroid_y)
    for ev in eventos:
        t0, t1 = ev.inicio_min * 60, ev.inicio_min * 60 + ev.duracao_s
        quando = lambda s: inicio + timedelta(seconds=float(s))  # noqa: E731
        if ev.tipo == "saida_do_leito":
            # a pessoa vai até a pose de fora em `rampa_s` e volta depois do fim
            subida = np.clip((t - t0) / sm.rampa_s, 0, 1)
            descida = np.clip((t - t1) / sm.rampa_s, 0, 1)
            peso = subida * (1 - descida)
            fora = (t >= t0) & (t < t1)
            if ev.oscila:
                # debruçada no leito: a cada 12 s, 4 s com in_bed verdadeiro e a pose no meio do caminho
                ciclo_t = (t - t0 - 8) % 12
                debrucada = fora & (t >= t0 + 8) & (t < t1 - 8) & (ciclo_t < 4)
                peso = np.where(debrucada, 0.5, peso)
                fora = fora & ~debrucada
            angulo = angulo * (1 - peso) + pose_fora[0] * peso
            cx = cx * (1 - peso) + pose_fora[1] * peso
            cy = cy * (1 - peso) + pose_fora[2] * peso
            em_leito &= ~fora
            rotulos.append(Rotulo("movement", "saida_do_leito", quando(t0), quando(t1), True, {"oscila": ev.oscila}))
        elif ev.tipo == "falso_quadro":
            em_leito &= ~((t >= t0) & (t < t1))
            rotulos.append(Rotulo("movement", "falso_quadro", quando(t0), quando(t1), False))
        elif ev.tipo == "perda_deteccao":
            # o I1 mantém o in_bed por 1 s sem detecção e depois marca falso
            sem = (t >= t0) & (t < t1)
            sem_pessoa |= sem
            em_leito &= ~((t >= t0 + 1) & (t < t1))
            rotulos.append(Rotulo("movement", "perda_deteccao", quando(t0), quando(t1), False))
        elif ev.tipo == "virar_no_leito":
            rotulos.append(Rotulo("movement", "virar_no_leito", quando(t0), quando(t0 + sm.rampa_s), False))
        elif ev.tipo == "imobilidade":
            # parado desde que a última virada terminou (a virada leva `rampa_s`)
            ultima = max((r + sm.rampa_s for r in reposicoes if r < t0), default=0.0)
            rotulos.append(
                Rotulo("movement", "imobilidade_prolongada", quando(ultima), quando(min(t1, total_s)), True, {"parado_desde_s": ultima})
            )

    # velocidade como o I1 calcula: deslocamento do centro por segundo, média móvel de 0,6 s
    velocidade = np.zeros(n)
    velocidade[1:] = np.hypot(np.diff(cx), np.diff(cy)) * hz
    velocidade = pd.Series(velocidade).rolling(max(1, round(0.6 * hz)), min_periods=1).mean().to_numpy().copy()

    for v in (angulo, cx, cy, velocidade):
        v[sem_pessoa] = np.nan
    quadro = pd.DataFrame(
        {
            "patient_id": patient_id,
            "timestamp": pd.date_range(inicio, periods=n, freq=pd.Timedelta(seconds=1 / hz)),
            "centroid_x": np.round(cx, 4),
            "centroid_y": np.round(cy, 4),
            "torso_angle": np.round(angulo, 1),
            "velocidade": np.round(velocidade, 4),
            "in_bed": em_leito,
        }
    )
    return quadro, rotulos


def gerar_cenario_movimento(
    cfg: SimuladorCfg, nome: str, patient_id: str, inicio: datetime, **kwargs
) -> tuple[pd.DataFrame, list[Rotulo]]:
    """Gera a série de movimentação de um cenário do yaml, com os rótulos de verdade."""
    if nome not in cfg.cenarios:
        raise KeyError(f"cenário {nome!r} não existe em simulator.yaml (tem: {', '.join(cfg.cenarios)})")
    cenario = cfg.cenarios[nome].movimento
    kwargs.setdefault("duracao_min", cenario.duracao_min)
    return gerar_movimento(cfg, patient_id, inicio, eventos=cenario.eventos, **kwargs)


def compor_movimento(
    cfg: SimuladorCfg, patient_id: str, clipe: pd.DataFrame, inicio_serie: datetime
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Estende o movement.parquet real do I1 (um vídeo de 30 s) com o paciente deitado e parado antes dele.

    Um vídeo curto não sustenta uma série de internação. A série composta tem a
    parte simulada de `inicio_serie` até o começo do clipe (paciente no leito, sem
    eventos) e depois o clipe real, sem alterar nenhuma amostra dele. Depois do
    clipe a série acaba: não inventamos o que o paciente fez em seguida.
    Devolve a série e um dicionário que documenta a composição.
    """
    faltam = [c for c in ("patient_id", "timestamp", "centroid_x", "centroid_y", "torso_angle", "velocidade", "in_bed") if c not in clipe.columns]
    if faltam:
        raise ValueError(f"faltam colunas no clipe de movimentação: {', '.join(faltam)}")
    if clipe.empty:
        raise ValueError("o clipe de movimentação está vazio")
    if clipe["timestamp"].dt.tz is None:
        raise ValueError("o timestamp do clipe precisa ter fuso horário (-03:00)")

    clipe = clipe.sort_values("timestamp").reset_index(drop=True)
    inicio_clipe = clipe["timestamp"].iloc[0].to_pydatetime()
    fim_clipe = clipe["timestamp"].iloc[-1].to_pydatetime()
    passo = clipe["timestamp"].diff().dropna().median().total_seconds() if len(clipe) > 1 else 1.0
    hz = 1 / passo
    antes_s = (inicio_clipe - inicio_serie).total_seconds()

    documento: dict[str, Any] = {
        "clipe": {
            "inicio": inicio_clipe.isoformat(),
            "fim": fim_clipe.isoformat(),
            "amostras": int(len(clipe)),
            "frequencia_hz": round(hz, 3),
        },
        "criterio": "série simulada (paciente deitado, sem eventos) até o começo do clipe, depois o clipe real sem alterações",
    }
    if antes_s < 1 / hz:
        documento["base_simulada"] = None
        return clipe, documento

    base, _ = gerar_movimento(cfg, patient_id, inicio_serie, duracao_min=antes_s / 60, hz=hz)
    base = base[base["timestamp"] < clipe["timestamp"].iloc[0]]  # o arredondamento pode repetir a 1ª amostra do clipe
    base["patient_id"] = patient_id
    documento["base_simulada"] = {
        "inicio": inicio_serie.isoformat(),
        "fim": base["timestamp"].iloc[-1].isoformat(),
        "amostras": int(len(base)),
        "semente": cfg.semente + 1,
    }
    serie = pd.concat([base, clipe], ignore_index=True)
    return serie, documento


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from contracts.finding import DEMO_TZ
    from modules.anomaly.config import carregar_config_simulador

    parser = argparse.ArgumentParser(description="Gera um cenário de sinais vitais simulados")
    parser.add_argument("--cenario", default="p001_demo")
    parser.add_argument("--paciente", default="P001")
    parser.add_argument("--grafico", help="caminho do PNG com o gráfico (ex.: outputs/anomalias/vitais.png)")
    args = parser.parse_args()

    cfg = carregar_config_simulador()
    inicio = datetime(2026, 11, 3, 8, 30, tzinfo=DEMO_TZ)
    quadro, rotulos = gerar_cenario_vitais(cfg, args.cenario, args.paciente, inicio)

    for r in rotulos:
        tipo = "anomalia" if r.anomalia else "distrator"
        print(f"{tipo}: {r.tipo} de {r.inicio:%H:%M:%S} até {r.fim:%H:%M:%S}")

    if args.grafico:
        import matplotlib

        matplotlib.use("Agg")  # só salva o arquivo, não abre janela
        import matplotlib.pyplot as plt

        fig, eixos = plt.subplots(len(SINAIS), 1, figsize=(10, 9), sharex=True)
        for eixo, sinal in zip(eixos, SINAIS):
            eixo.plot(quadro["timestamp"], quadro[sinal], linewidth=1)
            eixo.set_ylabel(sinal)
            for r in rotulos:
                eixo.axvspan(r.inicio, r.fim, alpha=0.2, color="red" if r.anomalia else "gray")
        eixos[0].set_title(f"{args.paciente}: {args.cenario} (vermelho = anomalia, cinza = distrator)")
        Path(args.grafico).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.grafico, dpi=110)
        print(f"gráfico salvo em {args.grafico}")
