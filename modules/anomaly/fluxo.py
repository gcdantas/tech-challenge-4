"""Simulador em fluxo: entrega os eventos ao monitor ponto a ponto, em tempo acelerado.

`intercalar` junta as fontes (vitais, movimentação, prescrições) numa fila só, em
ordem de tempo. `reproduzir` entrega essa fila no ritmo da série, acelerado pela
`velocidade` do config/simulator.yaml (60 = 1 s real para cada minuto da série).

Para ver rodando, com o paciente do config:

    python -m modules.anomaly.fluxo --paciente P001 --velocidade 120
"""
import heapq
import time
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime


def intercalar(*fontes: Iterable) -> Iterator:
    """Junta fontes que já vêm em ordem de tempo. Em empate, vale a ordem em que as fontes foram passadas."""
    return heapq.merge(*fontes, key=lambda evento: evento.timestamp)


def reproduzir(
    eventos: Iterable,
    velocidade: float,
    a_partir_de: datetime,
    esperar: Callable[[float], None] = time.sleep,
    agora: Callable[[], float] = time.monotonic,
) -> Iterator:
    """Entrega cada evento no instante certo, acelerado por `velocidade`. Com 0 entrega tudo de uma vez.

    Os eventos de antes de `a_partir_de` (o histórico, como os pedidos de admissão)
    saem sem pausa. O ritmo é medido contra o relógio desde o começo, e não por
    pausas curtas somadas, porque em pausa de poucos milissegundos o Windows atrasa.
    """
    inicio_real = agora()
    for evento in eventos:
        if velocidade > 0 and evento.timestamp > a_partir_de:
            alvo = inicio_real + (evento.timestamp - a_partir_de).total_seconds() / velocidade
            falta = alvo - agora()
            if falta > 0.005:
                esperar(falta)
        yield evento


if __name__ == "__main__":
    import argparse

    from modules.anomaly.config import carregar_config, carregar_config_simulador
    from modules.anomaly.monitor import MonitorPaciente
    from modules.anomaly.pipeline import eventos_da_serie, preparar_series

    parser = argparse.ArgumentParser(description="Reproduz o paciente em tempo acelerado e mostra os Findings quando saem")
    parser.add_argument("--paciente", default="P001")
    parser.add_argument("--velocidade", type=float, help="segundos da série por segundo real (padrão: o do yaml)")
    parser.add_argument("--input", help="movement.parquet do vídeo do leito (sem isso, a movimentação é simulada)")
    args = parser.parse_args()

    cfg, sim = carregar_config(), carregar_config_simulador()
    velocidade = sim.fluxo.velocidade if args.velocidade is None else args.velocidade
    series = preparar_series(args.paciente, args.input, cfg_sim=sim)
    monitor = MonitorPaciente(cfg, args.paciente)

    print(f"{args.paciente}: série de {series.inicio:%H:%M} a {series.fim:%H:%M}, velocidade {velocidade:g}x")
    ultimo_marco = None
    for evento in reproduzir(eventos_da_serie(series), velocidade, series.inicio):
        if evento.timestamp >= series.inicio:  # um marcador a cada 10 min da série
            marco = evento.timestamp.replace(minute=evento.timestamp.minute // 10 * 10, second=0, microsecond=0)
            if marco != ultimo_marco:
                print(f"-- {marco:%H:%M} --")
                ultimo_marco = marco
        for f in monitor.update(evento):
            print(f"[{f.timestamp_end:%H:%M:%S}] {f.source:<13}{f.type:<24}{f.severity:<9}{f.score:.2f}  {f.description}")
    for f in monitor.finalizar():
        print(f"[{f.timestamp_end:%H:%M:%S}] {f.source:<13}{f.type:<24}{f.severity:<9}{f.score:.2f}  {f.description}")
    print("fim da série")
