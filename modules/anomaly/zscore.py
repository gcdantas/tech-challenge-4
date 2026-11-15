"""Z-score com janela.

O z-score é (valor - média) / desvio padrão. Aqui trocamos:
  - a média pela MEDIANA da janela recente;
  - o desvio padrão pelo MAD (mediana dos desvios absolutos), multiplicado por
    1,4826 para ficar na mesma escala do desvio padrão de uma distribuição normal.

Média e desvio padrão são puxados por valores extremos. Se a anomalia entra na
janela, ela mesma aumenta o desvio padrão e some do z-score. A mediana e o MAD
quase não se mexem com alguns pontos fora do padrão.
"""
import math
from collections import deque

import numpy as np
import pandas as pd

from modules.anomaly.config import AnomaliaCfg

# Com 1,4826 o MAD de dados normais fica igual ao desvio padrão
FATOR_MAD = 1.4826


class LinhaBase:
    """Linha de base de UM sinal: mediana e MAD dos últimos `janela_s` segundos.

    Uso em fluxo, uma amostra por vez:
        z = linha.pontuar(t, valor)   # só calcula, não guarda
        linha.guardar(t, valor)       # coloca a amostra na janela
    São dois métodos para quem usa decidir se uma amostra anômala deve entrar
    na linha de base ou não.
    """

    def __init__(self, janela_s: float, min_pontos: int, escala_min: float):
        self.janela_s = janela_s
        self.min_pontos = min_pontos
        self.escala_min = escala_min
        self.pontos: deque[tuple[float, float]] = deque()  # (tempo em s, valor)

    def _conferir_ordem(self, t: float) -> None:
        if self.pontos and t <= self.pontos[-1][0]:
            raise ValueError("timestamp fora de ordem: cada amostra precisa ser mais nova que a anterior")

    def _descartar_antigos(self, t: float) -> None:
        # a janela é por TEMPO e não por quantidade: se o sensor ficou fora do
        # ar por 1 h, os pontos antigos saem e a linha de base recomeça
        while self.pontos and self.pontos[0][0] < t - self.janela_s:
            self.pontos.popleft()

    def pronta(self) -> bool:
        return len(self.pontos) >= self.min_pontos

    def mediana_e_escala(self) -> tuple[float, float]:
        valores = np.array([v for _, v in self.pontos])
        mediana = float(np.median(valores))
        mad = float(np.median(np.abs(valores - mediana)))
        return mediana, max(FATOR_MAD * mad, self.escala_min)

    def pontuar(self, t: float, valor: float) -> float | None:
        """Z-score de `valor`, ou None se ainda não dá para calcular."""
        if not math.isfinite(valor):
            return None
        self._conferir_ordem(t)
        self._descartar_antigos(t)
        if not self.pronta():
            return None
        mediana, escala = self.mediana_e_escala()
        return (valor - mediana) / escala

    def guardar(self, t: float, valor: float) -> None:
        if not math.isfinite(valor):
            return
        self._conferir_ordem(t)
        self._descartar_antigos(t)
        self.pontos.append((t, valor))


def z_da_serie(quadro: pd.DataFrame, sinal: str, cfg: AnomaliaCfg, congelar: bool = True) -> pd.Series:
    """Z-score de um sinal ao longo de toda a série (para análise e testes).

    - Leitura fora da `faixa_fisica` é erro de sensor: não é pontuada nem guardada.
    - Com `congelar`, amostra com |z| acima do limiar NÃO entra na linha de base.
      Assim uma anomalia longa não vira "o novo normal". O outro lado dessa
      moeda é que, se o paciente mudar de patamar de verdade, a linha de base
      não acompanha; o detector de vitais resolve isso depois.
    """
    zc = cfg.vitais.zscore
    minimo, maximo = cfg.vitais.sinais[sinal].faixa_fisica
    linha = LinhaBase(zc.janela_s, zc.min_pontos, zc.escala_min[sinal])

    zs = []
    for ts, valor in zip(quadro["timestamp"], quadro[sinal]):
        t = ts.timestamp()
        if not (math.isfinite(valor) and minimo <= valor <= maximo):
            zs.append(math.nan)
            continue
        z = linha.pontuar(t, valor)
        zs.append(math.nan if z is None else z)
        if not (congelar and z is not None and abs(z) > zc.limiar):
            linha.guardar(t, valor)
    return pd.Series(zs, index=quadro.index, name=f"z_{sinal}")


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    from contracts.finding import DEMO_TZ
    from modules.anomaly.config import SINAIS, carregar_config, carregar_config_simulador
    from modules.anomaly.simulator import gerar_cenario_vitais

    parser = argparse.ArgumentParser(description="Mostra o z-score num cenário simulado")
    parser.add_argument("--cenario", default="p001_demo")
    parser.add_argument("--paciente", default="P001")
    args = parser.parse_args()

    cfg = carregar_config()
    inicio = datetime(2026, 11, 3, 8, 30, tzinfo=DEMO_TZ)
    quadro, rotulos = gerar_cenario_vitais(carregar_config_simulador(), args.cenario, args.paciente, inicio)
    limiar = cfg.vitais.zscore.limiar

    dentro = pd.Series(False, index=quadro.index)
    for r in rotulos:
        dentro |= (quadro["timestamp"] >= r.inicio) & (quadro["timestamp"] <= r.fim)

    print(f"limiar |z| > {limiar}")
    print(f"{'sinal':<6}{'pico |z| no evento':>20}{'pico |z| fora':>16}{'amostras fora > limiar':>26}")
    for sinal in SINAIS:
        z = z_da_serie(quadro, sinal, cfg).abs()
        pico_dentro = z[dentro].max() if dentro.any() else float("nan")
        pico_fora = z[~dentro].max()
        print(f"{sinal:<6}{pico_dentro:>20.1f}{pico_fora:>16.1f}{int((z[~dentro] > limiar).sum()):>26}")