"""Isolation Forest sobre os seis sinais juntos.

O z-score e os limites olham um sinal de cada vez. Aqui a pergunta é outra: esta
COMBINAÇÃO de FC, SpO2, FR, PAS, PAD e temperatura é comum para este paciente?

Como funciona o Isolation Forest: ele monta várias árvores que cortam os dados
em pontos e valores sorteados. Um ponto comum fica no meio da nuvem e leva
muitos cortes para ficar sozinho. Um ponto estranho fica isolado com poucos
cortes. O escore é a média disso nas árvores: quanto mais perto de 1, mais
estranho. É aprendizado não supervisionado: não precisa de exemplos de anomalia.

Aqui ele treina só com leituras que os outros detectores consideraram normais,
e o limiar vem dos escores do próprio treino.
"""
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from modules.anomaly.common import SEVERIDADES, classificar, novo_finding
from modules.anomaly.config import SINAIS, AnomaliaCfg, IsolationForestCfg
from modules.anomaly.episodios import Episodio, RastreadorEpisodio
from modules.anomaly.vitals import DetectorVitais, PontoVital, pontos_do_quadro
from modules.anomaly.zscore import FATOR_MAD

MODELO = "isolation-forest"
VERSAO = "1.0"

# Severidade pelo excesso normalizado (0 = no limiar, 1 = escore de saturação)
FAIXAS_EXCESSO = [(0.0, "low"), (0.4, "medium"), (0.8, "high")]


@dataclass
class Resultado:
    escore: float  # escore de anomalia, de 0 a 1
    limiar: float
    excesso: float  # quanto passou do limiar, de 0 a 1
    contribuintes: list[dict]  # os sinais que mais se afastam do normal do paciente


class ModeloIF:
    def __init__(self, cfg: IsolationForestCfg, escala_min: dict[str, float]):
        self.cfg = cfg
        self._escala_min = np.array([escala_min[s] for s in SINAIS])
        self.buffer: deque[tuple[float, list[float]]] = deque(maxlen=cfg.max_treino)
        self.reiniciar()

    def reiniciar(self) -> None:
        """Esquece tudo: o modelo e as leituras guardadas."""
        self.buffer.clear()
        self.modelo: IsolationForest | None = None
        self.limiar = 0.0
        self.mediana = np.zeros(len(SINAIS))
        self.escala = np.ones(len(SINAIS))
        self._treinado_em: float | None = None

    @property
    def treinado(self) -> bool:
        return self.modelo is not None

    def guardar_normal(self, t: float, vetor: list[float]) -> None:
        self.buffer.append((t, vetor))

    def treinar_se_preciso(self, t: float) -> None:
        if len(self.buffer) < self.cfg.min_treino:
            return
        if self.modelo is None or t - self._treinado_em >= self.cfg.refit_s:
            self._treinar(t)

    def _treinar(self, t: float) -> None:
        X = np.array([v for _, v in self.buffer])
        self.modelo = IsolationForest(n_estimators=self.cfg.n_estimators, random_state=self.cfg.semente).fit(X)
        # score_samples devolve o contrário (quanto maior, mais normal), por isso o sinal de menos
        escores = -self.modelo.score_samples(X)
        self.limiar = float(np.quantile(escores, self.cfg.quantil_limiar)) * self.cfg.margem_limiar
        # mediana e escala do treino, só para explicar quais sinais se afastaram
        self.mediana = np.median(X, axis=0)
        mad = np.median(np.abs(X - self.mediana), axis=0)
        self.escala = np.maximum(FATOR_MAD * mad, self._escala_min)
        self._treinado_em = t

    def pontuar(self, vetor: list[float]) -> Resultado | None:
        if self.modelo is None:
            return None
        x = np.array([vetor])
        escore = float(-self.modelo.score_samples(x)[0])
        faixa = max(self.cfg.a_saturacao - self.limiar, 0.05)
        excesso = min(max((escore - self.limiar) / faixa, 0.0), 1.0)

        desvios = (x[0] - self.mediana) / self.escala
        ordem = np.argsort(-np.abs(desvios))[:3]
        contribuintes = [
            {
                "sinal": SINAIS[i],
                "valor": float(x[0][i]),
                "mediana_treino": round(float(self.mediana[i]), 2),
                "desvio_robusto": round(float(desvios[i]), 2),
            }
            for i in ordem
        ]
        return Resultado(escore, self.limiar, excesso, contribuintes)


class MonitorVitais:
    """Detector de vitais completo: sinais (limites + z-score) e Isolation Forest juntos."""

    def __init__(self, cfg: AnomaliaCfg, patient_id: str):
        self.cfg = cfg
        self.patient_id = patient_id
        self.sinais = DetectorVitais(cfg, patient_id)
        self.ativo = cfg.isolation_forest.ativo
        self.modelo = ModeloIF(cfg.isolation_forest, cfg.vitais.zscore.escala_min)
        d = cfg.isolation_forest.duracao_min_s
        self.rastreador = RastreadorEpisodio("multivariado", d, d, cfg.vitais.recuperacao_s)
        self._ultimo_t: float | None = None

    def _vetor(self, ponto: PontoVital) -> list[float] | None:
        """Os seis sinais em ordem, ou None se algum faltar ou estiver fora da faixa física."""
        vetor = []
        for sinal in SINAIS:
            valor = ponto.valores.get(sinal)
            minimo, maximo = self.cfg.vitais.sinais[sinal].faixa_fisica
            if valor is None or not math.isfinite(valor) or not minimo <= valor <= maximo:
                return None
            vetor.append(float(valor))
        return vetor

    def update(self, ponto: PontoVital) -> list:
        fora_de_ordem = self.sinais.descartes["fora_de_ordem"]
        findings = self.sinais.update(ponto)
        if not self.ativo or self.sinais.descartes["fora_de_ordem"] > fora_de_ordem:
            return findings

        vetor = self._vetor(ponto)
        if vetor is None:
            return findings  # sem os seis sinais não dá para avaliar o conjunto
        t = ponto.timestamp.timestamp()

        # lacuna grande: o episódio anterior não vale mais
        if self._ultimo_t is not None and t - self._ultimo_t > self.cfg.vitais.lacuna_max_s:
            self.rastreador.encerrar()
        self._ultimo_t = t

        resultado = self.modelo.pontuar(vetor)
        ativo = resultado is not None and resultado.excesso > 0
        # Só entra no treino o que ninguém achou estranho. O que o modelo acha
        # estranho também fica de fora, senão ele aprenderia a anomalia
        if not ativo and not self.sinais.ultima_amostra_anormal:
            self.modelo.guardar_normal(t, vetor)
            self.modelo.treinar_se_preciso(t)

        rank, score = 0, 0.0
        if ativo:
            severidade, score, _ = classificar(resultado.excesso, FAIXAS_EXCESSO, "maior")
            rank = SEVERIDADES.index(severidade)
        suprimir = self.cfg.vitais.suprimir_correlatos and self.sinais.tem_episodio_emitido_aberto()
        eventos = self.rastreador.atualizar(
            t,
            ativo,
            rank=rank,
            score=score,
            tipo="padrao_multivariado",
            detalhes=None if not ativo else {"resultado": resultado, "n_treino": len(self.modelo.buffer)},
            pode_emitir=not suprimir,
        )

        ep = self.rastreador.atual
        if ep is not None and ({"emitir", "escalar"} & set(eventos)):
            findings.append(self._finding(ep, "confirmado" if "emitir" in eventos else "escalonado", ponto))

        # anomalia que não passa é, na verdade, o novo normal do paciente: recomeça o aprendizado
        if ep is not None and t - ep.inicio > self.cfg.isolation_forest.rebase_s:
            self.modelo.reiniciar()
            self.rastreador.encerrar()
        return findings

    def finalizar(self) -> None:
        self.sinais.finalizar()
        self.rastreador.encerrar()

    def episodios(self) -> list[Episodio]:
        return sorted([*self.sinais.episodios(), *self.rastreador.todos_emitidos()], key=lambda e: e.inicio)

    def _finding(self, ep: Episodio, evento: str, ponto: PontoVital):
        r: Resultado = ep.detalhes["resultado"]
        nomes = {s: self.cfg.vitais.sinais[s] for s in SINAIS}
        partes = [
            f"{nomes[c['sinal']].nome} {c['valor']:g} (mediana {c['mediana_treino']:g})" for c in r.contribuintes
        ]
        duracao = ep.ultimo_ativo - ep.inicio
        evidencia = {
            "escore_if": round(r.escore, 4),
            "limiar_if": round(r.limiar, 4),
            "excesso": round(r.excesso, 3),
            "sinais_contribuintes": r.contribuintes,
            "n_treino": ep.detalhes["n_treino"],
            "episodio_id": ep.id,
            "evento": evento,
            "duracao_s": round(duracao, 1),
        }
        return novo_finding(
            patient_id=self.patient_id,
            source="vitals",
            tipo=ep.tipo,
            inicio=datetime.fromtimestamp(ep.inicio, tz=ponto.timestamp.tzinfo),
            fim=ponto.timestamp,
            score=ep.pior_score,
            severidade=SEVERIDADES[ep.pior_rank],
            descricao="Combinação incomum de sinais vitais para este paciente: " + "; ".join(partes) + ".",
            evidencia=evidencia,
            modelo=MODELO,
            versao=VERSAO,
        )


def analisar_quadro_completo(cfg: AnomaliaCfg, quadro: pd.DataFrame) -> list:
    """Roda sinais + Isolation Forest numa série inteira e devolve todos os Findings."""
    monitor = MonitorVitais(cfg, str(quadro["patient_id"].iloc[0]))
    findings = []
    for ponto in pontos_do_quadro(quadro):
        findings += monitor.update(ponto)
    monitor.finalizar()
    return findings
