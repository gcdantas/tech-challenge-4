"""Detector de sinais vitais em fluxo.

Duas formas de detectar, que trabalham juntas em cada sinal:
  - limites clínicos (config/anomaly.yaml): define a severidade pelo valor;
  - z-score robusto (modules/anomaly/zscore.py): pega o que foge do padrão do
    próprio paciente, mesmo sem passar de nenhum limite.

Uso: um `update()` por leitura, na ordem do tempo. Ele devolve os Findings novos
(quase sempre uma lista vazia).
"""
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from modules.anomaly.common import SEVERIDADES, classificar, novo_finding
from modules.anomaly.config import SINAIS, AnomaliaCfg
from modules.anomaly.episodios import Episodio, RastreadorEpisodio
from modules.anomaly.zscore import LinhaBase

MODELO = "vitais-limites-zscore"
VERSAO = "1.0"

# Nome do achado quando um limite clínico é cruzado, por sinal e lado
TIPOS_CLINICOS = {
    ("fc", "baixo"): "bradicardia",
    ("fc", "alto"): "taquicardia",
    ("spo2", "baixo"): "dessaturacao",
    ("fr", "baixo"): "bradipneia",
    ("fr", "alto"): "taquipneia",
    ("pas", "baixo"): "hipotensao",
    ("pas", "alto"): "hipertensao",
    ("pad", "baixo"): "hipotensao_diastolica",
    ("pad", "alto"): "hipertensao_diastolica",
    ("temp", "baixo"): "hipotermia",
    ("temp", "alto"): "febre",
}


@dataclass(frozen=True)
class PontoVital:
    timestamp: datetime  # com fuso horário
    patient_id: str
    valores: dict[str, float]  # fc, spo2, fr, pas, pad, temp (pode faltar ou ser NaN)


@dataclass
class _Avaliacao:
    """O que se concluiu sobre um sinal em uma amostra."""

    sinal: str
    valor: float
    ativo: bool = False
    clinico: bool = False
    rank: int = 0
    score: float = 0.0
    tipo: str = ""
    detalhes: dict | None = None


def _limites(lado) -> list[tuple[float, str]]:
    return [(lado.medium, "medium"), (lado.high, "high"), (lado.critical, "critical")]


class DetectorVitais:
    def __init__(self, cfg: AnomaliaCfg, patient_id: str):
        self.cfg = cfg.vitais
        self.patient_id = patient_id
        zc = self.cfg.zscore
        self.linhas = {s: LinhaBase(zc.janela_s, zc.min_pontos, zc.escala_min[s]) for s in SINAIS}
        self.rastreadores = {
            s: RastreadorEpisodio(
                s, self.cfg.duracao_min_s, self.cfg.duracao_min_estatistico_s, self.cfg.recuperacao_s
            )
            for s in SINAIS
        }
        self._z_limites = [(zc.limiar, "low"), (zc.limiar_medium, "medium"), (zc.limiar_high, "high")]
        self._ultimo_t: float | None = None
        self._ultimo_t_sinal: dict[str, float] = {}
        self.descartes: Counter = Counter()  # por que cada leitura foi ignorada
        self.ultima_amostra_anormal = False  # algum sinal ficou anormal na última leitura?

    # ------------------------------------------------------------------ entrada

    def update(self, ponto: PontoVital) -> list:
        if ponto.patient_id != self.patient_id:
            raise ValueError(f"ponto do paciente {ponto.patient_id!r} enviado ao detector de {self.patient_id!r}")
        if ponto.timestamp.tzinfo is None:
            raise ValueError("timestamp precisa ter fuso horário")
        t = ponto.timestamp.timestamp()
        if self._ultimo_t is not None and t <= self._ultimo_t:
            self.descartes["fora_de_ordem"] += 1
            return []
        self._ultimo_t = t

        avaliacoes = []
        for sinal in SINAIS:
            valor = ponto.valores.get(sinal)
            if valor is None or not math.isfinite(valor):
                self.descartes["ausente"] += 1
                continue
            minimo, maximo = self.cfg.sinais[sinal].faixa_fisica
            if not minimo <= valor <= maximo:
                self.descartes[f"{sinal}_fora_da_faixa_fisica"] += 1  # erro de sensor, não é o paciente
                continue

            # lacuna grande sem leitura: o episódio anterior não vale mais
            ultimo = self._ultimo_t_sinal.get(sinal)
            if ultimo is not None and t - ultimo > self.cfg.lacuna_max_s:
                self.rastreadores[sinal].encerrar()
            self._ultimo_t_sinal[sinal] = t
            avaliacoes.append(self._avaliar(sinal, t, valor))

        self.ultima_amostra_anormal = any(av.ativo for av in avaliacoes)

        # Os sinais com limite clínico vão primeiro. Assim, se a dessaturação e a
        # alta da FC confirmam na mesma amostra, a FC já encontra o episódio clínico
        findings = []
        for av in sorted(avaliacoes, key=lambda a: not a.clinico):
            findings += self._atualizar_episodio(av, t, ponto.timestamp)
        return findings

    def tem_episodio_emitido_aberto(self) -> bool:
        return any(r.atual is not None and r.atual.emitido for r in self.rastreadores.values())

    def finalizar(self) -> None:
        """Fim da série: fecha os episódios que ainda estão abertos."""
        for r in self.rastreadores.values():
            r.encerrar()

    def episodios(self) -> list[Episodio]:
        """Episódios que geraram Finding, em ordem de início."""
        todos = [ep for r in self.rastreadores.values() for ep in r.todos_emitidos()]
        return sorted(todos, key=lambda ep: ep.inicio)

    # ------------------------------------------------------------------ por sinal

    def _avaliar(self, sinal: str, t: float, valor: float) -> _Avaliacao:
        cfg = self.cfg.sinais[sinal]
        linha = self.linhas[sinal]
        av = _Avaliacao(sinal, valor)

        clinico = None
        for lado in ("baixo", "alto"):
            limites = getattr(cfg, lado)
            if limites is None:
                continue
            c = classificar(valor, _limites(limites), "menor" if lado == "baixo" else "maior")
            if c:
                clinico = (lado, *c)

        z = linha.pontuar(t, valor)
        estatistico = classificar(abs(z), self._z_limites, "maior") if z is not None else None

        if clinico is None and estatistico is None:
            linha.guardar(t, valor)  # leitura normal: entra na linha de base
            return av

        # Leitura anormal NÃO entra na linha de base, senão uma anomalia longa viraria o "novo normal"
        av.ativo = True
        mediana = linha.mediana_e_escala()[0] if linha.pronta() else None
        av.detalhes = {
            "sinal": sinal,
            "nome": cfg.nome,
            "unidade": cfg.unidade,
            "valor": valor,
            "z_robusto": None if z is None else round(z, 2),
            "mediana_base": None if mediana is None else round(mediana, 2),
            "limite_cruzado": None,
        }
        if clinico:
            lado, severidade, score, limite = clinico
            av.clinico = True
            av.tipo = TIPOS_CLINICOS[(sinal, lado)]
            av.rank, av.score = SEVERIDADES.index(severidade), score
            av.detalhes["limite_cruzado"] = limite
        if estatistico:
            severidade, score, _ = estatistico
            rank_z = SEVERIDADES.index(severidade)
            if not clinico:
                av.tipo = f"desvio_{sinal}"
            if (rank_z, score) > (av.rank, av.score) or not clinico:
                av.rank, av.score = rank_z, score
        return av

    def _outro_sinal_com_episodio_clinico(self, sinal: str) -> bool:
        return any(
            r.atual is not None and r.atual.clinico and r.atual.emitido
            for s, r in self.rastreadores.items()
            if s != sinal
        )

    def _atualizar_episodio(self, av: _Avaliacao, t: float, ts: datetime) -> list:
        suprimir = (
            self.cfg.suprimir_correlatos and not av.clinico and self._outro_sinal_com_episodio_clinico(av.sinal)
        )
        rastreador = self.rastreadores[av.sinal]
        eventos = rastreador.atualizar(
            t,
            av.ativo,
            rank=av.rank,
            score=av.score,
            tipo=av.tipo,
            clinico=av.clinico,
            detalhes=av.detalhes,
            pode_emitir=not suprimir,
        )
        ep = rastreador.atual
        if ep is None or not ({"emitir", "escalar"} & set(eventos)):
            return []
        return [self._finding(ep, "confirmado" if "emitir" in eventos else "escalonado", ts)]

    # ------------------------------------------------------------------ saída

    def _finding(self, ep: Episodio, evento: str, ts: datetime):
        d = ep.detalhes
        inicio = datetime.fromtimestamp(ep.inicio, tz=ts.tzinfo)
        duracao = ep.ultimo_ativo - ep.inicio

        if d["limite_cruzado"] is not None:
            descricao = (
                f"{d['nome'].capitalize()}: pior valor {d['valor']:g} {d['unidade']}, "
                f"além do limite de {d['limite_cruzado']:g} {d['unidade']} (episódio de {duracao:.0f} s)."
            )
        else:
            descricao = (
                f"{d['nome'].capitalize()} fora do padrão recente do paciente: {d['valor']:g} {d['unidade']} "
                f"para uma mediana de {d['mediana_base']:g} (episódio de {duracao:.0f} s)."
            )
        evidencia = {
            **d,
            "episodio_id": ep.id,
            "evento": evento,
            "duracao_s": round(duracao, 1),
            "estatistico_apenas": not ep.clinico,
        }
        return novo_finding(
            patient_id=self.patient_id,
            source="vitals",
            tipo=ep.tipo,
            inicio=inicio,
            fim=ts,
            score=ep.pior_score,
            severidade=SEVERIDADES[ep.pior_rank],
            descricao=descricao,
            evidencia=evidencia,
            modelo=MODELO,
            versao=VERSAO,
        )


def pontos_do_quadro(quadro: pd.DataFrame) -> Iterable[PontoVital]:
    """Transforma o DataFrame do simulador em pontos, na ordem do tempo."""
    for linha in quadro.to_dict("records"):
        yield PontoVital(
            timestamp=linha["timestamp"].to_pydatetime(),
            patient_id=linha["patient_id"],
            valores={s: linha[s] for s in SINAIS},
        )


def analisar_quadro(cfg: AnomaliaCfg, quadro: pd.DataFrame) -> list:
    """Roda o detector numa série inteira (modo em lote) e devolve todos os Findings."""
    detector = DetectorVitais(cfg, str(quadro["patient_id"].iloc[0]))
    findings = []
    for ponto in pontos_do_quadro(quadro):
        findings += detector.update(ponto)
    detector.finalizar()
    return findings
