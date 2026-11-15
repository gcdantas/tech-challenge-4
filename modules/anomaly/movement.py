"""Detector de movimentação do leito (Integrante 3).

Recebe a série do vídeo do leito (movement.parquet, feita pelo Integrante 1) e
procura dois eventos:

  saida_do_leito          a pessoa saiu do leito e não voltou
  imobilidade_prolongada  a pessoa ficou na mesma pose por tempo demais

São regras, não aprendizado de máquina. O Integrante 1 só entrega a série; quem
emite o Finding (source "movement") é este módulo.

Saída do leito. O `in_bed` do vídeo pisca: um quadro errado, a pessoa se
debruçando no leito, uma oclusão. Por isso a regra tem histerese: só vale depois
de `confirmar_fora_s` seguidos fora do leito, e só termina depois de `retorno_s`
seguidos de volta. Também só conta como saída se a pessoa estava no leito antes.

Imobilidade. A pose é resumida em blocos de 1 min (mediana do ângulo do tronco e
do centro). Enquanto os blocos seguintes ficam perto da pose do primeiro bloco,
a contagem continua. Mudou de pose (a equipe virou o paciente) ou saiu do leito:
a contagem zera.
"""
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, tzinfo

import numpy as np
import pandas as pd

from modules.anomaly.common import classificar, novo_finding
from modules.anomaly.config import AnomaliaCfg
from modules.anomaly.episodios import Episodio, RastreadorEpisodio

MODELO = "movimento-regras"
VERSAO = "1.0"

COLUNAS = ["patient_id", "timestamp", "centroid_x", "centroid_y", "torso_angle", "velocidade", "in_bed"]


@dataclass(frozen=True)
class PontoMovimento:
    timestamp: datetime  # com fuso horário
    patient_id: str
    in_bed: bool | None
    torso_angle: float = math.nan  # NaN quando a pessoa não foi detectada
    centroid_x: float = math.nan
    centroid_y: float = math.nan
    velocidade: float = math.nan


@dataclass
class _Bloco:
    indice: int
    total: int = 0
    no_leito: int = 0
    angulos: list[float] = field(default_factory=list)
    xs: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)


def _numero(valor: float) -> float | None:
    """Para a evidência: NaN não vai para o JSON."""
    return None if not math.isfinite(valor) else round(float(valor), 4)


def _duracao_texto(segundos: float) -> str:
    minutos = int(round(segundos / 60))
    horas, resto = divmod(minutos, 60)
    if horas and resto:
        return f"{horas} h {resto} min"
    return f"{horas} h" if horas else f"{minutos} min"


class DetectorMovimento:
    def __init__(self, cfg: AnomaliaCfg, patient_id: str):
        self.cfg = cfg.movimento
        self.patient_id = patient_id
        s, i = self.cfg.saida_leito, self.cfg.imobilidade
        self.saida = RastreadorEpisodio("saida_leito", s.confirmar_fora_s, s.confirmar_fora_s, s.retorno_s)
        # recuperacao 0: ao mudar de pose o episódio de imobilidade fecha na hora
        self.imob = RastreadorEpisodio("imobilidade", i.limiar_s, i.limiar_s, 0.0)
        self.descartes: Counter = Counter()
        self._ultimo_t: float | None = None
        self._tz: tzinfo | None = None
        # saída do leito
        self._no_leito_desde: float | None = None
        self._esteve_no_leito = False
        # imobilidade
        self._bloco: _Bloco | None = None
        self._ancora: tuple[float, float, float] | None = None
        self._ultimo_valido_fim: float | None = None

    # ------------------------------------------------------------------ entrada

    def update(self, ponto: PontoMovimento) -> list:
        if ponto.patient_id != self.patient_id:
            raise ValueError(f"ponto do paciente {ponto.patient_id!r} enviado ao detector de {self.patient_id!r}")
        if ponto.timestamp.tzinfo is None:
            raise ValueError("timestamp precisa ter fuso horário")
        t = ponto.timestamp.timestamp()
        if self._ultimo_t is not None and t < self._ultimo_t:
            self.descartes["fora_de_ordem"] += 1
            return []
        self._tz = ponto.timestamp.tzinfo

        # lacuna grande nos dados: o que estava em andamento não vale mais
        if self._ultimo_t is not None and t - self._ultimo_t > self.cfg.imobilidade.lacuna_max_s:
            self.saida.encerrar()
            self._no_leito_desde = None
            self._esteve_no_leito = False
            self._zerar_imobilidade()
        self._ultimo_t = t

        return self._saida_do_leito(ponto, t) + self._imobilidade(ponto, t)

    def finalizar(self) -> list:
        """Fim da série: resume o último bloco e fecha os episódios abertos."""
        findings = []
        if self._bloco is not None and self._tz is not None:
            findings += self._fechar_bloco(self._bloco)
            self._bloco = None
        self.saida.encerrar()
        self.imob.encerrar()
        return findings

    def episodios(self) -> list[Episodio]:
        return sorted([*self.saida.todos_emitidos(), *self.imob.todos_emitidos()], key=lambda e: e.inicio)

    # ------------------------------------------------------------------ saída do leito

    def _saida_do_leito(self, ponto: PontoMovimento, t: float) -> list:
        cfg = self.cfg.saida_leito
        if ponto.in_bed is None:
            self.descartes["in_bed_ausente"] += 1
            return []

        if ponto.in_bed:
            # saída ainda não confirmada: voltar ao leito zera a contagem. Sem isso, um quadro falso de
            # in_bed=False no meio da deitada ficaria como início da saída de verdade (visto no SSP8C, aos 11,8 s)
            if self.saida.atual is not None and not self.saida.atual.emitido:
                self.saida.encerrar()
            if self._no_leito_desde is None:
                self._no_leito_desde = t
            if t - self._no_leito_desde >= cfg.no_leito_min_s:
                self._esteve_no_leito = True
            ativo = False
        else:
            self._no_leito_desde = None
            ativo = self._esteve_no_leito
            if not ativo and self.saida.atual is None:
                self.descartes["fora_do_leito_sem_leito_previo"] += 1

        ep = self.saida.atual
        fora_ha = t - (ep.inicio if ep is not None else t)
        alto = fora_ha >= cfg.ausencia_high_s
        eventos = self.saida.atualizar(
            t,
            ativo,
            rank=2 if alto else 1,
            score=cfg.score_high if alto else cfg.score,
            tipo="saida_do_leito",
            detalhes={
                "torso_angle": _numero(ponto.torso_angle),
                "centroid_x": _numero(ponto.centroid_x),
                "centroid_y": _numero(ponto.centroid_y),
                "pessoa_detectada": math.isfinite(ponto.torso_angle),
            },
        )
        ep = self.saida.atual
        if ep is None or not ({"emitir", "escalar"} & set(eventos)):
            return []
        return [self._finding_saida(ep, "confirmado" if "emitir" in eventos else "escalonado", ponto.timestamp, t)]

    def _finding_saida(self, ep: Episodio, evento: str, agora: datetime, t: float):
        cfg = self.cfg.saida_leito
        fora_s = t - ep.inicio
        if evento == "escalonado":
            descricao = f"Paciente continua fora do leito há {_duracao_texto(fora_s)}."
        else:
            descricao = "Paciente saiu do leito."
        evidencia = {
            **ep.detalhes,
            "episodio_id": ep.id,
            "evento": evento,
            "duracao_fora_s": round(fora_s, 1),
            "confirmar_fora_s": cfg.confirmar_fora_s,
            "ausencia_high_s": cfg.ausencia_high_s,
        }
        return novo_finding(
            patient_id=self.patient_id,
            source="movement",
            tipo="saida_do_leito",
            inicio=datetime.fromtimestamp(ep.inicio, tz=agora.tzinfo),
            fim=agora,
            score=ep.pior_score,
            severidade=("low", "medium", "high", "critical")[ep.pior_rank],
            descricao=descricao,
            evidencia=evidencia,
            modelo=MODELO,
            versao=VERSAO,
        )

    # ------------------------------------------------------------------ imobilidade

    def _imobilidade(self, ponto: PontoMovimento, t: float) -> list:
        bin_s = self.cfg.imobilidade.bin_s
        indice = int(t // bin_s)
        findings = []
        if self._bloco is not None and indice != self._bloco.indice:
            findings += self._fechar_bloco(self._bloco)
            self._bloco = None
        if self._bloco is None:
            self._bloco = _Bloco(indice)

        b = self._bloco
        if ponto.in_bed is not None:
            b.total += 1
            b.no_leito += bool(ponto.in_bed)
        if all(math.isfinite(v) for v in (ponto.torso_angle, ponto.centroid_x, ponto.centroid_y)):
            b.angulos.append(ponto.torso_angle)
            b.xs.append(ponto.centroid_x)
            b.ys.append(ponto.centroid_y)
        return findings

    def _zerar_imobilidade(self) -> None:
        self.imob.encerrar()
        self._ancora = None
        self._ultimo_valido_fim = None

    def _fechar_bloco(self, b: _Bloco) -> list:
        cfg = self.cfg.imobilidade
        inicio_b = b.indice * cfg.bin_s
        fim_b = inicio_b + cfg.bin_s

        if b.total and b.no_leito / b.total < 0.5:
            self._zerar_imobilidade()  # saiu do leito
            return []
        if len(b.angulos) < cfg.min_amostras_bin:
            # sem pose neste bloco (pessoa não detectada): não prova nada, mas não pode durar para sempre
            if self._ultimo_valido_fim is not None and fim_b - self._ultimo_valido_fim > cfg.lacuna_max_s:
                self._zerar_imobilidade()
            return []

        pose = (float(np.median(b.angulos)), float(np.median(b.xs)), float(np.median(b.ys)))
        if self._ultimo_valido_fim is not None and inicio_b - self._ultimo_valido_fim > cfg.lacuna_max_s:
            self._zerar_imobilidade()
        if self._ancora is not None and self._mudou_de_pose(pose, self._ancora):
            self._zerar_imobilidade()
        if self._ancora is None:
            self._ancora = pose
        self._ultimo_valido_fim = fim_b

        primeiro = self.imob.atual is None
        duracao = fim_b - (inicio_b if primeiro else self.imob.atual.inicio)
        c = classificar(
            duracao / cfg.limiar_s,
            [(1.0, "low"), (cfg.fator_medium, "medium"), (cfg.fator_high, "high")],
            "maior",
        )
        rank, score = (("low", "medium", "high").index(c[0]), c[1]) if c else (0, 0.0)
        eventos = self.imob.atualizar(
            inicio_b if primeiro else fim_b,
            True,
            rank=rank,
            score=score,
            tipo="imobilidade_prolongada",
            detalhes={"pose_ancora": self._ancora},
        )
        ep = self.imob.atual
        if ep is None or not ({"emitir", "escalar"} & set(eventos)):
            return []
        return [self._finding_imobilidade(ep, "confirmado" if "emitir" in eventos else "escalonado", fim_b)]

    def _mudou_de_pose(self, pose, ancora) -> bool:
        cfg = self.cfg.imobilidade
        return abs(pose[0] - ancora[0]) > cfg.tol_angulo or math.hypot(pose[1] - ancora[1], pose[2] - ancora[2]) > cfg.tol_centro

    def _finding_imobilidade(self, ep: Episodio, evento: str, fim_b: float):
        cfg = self.cfg.imobilidade
        parado_s = fim_b - ep.inicio
        angulo, cx, cy = ep.detalhes["pose_ancora"]
        evidencia = {
            "episodio_id": ep.id,
            "evento": evento,
            "parado_s": round(parado_s, 1),
            "limiar_s": cfg.limiar_s,
            "pose": {"torso_angle": round(angulo, 1), "centroid_x": round(cx, 3), "centroid_y": round(cy, 3)},
            "tolerancia": {"torso_angle": cfg.tol_angulo, "centro": cfg.tol_centro},
        }
        return novo_finding(
            patient_id=self.patient_id,
            source="movement",
            tipo="imobilidade_prolongada",
            inicio=datetime.fromtimestamp(ep.inicio, tz=self._tz),
            fim=datetime.fromtimestamp(fim_b, tz=self._tz),
            score=ep.pior_score,
            severidade=("low", "medium", "high", "critical")[ep.pior_rank],
            descricao=(
                f"Paciente na mesma pose há {_duracao_texto(parado_s)} "
                f"(limiar de {_duracao_texto(cfg.limiar_s)}). Avaliar mudança de decúbito."
            ),
            evidencia=evidencia,
            modelo=MODELO,
            versao=VERSAO,
        )


# ---------------------------------------------------------------------- série inteira


def pontos_do_quadro(quadro: pd.DataFrame) -> Iterable[PontoMovimento]:
    """Transforma o movement.parquet (um DataFrame) em pontos, na ordem das linhas."""
    faltam = [c for c in COLUNAS if c not in quadro.columns]
    if faltam:
        raise ValueError(f"faltam colunas no quadro de movimentação: {', '.join(faltam)}")
    for linha in quadro.itertuples(index=False):
        in_bed = getattr(linha, "in_bed")
        yield PontoMovimento(
            timestamp=linha.timestamp.to_pydatetime(),
            patient_id=str(linha.patient_id),
            in_bed=None if pd.isna(in_bed) else bool(in_bed),
            torso_angle=float(linha.torso_angle),
            centroid_x=float(linha.centroid_x),
            centroid_y=float(linha.centroid_y),
            velocidade=float(linha.velocidade),
        )


def analisar_quadro(cfg: AnomaliaCfg, quadro: pd.DataFrame) -> list:
    """Roda o detector numa série inteira e devolve os Findings."""
    detector = DetectorMovimento(cfg, str(quadro["patient_id"].iloc[0]))
    findings = []
    for ponto in pontos_do_quadro(quadro):
        findings += detector.update(ponto)
    findings += detector.finalizar()
    return findings


if __name__ == "__main__":
    import argparse

    from contracts.finding import DEMO_TZ
    from modules.anomaly.config import carregar_config, carregar_config_simulador
    from modules.anomaly.simulator import gerar_cenario_movimento

    parser = argparse.ArgumentParser(description="Roda o detector de movimentação num cenário simulado")
    parser.add_argument("--cenario", default="movimento_demo")
    parser.add_argument("--paciente", default="P001")
    args = parser.parse_args()

    inicio = datetime(2026, 11, 3, 8, 30, tzinfo=DEMO_TZ)
    quadro, rotulos = gerar_cenario_movimento(carregar_config_simulador(), args.cenario, args.paciente, inicio)
    print(f"série: {len(quadro)} amostras, {quadro['timestamp'].iloc[0]:%H:%M} a {quadro['timestamp'].iloc[-1]:%H:%M}")
    print("\nrótulos (o que foi injetado):")
    for r in rotulos:
        marca = "anomalia" if r.anomalia else "distrator"
        print(f"  {r.inicio:%H:%M:%S} a {r.fim:%H:%M:%S}  {r.tipo:<24}{marca}")
    print("\nFindings:")
    for f in analisar_quadro(carregar_config(), quadro):
        print(f"  {f.timestamp_start:%H:%M:%S} a {f.timestamp_end:%H:%M:%S}  {f.type:<24}{f.severity:<8}{f.score:.2f}  {f.description}")
