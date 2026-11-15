"""Detector de anomalias nas prescrições, por regras determinísticas.

Recebe os eventos do prontuário um a um (`update`) e devolve Findings com
source "prescription". São quatro regras, com limites e catálogo de
medicamentos em config/anomaly.yaml (valores didáticos, não são bula):

  dose_fora_da_faixa   a dose por tomada ou a dose diária passa do máximo do
                       catálogo (ou fica muito abaixo do mínimo)
  dose_duplicada       pedido de um medicamento que já tem pedido ativo
  aumento_abrupto_dose a dose diária sobe demais em relação ao pedido anterior
  suspensao_abrupta    medicamento de uso contínuo suspenso e ninguém pede de novo

A suspensão abrupta só pode ser afirmada depois de esperar a janela de
substituição. Por isso existe `avancar_relogio`: quem usa o detector avisa que o
tempo passou, mesmo sem evento novo.
"""
import math
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from modules.anomaly.common import SCORE_NO_LIMITE, classificar, novo_finding
from modules.anomaly.config import AnomaliaCfg, MedicamentoCfg

MODELO = "prescricoes-regras"
VERSAO = "1.0"

# Quantas vezes o máximo: acima de 1 já é medium, 2x é high e 5x é critical
LIMITES_RAZAO = [(1.01, "medium"), (2.0, "high"), (5.0, "critical")]
# Dose muito abaixo do mínimo só merece um aviso leve
FRACAO_MINIMA = 0.5


@dataclass(frozen=True)
class EventoPrescricao:
    timestamp: datetime  # com fuso horário
    patient_id: str
    acao: Literal["prescrever", "suspender"]
    ordem_id: str
    medicamento: str | None = None  # só em "prescrever"
    dose_mg: float | None = None  # por tomada
    doses_por_dia: float | None = None


@dataclass
class _Ordem:
    id: str
    chave: str
    cfg: MedicamentoCfg
    dose_mg: float
    doses_por_dia: float
    inicio: datetime
    fim: datetime | None = None

    @property
    def dose_diaria_mg(self) -> float:
        return self.dose_mg * self.doses_por_dia


def normalizar_nome(texto: str) -> str:
    """'Dipirona ' vira 'dipirona' e 'Ácido' vira 'acido': é a chave do catálogo."""
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return sem_acento.strip().lower()


def ordenar_eventos(eventos: list[EventoPrescricao]) -> list[EventoPrescricao]:
    """Ordem do tempo. Na mesma hora a suspensão vem antes do pedido novo, para uma troca não parecer duplicidade."""
    return sorted(eventos, key=lambda e: (e.timestamp, e.acao != "suspender"))


def _num(valor: float) -> str:
    """1000.0 vira '1000' e 1.5 vira '1,5' (vírgula decimal, como no português)."""
    return f"{valor:g}".replace(".", ",")


class DetectorPrescricoes:
    def __init__(self, cfg: AnomaliaCfg, patient_id: str):
        self.cfg = cfg.prescricoes
        self.patient_id = patient_id
        self.ordens: dict[str, _Ordem] = {}
        self.ativas: dict[str, list[_Ordem]] = {}  # por medicamento
        self.encerradas: dict[str, list[_Ordem]] = {}
        self.pendentes: list[tuple[_Ordem, datetime]] = []  # suspensões esperando a janela
        self.relogio: datetime | None = None
        self.descartes: Counter = Counter()

    # ------------------------------------------------------------------ entrada

    def update(self, evento: EventoPrescricao) -> list:
        if evento.patient_id != self.patient_id:
            raise ValueError(f"evento do paciente {evento.patient_id!r} enviado ao detector de {self.patient_id!r}")
        if evento.timestamp.tzinfo is None:
            raise ValueError("timestamp precisa ter fuso horário")
        if self.relogio is not None and evento.timestamp < self.relogio:
            self.descartes["fora_de_ordem"] += 1
            return []

        findings = self.avancar_relogio(evento.timestamp)
        if evento.acao == "prescrever":
            findings += self._prescrever(evento)
        else:
            self._suspender(evento)
        return findings

    def avancar_relogio(self, agora: datetime) -> list:
        """Avisa que o tempo chegou em `agora`. Emite as suspensões abruptas cuja janela terminou."""
        if self.relogio is None or agora > self.relogio:
            self.relogio = agora
        janela = timedelta(hours=self.cfg.suspensao.janela_substituicao_h)
        findings, ainda_pendentes = [], []
        for ordem, quando in self.pendentes:
            if agora - quando >= janela:
                findings.append(self._finding_suspensao(ordem, quando, quando + janela))
            else:
                ainda_pendentes.append((ordem, quando))
        self.pendentes = ainda_pendentes
        return findings

    # ------------------------------------------------------------------ regras

    def _prescrever(self, ev: EventoPrescricao) -> list:
        if ev.medicamento is None or ev.dose_mg is None or ev.doses_por_dia is None:
            self.descartes["evento_incompleto"] += 1
            return []
        if not (math.isfinite(ev.dose_mg) and math.isfinite(ev.doses_por_dia)) or ev.dose_mg <= 0 or ev.doses_por_dia <= 0:
            self.descartes["dose_invalida"] += 1
            return []
        chave = normalizar_nome(ev.medicamento)
        catalogo = self.cfg.medicamentos.get(chave)
        if catalogo is None:
            self.descartes["medicamento_fora_do_catalogo"] += 1
            return []

        ordem = _Ordem(ev.ordem_id, chave, catalogo, ev.dose_mg, ev.doses_por_dia, ev.timestamp)
        outras_ativas = list(self.ativas.get(chave, []))
        findings = []

        f = self._regra_faixa(ordem)
        if f:
            findings.append(f)
        if outras_ativas:
            findings.append(self._regra_duplicada(ordem, outras_ativas))
        else:
            # com pedido ativo a comparação não faz sentido: o problema já é a duplicidade
            f = self._regra_aumento(ordem)
            if f:
                findings.append(f)

        # chegou um pedido novo do mesmo medicamento: a suspensão pendente era uma troca
        self.pendentes = [(o, q) for o, q in self.pendentes if o.chave != chave]
        self.ordens[ordem.id] = ordem
        self.ativas.setdefault(chave, []).append(ordem)
        return findings

    def _suspender(self, ev: EventoPrescricao) -> None:
        ordem = self.ordens.get(ev.ordem_id)
        if ordem is None:
            self.descartes["ordem_desconhecida"] += 1
            return
        if ordem.fim is not None:
            self.descartes["ordem_ja_suspensa"] += 1
            return
        ordem.fim = ev.timestamp
        self.ativas[ordem.chave].remove(ordem)
        self.encerradas.setdefault(ordem.chave, []).append(ordem)

        usada_ha_h = (ev.timestamp - ordem.inicio).total_seconds() / 3600
        if ordem.cfg.continuo and usada_ha_h >= self.cfg.suspensao.min_uso_h:
            self.pendentes.append((ordem, ev.timestamp))

    def _regra_faixa(self, ordem: _Ordem):
        cat = ordem.cfg
        razao_dose = ordem.dose_mg / cat.dose_max_mg
        razao_diaria = ordem.dose_diaria_mg / cat.dose_diaria_max_mg
        razao = max(razao_dose, razao_diaria)

        c = classificar(razao, LIMITES_RAZAO, "maior")
        if c:
            severidade, score, _ = c
            if razao_dose >= razao_diaria:
                descricao = (
                    f"{cat.nome}: {_num(ordem.dose_mg)} mg por tomada, {_num(round(razao, 1))} vez(es) "
                    f"o máximo de {_num(cat.dose_max_mg)} mg (ordem {ordem.id})."
                )
            else:
                descricao = (
                    f"{cat.nome}: dose diária de {_num(ordem.dose_diaria_mg)} mg, {_num(round(razao, 1))} vez(es) "
                    f"o máximo de {_num(cat.dose_diaria_max_mg)} mg (ordem {ordem.id})."
                )
            return self._finding(ordem, "dose_fora_da_faixa", severidade, score, descricao, {"razao": round(razao, 2), "direcao": "acima"})

        if ordem.dose_mg < cat.dose_min_mg * FRACAO_MINIMA:
            descricao = (
                f"{cat.nome}: {_num(ordem.dose_mg)} mg por tomada, bem abaixo do mínimo de "
                f"{_num(cat.dose_min_mg)} mg (ordem {ordem.id})."
            )
            return self._finding(
                ordem, "dose_fora_da_faixa", "low", SCORE_NO_LIMITE["low"], descricao,
                {"razao": round(ordem.dose_mg / cat.dose_min_mg, 2), "direcao": "abaixo"},
            )
        return None

    def _regra_duplicada(self, ordem: _Ordem, ativas: list[_Ordem]):
        cat = ordem.cfg
        soma = ordem.dose_diaria_mg + sum(o.dose_diaria_mg for o in ativas)
        passou = soma > cat.dose_diaria_max_mg
        severidade = "high" if passou else "medium"
        descricao = (
            f"{cat.nome}: já existe pedido ativo do mesmo medicamento (ordem {ativas[0].id}); "
            f"a soma diária fica em {_num(soma)} mg"
            + (f", acima do máximo de {_num(cat.dose_diaria_max_mg)} mg." if passou else ".")
        )
        return self._finding(
            ordem, "dose_duplicada", severidade, SCORE_NO_LIMITE[severidade], descricao,
            {"ordens_ativas": [o.id for o in ativas], "soma_diaria_mg": soma},
        )

    def _regra_aumento(self, ordem: _Ordem):
        anteriores = [
            o for o in self.encerradas.get(ordem.chave, [])
            if ordem.inicio - o.fim <= timedelta(hours=self.cfg.aumento.janela_comparacao_h)
        ]
        if not anteriores:
            return None
        anterior = max(anteriores, key=lambda o: o.fim)
        pct = (ordem.dose_diaria_mg - anterior.dose_diaria_mg) / anterior.dose_diaria_mg * 100
        a = self.cfg.aumento
        c = classificar(pct, [(a.medium, "medium"), (a.high, "high"), (a.critical, "critical")], "maior")
        if not c:
            return None
        severidade, score, _ = c
        descricao = (
            f"{ordem.cfg.nome}: dose diária sobe de {_num(anterior.dose_diaria_mg)} mg para "
            f"{_num(ordem.dose_diaria_mg)} mg (+{pct:.0f} %) em relação ao pedido anterior (ordem {anterior.id})."
        )
        return self._finding(
            ordem, "aumento_abrupto_dose", severidade, score, descricao,
            {"ordem_anterior": anterior.id, "dose_diaria_anterior_mg": anterior.dose_diaria_mg, "aumento_pct": round(pct, 1)},
        )

    def _finding_suspensao(self, ordem: _Ordem, quando: datetime, fim: datetime):
        janela_min = self.cfg.suspensao.janela_substituicao_h * 60
        descricao = (
            f"{ordem.cfg.nome}: medicamento de uso contínuo suspenso (ordem {ordem.id}) "
            f"e sem novo pedido em {janela_min:.0f} min."
        )
        return self._finding(
            ordem, "suspensao_abrupta", "medium", SCORE_NO_LIMITE["medium"], descricao,
            {"janela_substituicao_min": janela_min}, inicio=quando, fim=fim,
        )

    # ------------------------------------------------------------------ saída

    def _finding(self, ordem: _Ordem, tipo, severidade, score, descricao, extra, inicio=None, fim=None):
        evidencia = {
            "regra": tipo,
            "medicamento": ordem.cfg.nome,
            "ordem_id": ordem.id,
            "dose_mg": ordem.dose_mg,
            "doses_por_dia": ordem.doses_por_dia,
            "dose_diaria_mg": ordem.dose_diaria_mg,
            "dose_max_mg": ordem.cfg.dose_max_mg,
            "dose_diaria_max_mg": ordem.cfg.dose_diaria_max_mg,
            **extra,
        }
        inicio = inicio or ordem.inicio
        return novo_finding(
            patient_id=self.patient_id,
            source="prescription",
            tipo=tipo,
            inicio=inicio,
            fim=fim or inicio,
            score=score,
            severidade=severidade,
            descricao=descricao,
            evidencia=evidencia,
            modelo=MODELO,
            versao=VERSAO,
        )


def analisar_eventos(cfg: AnomaliaCfg, eventos: list[EventoPrescricao], ate: datetime | None = None) -> list:
    """Roda o detector numa lista de eventos (modo em lote).

    `ate` é até quando o prontuário foi observado. Sem ele, uma suspensão recente
    fica sem veredito, porque ainda não deu tempo de esperar a janela.
    """
    if not eventos:
        return []
    detector = DetectorPrescricoes(cfg, eventos[0].patient_id)
    findings = []
    for ev in ordenar_eventos(eventos):
        findings += detector.update(ev)
    if ate is not None:
        findings += detector.avancar_relogio(ate)
    return findings
