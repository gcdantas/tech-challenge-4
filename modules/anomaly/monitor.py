"""Monitor de um paciente: reúne os detectores de vitais, prescrições e movimentação.

É o ponto de entrada do monitoramento em fluxo. O Streamlit (ou qualquer
laço) chama `update(evento)` com cada leitura que chega e recebe os Findings
novos, se houver. O tipo do evento diz para qual detector ele vai:

  PontoVital        -> vitais (limites, z-score e Isolation Forest)
  PontoMovimento    -> movimentação (saída do leito e imobilidade)
  EventoPrescricao  -> prescrições (regras)

Os eventos precisam chegar em ordem de tempo. O módulo `fluxo` já entrega assim.
O monitor só produz Findings. Quem emite alerta é a fusão do Integrante 4.
"""
from modules.anomaly.config import AnomaliaCfg
from modules.anomaly.episodios import Episodio
from modules.anomaly.movement import DetectorMovimento, PontoMovimento
from modules.anomaly.multivariado import MonitorVitais
from modules.anomaly.prescription import DetectorPrescricoes, EventoPrescricao
from modules.anomaly.vitals import PontoVital

Evento = PontoVital | PontoMovimento | EventoPrescricao


class MonitorPaciente:
    def __init__(self, cfg: AnomaliaCfg, patient_id: str):
        self.patient_id = patient_id
        self.vitais = MonitorVitais(cfg, patient_id)
        self.movimento = DetectorMovimento(cfg, patient_id)
        self.prescricoes = DetectorPrescricoes(cfg, patient_id)
        self.eventos_recebidos = 0

    def update(self, evento: Evento) -> list:
        if isinstance(evento, PontoVital):
            findings = self.vitais.update(evento)
            # o tempo passou: uma suspensão de medicamento pode ter passado da janela sem substituto
            findings += self.prescricoes.avancar_relogio(evento.timestamp)
        elif isinstance(evento, PontoMovimento):
            findings = self.movimento.update(evento)
            findings += self.prescricoes.avancar_relogio(evento.timestamp)
        elif isinstance(evento, EventoPrescricao):
            findings = self.prescricoes.update(evento)
        else:
            raise TypeError(f"evento de tipo desconhecido: {type(evento).__name__}")
        self.eventos_recebidos += 1
        return findings

    def finalizar(self) -> list:
        """Fim da série: fecha os episódios abertos. Devolve os Findings que isso ainda gerar."""
        self.vitais.finalizar()
        return self.movimento.finalizar()

    def episodios(self) -> dict[str, list[Episodio]]:
        """Episódios que geraram Finding, por fonte. Servem para marcar as séries no dashboard."""
        return {"vitals": self.vitais.episodios(), "movement": self.movimento.episodios()}

    def descartes(self) -> dict[str, dict[str, int]]:
        """Por que leituras foram ignoradas, por fonte. Útil para ver se a entrada está com problema."""
        return {
            "vitals": dict(self.vitais.sinais.descartes),
            "movement": dict(self.movimento.descartes),
            "prescription": dict(self.prescricoes.descartes),
        }
