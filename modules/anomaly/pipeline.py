"""Pipeline do módulo de anomalias: da entrada aos Findings.

    analyze(input, patient_id) -> list[Finding]

`input` é o movement.parquet do vídeo do leito (Integrante 1). Os sinais vitais e
as prescrições do paciente não vêm de arquivo: são SINTÉTICOS, gerados pelo
simulador a partir do cenário do paciente em config/simulator.yaml (a fonte real
das prescrições ainda não foi definida). Sem `input`, a movimentação também é simulada.

Horários: a série de vitais e de prescrições começa no instante do artefato
`vitais` do paciente em data/patients.yaml (demo_reference + demo_offset). O
movement.parquet já vem com os horários da demo.
"""
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from contracts.finding import Finding, write_findings
from contracts.patients import DEFAULT_PATIENTS_PATH, REPO_ROOT, load_patients
from modules.anomaly import movement
from modules.anomaly.common import SEVERIDADES
from modules.anomaly.config import (
    SINAIS,
    AnomaliaCfg,
    SimuladorCfg,
    carregar_config,
    carregar_config_simulador,
)
from modules.anomaly.fluxo import intercalar
from modules.anomaly.monitor import Evento, MonitorPaciente
from modules.anomaly.prescription import EventoPrescricao
from modules.anomaly.simulator import (
    Rotulo,
    compor_movimento,
    gerar_cenario_movimento,
    gerar_cenario_vitais,
    gerar_movimento,
    gerar_prescricoes,
    gerar_vitais,
)
from modules.anomaly.vitals import pontos_do_quadro as pontos_vitais

SAIDA_PADRAO = REPO_ROOT / "outputs" / "anomaly"
SINAIS_MOVIMENTO = ("torso_angle", "centroid_x", "centroid_y", "velocidade", "in_bed")


@dataclass
class SeriesPaciente:
    """Tudo que o monitor vai receber, mais os rótulos de verdade (só existem nas partes simuladas)."""

    patient_id: str
    cenario: str | None
    inicio: datetime
    fim: datetime
    vitais: pd.DataFrame
    movimento: pd.DataFrame
    prescricoes: list[EventoPrescricao]
    rotulos: list[Rotulo] = field(default_factory=list)
    composicao: dict[str, Any] | None = None  # só quando o movement.parquet real foi estendido
    movimento_e_mock: bool = False


def _arquivo_e_mock(caminho: Path) -> bool:
    """O movement.parquet do I1 marca `mock` nos metadados do arquivo quando é de mentira."""
    try:
        import pyarrow.parquet as pq

        metadados = pq.read_schema(caminho).metadata or {}
        return any(k.decode().lower() == "mock" and v.decode().strip().lower() == "true" for k, v in metadados.items())
    except Exception:  # sem pyarrow ou sem metadado: trata como dado real
        return False


def preparar_series(
    patient_id: str,
    input: str | Path | pd.DataFrame | None = None,
    *,
    cenario: str | None = None,
    cfg_sim: SimuladorCfg | None = None,
    patients_path: str | Path = DEFAULT_PATIENTS_PATH,
) -> SeriesPaciente:
    cfg_sim = cfg_sim or carregar_config_simulador()
    patients = load_patients(patients_path)

    nome_artefato = next(iter(patients.patient(patient_id).artifacts_of("vitals")), None)
    inicio = patients.artifact_time(patient_id, nome_artefato) if nome_artefato else patients.demo_reference
    nome_cenario = cenario or cfg_sim.pacientes.get(patient_id)
    cen = cfg_sim.cenarios[nome_cenario] if nome_cenario else None
    if nome_cenario and nome_cenario not in cfg_sim.cenarios:
        raise KeyError(f"cenário {nome_cenario!r} não existe em simulator.yaml (tem: {', '.join(cfg_sim.cenarios)})")

    # vitais
    if nome_cenario:
        vitais, rotulos = gerar_cenario_vitais(cfg_sim, nome_cenario, patient_id, inicio)
    else:
        vitais, rotulos = gerar_vitais(cfg_sim, patient_id, inicio), []

    # prescrições
    prescricoes: list[EventoPrescricao] = []
    if cen is not None:
        prescricoes, rot_presc = gerar_prescricoes(cfg_sim, patient_id, inicio, cen.prescricoes)
        rotulos += rot_presc

    # movimentação: o parquet real, estendido com a série simulada, ou tudo simulado
    composicao, e_mock = None, False
    if input is not None:
        if isinstance(input, pd.DataFrame):
            clipe = input
        else:
            caminho = Path(input)
            if not caminho.exists():
                raise FileNotFoundError(f"movement.parquet não encontrado: {caminho}")
            clipe = pd.read_parquet(caminho)
            e_mock = _arquivo_e_mock(caminho)
        if str(clipe["patient_id"].iloc[0]) != patient_id:
            raise ValueError(f"o movement.parquet é do paciente {clipe['patient_id'].iloc[0]!r}, não de {patient_id!r}")
        duracao_clipe_min = (clipe["timestamp"].max() - clipe["timestamp"].min()).total_seconds() / 60
        if duracao_clipe_min < cfg_sim.movimento.clipe_curto_min:
            mov, composicao = compor_movimento(cfg_sim, patient_id, clipe, inicio)
        else:
            mov = clipe.sort_values("timestamp").reset_index(drop=True)
    elif cen is not None:
        mov, rot_mov = gerar_cenario_movimento(cfg_sim, nome_cenario, patient_id, inicio)
        rotulos += rot_mov
    else:
        mov, _ = gerar_movimento(cfg_sim, patient_id, inicio)

    fim = max(vitais["timestamp"].iloc[-1], mov["timestamp"].iloc[-1]).to_pydatetime()
    return SeriesPaciente(
        patient_id, nome_cenario, inicio, fim, vitais, mov, prescricoes, rotulos, composicao, e_mock
    )


def eventos_da_serie(series: SeriesPaciente) -> list[Evento]:
    """Todos os eventos da série numa fila só, em ordem de tempo (a entrada do monitor)."""
    return list(
        intercalar(
            series.prescricoes,
            pontos_vitais(series.vitais),
            movement.pontos_do_quadro(series.movimento),
        )
    )


def executar(cfg: AnomaliaCfg, series: SeriesPaciente) -> tuple[list[Finding], MonitorPaciente]:
    """Roda o monitor na série inteira e devolve os Findings em ordem de tempo, mais o monitor (com os episódios)."""
    monitor = MonitorPaciente(cfg, series.patient_id)
    findings: list[Finding] = []
    for evento in eventos_da_serie(series):
        findings += monitor.update(evento)
    findings += monitor.finalizar()

    if series.movimento_e_mock:
        for f in findings:
            if f.source == "movement":
                f.evidence["mock"] = True
    findings.sort(key=lambda f: (f.timestamp_start, f.timestamp_end))
    return findings, monitor


def analyze(
    input: str | Path | pd.DataFrame | None,
    patient_id: str,
    *,
    cenario: str | None = None,
    cfg: AnomaliaCfg | None = None,
    cfg_sim: SimuladorCfg | None = None,
    patients_path: str | Path = DEFAULT_PATIENTS_PATH,
) -> list[Finding]:
    """Interface do módulo: Findings de vitais, prescrições e movimentação do paciente."""
    cfg = cfg or carregar_config()
    series = preparar_series(patient_id, input, cenario=cenario, cfg_sim=cfg_sim, patients_path=patients_path)
    return executar(cfg, series)[0]


# ---------------------------------------------------------------------- séries marcadas (dashboard)


def series_marcadas(series: SeriesPaciente, findings: list[Finding], monitor: MonitorPaciente) -> pd.DataFrame:
    """Série em formato longo, com cada amostra marcada se caiu dentro de um episódio que gerou Finding.

    Colunas: patient_id, timestamp, source (vitals ou movement), signal, value,
    flagged, finding_id e severity (os dois últimos vazios quando não marcada).
    Prescrições não entram: são eventos, e já estão no findings.jsonl.
    """
    # episódio -> Finding mais grave daquele episódio
    por_episodio: dict[str, Finding] = {}
    for f in findings:
        ep_id = f.evidence.get("episodio_id")
        if ep_id is None:
            continue
        atual = por_episodio.get(ep_id)
        if atual is None or SEVERIDADES.index(f.severity) >= SEVERIDADES.index(atual.severity):
            por_episodio[ep_id] = f

    epoca = pd.Timestamp("1970-01-01", tz="UTC")
    partes = []

    def montar(quadro: pd.DataFrame, source: str, sinais: tuple[str, ...], episodios) -> None:
        segundos = (quadro["timestamp"] - epoca).dt.total_seconds().to_numpy()
        for sinal in sinais:
            valor = quadro[sinal].astype(float)
            marcada = pd.Series(False, index=quadro.index)
            finding_id = pd.Series(None, index=quadro.index, dtype=object)
            severidade = pd.Series(None, index=quadro.index, dtype=object)
            for ep in episodios:
                if ep.origem not in (sinal, "multivariado", "saida_leito", "imobilidade"):
                    continue
                f = por_episodio.get(ep.id)
                if f is None:
                    continue
                dentro = (segundos >= ep.inicio) & (segundos <= (ep.fim if ep.fim is not None else ep.ultimo_ativo))
                marcada |= dentro
                finding_id[dentro] = str(f.finding_id)
                severidade[dentro] = f.severity
            partes.append(
                pd.DataFrame(
                    {
                        "patient_id": series.patient_id,
                        "timestamp": quadro["timestamp"],
                        "source": source,
                        "signal": sinal,
                        "value": valor,
                        "flagged": marcada,
                        "finding_id": finding_id,
                        "severity": severidade,
                    }
                )
            )

    episodios = monitor.episodios()
    montar(series.vitais, "vitals", SINAIS, episodios["vitals"])
    montar(series.movimento, "movement", SINAIS_MOVIMENTO, episodios["movement"])
    return pd.concat(partes, ignore_index=True)


# ---------------------------------------------------------------------- saída em disco


def salvar(
    series: SeriesPaciente,
    findings: list[Finding],
    monitor: MonitorPaciente,
    pasta: str | Path,
) -> dict[str, Path]:
    """Grava findings.jsonl, series_flagged.parquet e, se houve composição, composicao_movimento.json."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    caminhos = {"findings": write_findings(pasta / "findings.jsonl", findings)}

    caminho_series = pasta / "series_flagged.parquet"
    series_marcadas(series, findings, monitor).to_parquet(caminho_series, index=False)
    caminhos["series"] = caminho_series

    if series.composicao is not None:
        caminho_comp = pasta / "composicao_movimento.json"
        caminho_comp.write_text(json.dumps(series.composicao, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        caminhos["composicao"] = caminho_comp
    return caminhos
