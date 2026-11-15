"""Gráficos do módulo de anomalias, para o relatório e para a gravação do vídeo.

Gera PNGs com as anomalias marcadas (vitais, movimentação, prescrições) e com as
métricas da avaliação. Só usa matplotlib e grava arquivos, sem abrir janela.

    python -m modules.anomaly.graficos --out outputs/anomaly/graficos      # só os exemplos
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from contracts.finding import DEMO_TZ  # noqa: E402
from modules.anomaly import movement  # noqa: E402
from modules.anomaly.config import (  # noqa: E402
    SINAIS,
    InjecaoCfg,
    carregar_config,
    carregar_config_simulador,
)
from modules.anomaly.multivariado import analisar_quadro_completo  # noqa: E402
from modules.anomaly.prescription import analisar_eventos  # noqa: E402
from modules.anomaly.simulator import (  # noqa: E402
    gerar_cenario_movimento,
    gerar_prescricoes,
    gerar_vitais,
    rotulos_das_injecoes,
)

REFERENCIA = datetime(2026, 11, 3, 10, 0, tzinfo=DEMO_TZ)

# Cores: azul e laranja são os dois primeiros da paleta categórica; o resto é tinta e cinza
AZUL, LARANJA, VERDE = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA_2, GRADE, FUNDO = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _estilo() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": FUNDO,
            "axes.facecolor": FUNDO,
            "savefig.facecolor": FUNDO,
            "axes.edgecolor": GRADE,
            "axes.labelcolor": TINTA_2,
            "xtick.color": TINTA_2,
            "ytick.color": TINTA_2,
            "text.color": TINTA,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRADE,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "legend.frameon": False,
            "lines.linewidth": 1.4,
        }
    )


def _formatar_horas(ax) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=DEMO_TZ))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(tz=DEMO_TZ, minticks=5, maxticks=9))


def _salvar(fig, caminho: Path) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return caminho


def _faixas_de_verdade(ax, rotulos, rotulado: set) -> None:
    """Pinta o intervalo de cada evento injetado: laranja para anomalia, cinza hachurado para distrator."""
    for r in rotulos:
        if r.fonte not in ("vitals", "movement"):
            continue
        fim = max(r.fim, r.inicio + timedelta(seconds=40))  # evento de 2 s ainda precisa aparecer
        if r.anomalia:
            ax.axvspan(r.inicio, fim, color=LARANJA, alpha=0.18, lw=0, label="_" if "a" in rotulado else "Evento injetado")
            rotulado.add("a")
        else:
            ax.axvspan(r.inicio, fim, facecolor="none", edgecolor=TINTA_2, hatch="////", alpha=0.45, lw=0)


def _linhas_de_finding(ax, findings, fontes=None) -> None:
    for f in findings:
        if fontes is None or f.source in fontes:
            ax.axvline(f.timestamp_end, color=TINTA, lw=1.1, alpha=0.9)


def _legenda_verdade(fig, extra=()) -> None:
    itens = [
        Patch(facecolor=LARANJA, alpha=0.35, label="Evento injetado (verdade)"),
        Patch(facecolor="none", edgecolor=TINTA_2, hatch="////", label="Distrator (não é problema)"),
        Line2D([0], [0], color=TINTA, lw=1.2, label="Finding emitido"),
        *extra,
    ]
    fig.legend(handles=itens, loc="upper center", ncol=len(itens), bbox_to_anchor=(0.5, 0.965), fontsize=9)


# ---------------------------------------------------------------------- exemplos com anomalias marcadas


def grafico_exemplo_vitais(cfg, sim, caminho: Path) -> Path:
    """Dois pacientes simulados de 2 h com quatro anomalias e um distrator, com os Findings marcados."""
    _estilo()
    injecoes = [
        InjecaoCfg(tipo="dessaturacao", inicio_min=25, duracao_min=3, magnitude=10, rampa_s=40),
        InjecaoCfg(tipo="taquicardia", inicio_min=50, duracao_min=5, magnitude=40, rampa_s=60),
        InjecaoCfg(tipo="artefato_pico", inicio_min=72, duracao_min=0.4, magnitude=12),
        InjecaoCfg(tipo="deterioracao_multivariada", inicio_min=90, duracao_min=10, magnitude=1.3, rampa_s=120),
    ]
    quadro = gerar_vitais(sim, "EXEMPLO", REFERENCIA, duracao_min=120, variar_paciente=True, injecoes=injecoes)
    findings = analisar_quadro_completo(cfg, quadro)
    rotulos = rotulos_das_injecoes(REFERENCIA, injecoes)

    fig, eixos = plt.subplots(len(SINAIS), 1, figsize=(10, 9), sharex=True)
    rotulado: set = set()
    for ax, sinal in zip(eixos, SINAIS):
        cfg_s = cfg.vitais.sinais[sinal]
        ax.plot(quadro["timestamp"], quadro[sinal], color=AZUL, lw=1.2)
        _faixas_de_verdade(ax, rotulos, rotulado)
        _linhas_de_finding(ax, findings)
        for lado in (cfg_s.baixo, cfg_s.alto):  # primeiro limite clínico, tracejado
            if lado is not None:
                ax.axhline(lado.medium, color=TINTA_2, lw=0.8, ls="--")
        ax.set_ylabel(f"{sinal.upper() if sinal != 'temp' else 'Temp.'}\n({cfg_s.unidade})", fontsize=8)
    _formatar_horas(eixos[-1])
    eixos[-1].set_xlabel("Horário da série simulada")
    fig.suptitle("Sinais vitais simulados com anomalias marcadas", x=0.125, ha="left", fontweight="bold", y=1.0)
    _legenda_verdade(fig, extra=[Line2D([0], [0], color=TINTA_2, lw=0.8, ls="--", label="Primeiro limite clínico")])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _salvar(fig, caminho)


def grafico_exemplo_movimento(cfg, sim, caminho: Path) -> Path:
    _estilo()
    quadro, rotulos = gerar_cenario_movimento(sim, "movimento_demo", "EXEMPLO", REFERENCIA)
    findings = movement.analisar_quadro(cfg, quadro)

    fig, eixos = plt.subplots(3, 1, figsize=(10, 6.2), sharex=True, height_ratios=[2, 2, 1])
    rotulado: set = set()
    series = [("torso_angle", "Ângulo do\ntronco (°)"), ("centroid_x", "Centro da pessoa\n(x, 0 a 1)")]
    for ax, (coluna, nome) in zip(eixos[:2], series):
        ax.plot(quadro["timestamp"], quadro[coluna], color=AZUL, lw=0.8)
        ax.set_ylabel(nome, fontsize=8)
    eixos[2].fill_between(quadro["timestamp"], quadro["in_bed"].astype(int), step="post", color=AZUL, alpha=0.5, lw=0)
    eixos[2].set_ylabel("No leito", fontsize=8)
    eixos[2].set_yticks([0, 1], ["não", "sim"])
    for ax in eixos:
        _faixas_de_verdade(ax, rotulos, rotulado)
        _linhas_de_finding(ax, findings)
    # imobilidade: o Finding sai depois de 2 h e o episódio continua; mostra o trecho até o fim do Finding
    for f in findings:
        if f.type == "imobilidade_prolongada":
            eixos[0].annotate(
                f"imobilidade {f.severity}", (f.timestamp_end, quadro["torso_angle"].max()), xytext=(-4, -2),
                textcoords="offset points", ha="right", va="top", fontsize=8, color=TINTA,
            )
    _formatar_horas(eixos[-1])
    eixos[-1].set_xlabel("Horário da série simulada")
    fig.suptitle("Movimentação do leito: saída do leito e imobilidade prolongada", x=0.125, ha="left", fontweight="bold", y=1.0)
    _legenda_verdade(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _salvar(fig, caminho)


def grafico_exemplo_prescricoes(cfg, sim, caminho: Path) -> Path:
    """Pedidos ativos de cada medicamento ao longo do tempo, com os Findings de prescrição marcados."""
    _estilo()
    eventos, rotulos = gerar_prescricoes(sim, "EXEMPLO", REFERENCIA, sim.cenarios["prescricoes_demo"].prescricoes)
    findings = analisar_eventos(cfg, eventos, ate=REFERENCIA + timedelta(minutes=90))

    ordens: dict[str, dict] = {}
    for e in eventos:
        if e.acao == "prescrever":
            ordens[e.ordem_id] = {
                "med": cfg.prescricoes.medicamentos[e.medicamento.lower()].nome,
                "inicio": e.timestamp,
                "fim": None,
                "diaria": e.dose_mg * e.doses_por_dia,
            }
        elif e.ordem_id in ordens:
            ordens[e.ordem_id]["fim"] = e.timestamp
    meds = sorted({o["med"] for o in ordens.values()})
    linha = {m: i for i, m in enumerate(meds)}
    por_ordem = {}
    for f in findings:
        por_ordem.setdefault(f.evidence["ordem_id"], []).append(f)

    fig, ax = plt.subplots(figsize=(10, 0.55 * len(meds) + 1.8))
    esquerda, direita = REFERENCIA - timedelta(minutes=15), REFERENCIA + timedelta(minutes=90)
    vistos: dict[str, int] = {}  # quantos pedidos do mesmo medicamento já foram desenhados: eles alternam acima e abaixo
    for oid, o in ordens.items():
        vistos[o["med"]] = vistos.get(o["med"], 0) + 1
        y = linha[o["med"]] + (0.18 if vistos[o["med"]] % 2 == 0 else -0.18)
        ini, fim = max(o["inicio"], esquerda), min(o["fim"] or direita, direita)
        marcado = oid in por_ordem
        ax.barh(y, (fim - ini).total_seconds() / 86400, left=mdates.date2num(ini), height=0.3,
                color=LARANJA if marcado else AZUL, alpha=0.9 if marcado else 0.55)
        ax.text(mdates.date2num(ini) + 0.0008, y, f"{o['diaria']:g} mg/dia", va="center", fontsize=7.5, color=TINTA)
        for f in por_ordem.get(oid, []):
            ax.annotate(f.type, (mdates.date2num(f.timestamp_start), y), xytext=(0, 9), textcoords="offset points",
                        fontsize=7.5, color=TINTA, ha="left")
    ax.set_yticks(range(len(meds)), meds)
    ax.set_ylim(-0.7, len(meds) - 0.3)
    ax.set_xlim(mdates.date2num(esquerda), mdates.date2num(direita))
    ax.xaxis_date(tz=DEMO_TZ)
    _formatar_horas(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Horário da série simulada")
    ax.set_title("Pedidos de medicamento ativos e Findings de prescrição")
    ax.legend(handles=[Patch(color=AZUL, alpha=0.55, label="Pedido sem Finding"), Patch(color=LARANJA, label="Pedido com Finding")],
              loc="lower right", bbox_to_anchor=(1, 1.01), ncol=2, fontsize=8)
    fig.tight_layout()
    return _salvar(fig, caminho)


# ---------------------------------------------------------------------- gráficos da avaliação


def grafico_metricas(metricas: dict, caminho: Path) -> Path:
    _estilo()
    from modules.anomaly.avaliacao import NOMES

    modulos = [(k, m) for k, m in metricas["modulos"].items() if k != "vitais_if"]
    nomes = [NOMES[k].replace("Vitais: ", "Vitais\n").replace(" (regras)", "\n(regras)") for k, _ in modulos]
    x = np.arange(len(modulos))
    largura = 0.26
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for k, (rotulo, chave, cor, ic) in enumerate(
        [("Precisão", "precisao", AZUL, "precisao_ic95"), ("Recall", "recall", LARANJA, "recall_ic95"), ("F1", "f1", VERDE, None)]
    ):
        valores = np.array([m[chave] if m[chave] is not None else 0 for _, m in modulos])
        barras = ax.bar(x + (k - 1) * largura, valores, largura * 0.92, color=cor, label=rotulo)
        if ic:
            ics = [m[ic] for _, m in modulos]
            erro = np.array([[v - i[0], i[1] - v] if i else [0, 0] for v, i in zip(valores, ics)]).T
            ax.errorbar(x + (k - 1) * largura, valores, yerr=np.clip(erro, 0, None), fmt="none", ecolor=TINTA, elinewidth=0.9, capsize=2.5)
        for b, v in zip(barras, valores):
            ax.text(b.get_x() + b.get_width() / 2, 0.03, f"{100 * v:.0f}", ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")
    ax.set_xticks(x, nomes)
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.25), ["0", "25", "50", "75", "100"])
    ax.set_ylabel("% (as linhas finas são o intervalo de confiança de 95 %)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Precisão, recall e F1 por módulo (dados sintéticos)")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    return _salvar(fig, caminho)


def grafico_sensibilidade(metricas: dict, caminho: Path) -> Path:
    _estilo()
    sens = metricas["sensibilidade_vitais"]
    tipos = list(sens["vitais_sinais"])
    unidades = {"dessaturacao": "queda de SpO2 (pontos)", "taquicardia": "aumento da FC (bpm)", "hipotensao": "queda da PAS (mmHg)",
                "febre": "aumento da temperatura (°C)", "taquipneia": "aumento da FR (irpm)", "deterioracao_multivariada": "intensidade (× padrão)"}
    fig, eixos = plt.subplots(2, 3, figsize=(10, 5.6), sharey=True)
    for ax, tipo in zip(eixos.ravel(), tipos):
        for variante, cor, marca in (("vitais_sinais", AZUL, "o"), ("vitais_if", LARANJA, "s")):
            pts = sens[variante][tipo]
            ax.plot([p["magnitude"] for p in pts], [p["recall"] for p in pts], color=cor, marker=marca, ms=5, mec=FUNDO, mew=1.2)
        ax.set_title(tipo.replace("_", " "))
        ax.set_xlabel(unidades.get(tipo, "magnitude"), fontsize=8)
        ax.set_ylim(-0.05, 1.08)
    for ax in eixos[:, 0]:
        ax.set_ylabel("Recall")
    n = metricas["pacientes"].get("varredura", 0)
    fig.suptitle(f"Sensibilidade: eventos achados conforme a intensidade ({n} pacientes simulados)", x=0.06, ha="left", fontweight="bold")
    fig.legend(
        handles=[Line2D([0], [0], color=AZUL, marker="o", label="Limites clínicos + z-score"), Line2D([0], [0], color=LARANJA, marker="s", label="Só Isolation Forest")],
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return _salvar(fig, caminho)


def grafico_atraso(metricas: dict, caminho: Path) -> Path:
    _estilo()
    from modules.anomaly.avaliacao import NOMES

    linhas = []
    for chave in ("vitais_completo", "prescricoes", "movimentacao"):
        for tipo, t in metricas["modulos"][chave]["por_tipo"].items():
            if t["atraso"]:
                linhas.append((NOMES[chave].split(":")[0].split(" (")[0], tipo, t["atraso"]["mediana_s"], t["atraso"]["p90_s"], t["detectados"]))
    fig, ax = plt.subplots(figsize=(10, 0.36 * len(linhas) + 1.6))
    y = np.arange(len(linhas))[::-1]
    ax.barh(y, [m for _, _, m, _, _ in linhas], height=0.55, color=AZUL, label="Mediana")
    ax.scatter([l[3] for l in linhas], y, marker="|", s=140, color=TINTA, zorder=3, label="Percentil 90")
    for yi, (_, _, mediana, p90, n) in zip(y, linhas):
        ax.text(max(p90, mediana) + 4, yi, f"{mediana:.0f} s (n={n})", va="center", fontsize=8, color=TINTA_2)
    ax.set_yticks(y, [f"{mod} · {tipo.replace('_', ' ')}" for mod, tipo, *_ in linhas])
    ax.set_xlabel("Atraso de detecção (segundos desde o início do evento)")
    ax.set_xlim(0, max(l[3] for l in linhas) * 1.25 + 10)
    ax.grid(axis="y", visible=False)
    ax.set_title("Atraso de detecção por tipo de evento")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _salvar(fig, caminho)


# ---------------------------------------------------------------------- conjunto


def gerar_exemplos(pasta: Path) -> list[Path]:
    cfg, sim = carregar_config(), carregar_config_simulador()
    pasta = Path(pasta)
    return [
        grafico_exemplo_vitais(cfg, sim, pasta / "exemplo_vitais.png"),
        grafico_exemplo_movimento(cfg, sim, pasta / "exemplo_movimento.png"),
        grafico_exemplo_prescricoes(cfg, sim, pasta / "exemplo_prescricoes.png"),
    ]


def gerar_todos(metricas: dict, pasta: Path) -> list[Path]:
    pasta = Path(pasta)
    return [
        *gerar_exemplos(pasta),
        grafico_metricas(metricas, pasta / "metricas_por_modulo.png"),
        grafico_sensibilidade(metricas, pasta / "sensibilidade_vitais.png"),
        grafico_atraso(metricas, pasta / "atraso_deteccao.png"),
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gráficos de exemplo com as anomalias marcadas")
    parser.add_argument("--out", default="outputs/anomaly/graficos")
    for caminho in gerar_exemplos(Path(parser.parse_args().out)):
        print(f"gráfico: {caminho}")
