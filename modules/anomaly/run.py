"""Linha de comando do módulo de anomalias.

    python -m modules.anomaly.run --input outputs/video/P001/movement.parquet --patient P001 --out outputs/anomaly/P001/findings.jsonl

Grava, na pasta do --out: findings.jsonl, series_flagged.parquet (para o dashboard)
e, quando o movement.parquet é curto e foi estendido, composicao_movimento.json.
Sem --input a movimentação também é simulada.
"""
import argparse
from collections import Counter
from pathlib import Path

from modules.anomaly.config import carregar_config, carregar_config_simulador
from modules.anomaly.pipeline import SAIDA_PADRAO, executar, preparar_series, salvar


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detecta anomalias de vitais, prescrições e movimentação de um paciente")
    parser.add_argument("--input", help="movement.parquet do vídeo do leito (sem isso, a movimentação é simulada)")
    parser.add_argument("--patient", required=True, help="id do paciente, ex.: P001")
    parser.add_argument("--out", help="arquivo findings.jsonl (padrão: outputs/anomaly/<paciente>/findings.jsonl)")
    parser.add_argument("--cenario", help="cenário do simulator.yaml (padrão: o do paciente)")
    args = parser.parse_args(argv)

    saida = Path(args.out) if args.out else SAIDA_PADRAO / args.patient / "findings.jsonl"
    cfg, sim = carregar_config(), carregar_config_simulador()
    series = preparar_series(args.patient, args.input, cenario=args.cenario, cfg_sim=sim)
    findings, monitor = executar(cfg, series)
    caminhos = salvar(series, findings, monitor, saida.parent)
    if saida.name != "findings.jsonl":  # o --out pode ter outro nome de arquivo
        caminhos["findings"].replace(saida)
        caminhos["findings"] = saida

    print(f"{args.patient}: série de {series.inicio:%H:%M} a {series.fim:%H:%M}, cenário {series.cenario or 'estável'}")
    for f in findings:
        print(f"  {f.timestamp_start:%H:%M:%S}  {f.source:<13}{f.type:<24}{f.severity:<9}{f.score:.2f}")
    por_fonte = Counter(f.source for f in findings)
    print(f"{len(findings)} Findings ({', '.join(f'{n} {s}' for s, n in sorted(por_fonte.items())) or 'nenhum'})")
    for nome, caminho in caminhos.items():
        print(f"  {nome}: {caminho}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
