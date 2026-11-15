"""Confere o cenário P001 de ponta a ponta com as saídas que estão em outputs/ (dono: I4).

    python -m scripts.e2e_p001              # relatório na tela; código 1 se o cenário falhar
    python -m scripts.e2e_p001 --json r.json

Não roda nenhum módulo: lê os findings.jsonl que cada integrante gerou, diz de
onde veio cada fonte (real ou mock), roda a fusão e confere o roteiro da seção
10.1 do README:
- todas as seis fontes presentes;
- um alerta crítico pela regra respiratório + vocal + SpO2;
- nenhum alerta crítico se a dessaturação sair da janela (controle negativo).
Não envia e-mail.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

from app.fusion import fuse, load_fusion_config
from app.loaders import load_findings, outputs_dir
from contracts.finding import SOURCES, Finding

CRITICAL_RULE = "respiratorio_vocal_spo2"


def origin(findings: list[Finding]) -> str:
    if not findings:
        return "ausente"
    mock = sum(bool(f.evidence.get("mock")) or f.model.name.startswith("mock") for f in findings)
    if mock == 0:
        return "real"
    return "mock" if mock == len(findings) else f"misto ({mock} de {len(findings)} mock)"


def check(out_dir: Path, patient_id: str = "P001") -> dict:
    loaded = load_findings(out_dir, patient_id)
    by_source: dict[str, list[Finding]] = defaultdict(list)
    for f in loaded.findings:
        by_source[f.source].append(f)
    cfg = load_fusion_config()
    alerts = fuse(loaded.findings, cfg)
    crit = [a for a in alerts if a.level == "critico" and a.rule == CRITICAL_RULE]

    # Controle negativo: dessaturações 2 h depois, fora da janela.
    shifted = [
        f.model_copy(update={"timestamp_start": f.timestamp_start + timedelta(hours=2),
                             "timestamp_end": f.timestamp_end + timedelta(hours=2)})
        if f.source == "vitals" and f.type == "dessaturacao" else f
        for f in loaded.findings
    ]
    negative = [a for a in fuse(shifted, cfg) if a.rule == CRITICAL_RULE]

    checks = {
        "seis fontes presentes": all(by_source.get(s) for s in SOURCES),
        "arquivos válidos": not loaded.errors,
        "alerta crítico pela regra": len(crit) >= 1,
        "controle negativo sem crítico da regra": not negative,
    }
    return {
        "patient_id": patient_id,
        "outputs": str(out_dir),
        "fontes": {
            s: {"origem": origin(by_source.get(s, [])), "findings": len(by_source.get(s, [])),
                "types": dict(Counter(f.type for f in by_source.get(s, [])))}
            for s in SOURCES
        },
        "alertas": [
            {"nivel": a.level, "horario": a.created_at.isoformat(), "regra": a.rule, "score": a.score,
             "findings": len(a.finding_ids), "deduplicados": len(a.suppressed_finding_ids)}
            for a in alerts
        ],
        "findings_total": len(loaded.findings),
        "erros": loaded.errors,
        "verificacoes": checks,
        "ok": all(checks.values()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Confere o cenário P001 com as saídas em outputs/")
    ap.add_argument("--outputs", type=Path, default=None, help="pasta de saídas (padrão: outputs/ ou TC_OUTPUTS_DIR)")
    ap.add_argument("--patient", default="P001")
    ap.add_argument("--json", type=Path, help="grava o relatório em JSON")
    args = ap.parse_args(argv)

    rep = check(args.outputs or outputs_dir(), args.patient)
    print(f"Cenário {rep['patient_id']} · saídas em {rep['outputs']}")
    print("\nFontes:")
    for s, info in rep["fontes"].items():
        types = ", ".join(f"{t} ×{n}" for t, n in info["types"].items()) or "-"
        print(f"  {s:<13} {info['origem']:<22} {info['findings']:>3} Finding(s)  {types}")
    print(f"\nAlertas ({len(rep['alertas'])} para {rep['findings_total']} Findings):")
    for a in rep["alertas"]:
        print(f"  {a['nivel']:<8} {a['horario'][11:19]}  {a['regra']:<38} score {a['score']:.2f}  "
              f"{a['findings']} Finding(s) +{a['deduplicados']} deduplicado(s)")
    for e in rep["erros"]:
        print(f"  ERRO: {e}")
    print("\nVerificações:")
    for name, ok in rep["verificacoes"].items():
        print(f"  [{'ok' if ok else 'FALHOU'}] {name}")
    if args.json:
        args.json.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
