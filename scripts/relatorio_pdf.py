"""Gera o PDF consolidado do relatório técnico a partir do README (dono: I4).

    python -m scripts.relatorio_pdf                  # docs/relatorio_tecnico.pdf
    python -m scripts.relatorio_pdf --html-only      # só o HTML, para conferir

O README é o relatório: cada capítulo do plano é uma seção dele (seção 2).
O script converte o Markdown em HTML (com as figuras de docs/img/ embutidas) e
imprime em PDF com o Chrome ou o Edge em modo headless, que já vêm no Windows.
Seções de bastidor do grupo (status por integrante, estrutura do repositório)
ficam de fora do PDF; o README continua completo no Git.
"""

from __future__ import annotations

import argparse
import base64
import html
import mimetypes
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
OUT = ROOT / "docs" / "relatorio_tecnico.pdf"

# Seções (pelo número do título "## N.") que não entram no PDF.
SKIP_SECTIONS = {"3", "4"}

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "google-chrome",
    "chromium",
    "msedge",
]

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: "Segoe UI", Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1f2933; }
h1 { font-size: 20pt; margin: 0 0 6pt; }
h2 { font-size: 15pt; margin-top: 18pt; border-bottom: 1px solid #cbd2d9; padding-bottom: 3pt; page-break-after: avoid; }
h2.capitulo { page-break-before: always; }
h3 { font-size: 12.5pt; margin-top: 14pt; page-break-after: avoid; }
h4 { font-size: 11pt; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 9pt; page-break-inside: auto; }
tr { page-break-inside: avoid; }
th, td { border: 1px solid #cbd2d9; padding: 3pt 5pt; vertical-align: top; text-align: left; }
th { background: #f0f4f8; }
code { font-family: Consolas, "Courier New", monospace; font-size: 8.8pt; background: #f5f7fa; padding: 0 2pt; }
pre { background: #f5f7fa; padding: 6pt; overflow-x: hidden; white-space: pre-wrap; font-size: 8pt; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
blockquote { border-left: 3px solid #9aa5b1; margin: 6pt 0; padding: 2pt 10pt; color: #3e4c59; }
img { max-width: 100%; max-height: 120mm; }
td img { max-height: 45mm; }
.capa { text-align: center; padding-top: 70mm; page-break-after: always; }
.capa h1 { font-size: 24pt; }
.capa p { font-size: 12pt; color: #3e4c59; }
"""


def drop_sections(md: str) -> str:
    """Remove as seções de nível 2 listadas em SKIP_SECTIONS e o sumário."""
    parts = re.split(r"(?m)^(?=## )", md)
    kept = []
    for part in parts:
        m = re.match(r"## (\d+)\.", part)
        if (m and m.group(1) in SKIP_SECTIONS) or part.startswith("## Sumário"):
            continue
        kept.append(part)
    return "".join(kept)


_LIST = re.compile(r"^((?:> ?)*)( *)([-*+]|\d+\.) ")


def normalize_lists(md: str) -> str:
    """Ajusta o Markdown do GitHub ao Python-Markdown.

    - lista logo depois de um parágrafo ganha uma linha em branco antes;
    - sublista recuada com 2 espaços passa a 4 (o Python-Markdown exige 4).
    Blocos de código ficam intactos.
    """
    out: list[str] = []
    in_code = False
    prev_list = False
    for line in md.splitlines():
        if line.lstrip("> ").startswith("```"):
            in_code = not in_code
            out.append(line)
            prev_list = False
            continue
        if in_code:
            out.append(line)
            continue
        m = _LIST.match(line)
        if m:
            prefix, indent = m.group(1), m.group(2)
            prev = out[-1] if out else ""
            prev_text = prev[len(prefix):].strip() if prev.startswith(prefix.rstrip()) else prev.strip()
            if not indent and prev_text and not prev_list and not prev_text.startswith("|"):
                out.append(prefix.rstrip())
            line = prefix + " " * ((len(indent) // 2) * 4) + line[len(prefix) + len(indent):]
            prev_list = True
        elif line.strip() == "" or not line.startswith((" ", ">")):
            prev_list = False  # parágrafo ou linha em branco: a lista acabou
        out.append(line)
    return "\n".join(out) + "\n"


def drop_status_block(md: str) -> str:
    """Tira o quadro "Estado atual" (bastidor do grupo) do início."""
    return re.sub(r"(?ms)^> \*\*Estado atual.*?(?=^[^>])", "", md, count=1)


def embed_images(body: str) -> str:
    """Troca src relativos por data URI, para o PDF não depender de caminhos."""

    def repl(m: re.Match) -> str:
        src = html.unescape(m.group(2))
        path = (ROOT / src).resolve()
        if src.startswith(("http:", "https:", "data:")) or not path.is_file():
            return m.group(0)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = base64.b64encode(path.read_bytes()).decode()
        return f'{m.group(1)}data:{mime};base64,{data}"'

    return re.sub(r'(<img[^>]*?src=")([^"]+)"', repl, body)


def build_html(readme: Path = README) -> str:
    import markdown

    md = readme.read_text(encoding="utf-8")
    title_line, _, rest = md.partition("\n")
    title = title_line.lstrip("# ").strip()
    body = markdown.markdown(
        normalize_lists(drop_status_block(drop_sections(rest))), extensions=["tables", "fenced_code", "sane_lists", "toc"], output_format="html5"
    )
    body = re.sub(r"<h2 ", '<h2 class="capitulo" ', body)
    body = embed_images(body)
    capa = (
        f'<div class="capa"><h1>{html.escape(title)}</h1>'
        "<p>Relatório técnico consolidado · FIAP Pós Tech IADT</p>"
        f"<p>Gerado a partir do README do repositório em {date.today():%d/%m/%Y}</p>"
        "<p>Protótipo de demonstração técnica. O sistema sinaliza risco e apoia a equipe; a decisão é clínica.</p>"
        "</div>"
    )
    return (
        '<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
        f"<body>{capa}{body}</body></html>"
    )


def find_browser() -> str:
    for b in BROWSERS:
        if Path(b).is_file() or shutil.which(b):
            return b
    raise FileNotFoundError("Chrome ou Edge não encontrado; use --html-only e imprima em PDF pelo navegador")


def to_pdf(html_path: Path, pdf_path: Path, browser: str | None = None) -> Path:
    browser = browser or find_browser()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run(
            [
                browser, "--headless=new", "--disable-gpu", f"--user-data-dir={profile}",
                "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri(),
            ],
            check=True, capture_output=True, timeout=180,
        )
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise RuntimeError(f"o navegador não gerou {pdf_path}")
    return pdf_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT, help="PDF de saída")
    ap.add_argument("--html-only", action="store_true", help="grava só o HTML ao lado do PDF")
    args = ap.parse_args(argv)

    html_path = args.out.with_suffix(".html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(build_html(), encoding="utf-8")
    if args.html_only:
        print(html_path)
        return 0
    pdf = to_pdf(html_path, args.out)
    html_path.unlink()
    print(f"{pdf} ({pdf.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
