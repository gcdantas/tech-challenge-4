"""Scripts da entrega do I4: verificação ponta a ponta do P001 e PDF do relatório."""

from __future__ import annotations

import shutil

from scripts.e2e_p001 import check
from scripts.relatorio_pdf import build_html, drop_sections, normalize_lists


def test_e2e_com_mocks_passa_e_marca_a_origem(mocks_dir):
    rep = check(mocks_dir)
    assert rep["ok"], rep["verificacoes"]
    assert {info["origem"] for info in rep["fontes"].values()} == {"mock"}
    assert any(a["nivel"] == "critico" for a in rep["alertas"])


def test_e2e_falha_sem_uma_fonte(mocks_dir, tmp_path):
    out = tmp_path / "out"
    shutil.copytree(mocks_dir, out)
    shutil.rmtree(out / "text")
    shutil.rmtree(out / "audio")
    rep = check(out)
    assert not rep["ok"]
    assert not rep["verificacoes"]["seis fontes presentes"]
    assert not rep["verificacoes"]["alerta crítico pela regra"]
    assert rep["fontes"]["text"]["origem"] == "ausente"


def test_listas_do_github_viram_listas_no_pdf():
    md = "Regras:\n- uma\n  - sub\n- duas\n\n```\n- código\n```\n> Nota:\n> - item\n"
    out = normalize_lists(md)
    assert "Regras:\n\n- uma\n    - sub\n- duas" in out
    assert "```\n- código\n```" in out
    assert "> Nota:\n>\n> - item" in out


def test_pdf_deixa_de_fora_o_bastidor_e_embute_as_figuras():
    md = "intro\n## Sumário\nx\n## 3. Status\ny\n## 6. Arquitetura\nz\n"
    kept = drop_sections(md)
    assert "Status" not in kept and "Sumário" not in kept and "## 6. Arquitetura" in kept

    html = build_html()
    assert "Status por integrante</h2>" not in html
    assert "Introdução e problema" in html and "Ética e LGPD" in html
    assert 'src="data:image/svg+xml;base64,' in html  # diagrama de arquitetura embutido
    assert 'src="docs/img/' not in html
