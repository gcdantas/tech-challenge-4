"""Testes da leitura e validação do config/anomaly.yaml."""
import copy

import pytest
import yaml
from pydantic import ValidationError

from modules.anomaly.config import CAMINHO_PADRAO, SINAIS, carregar_config


def dados_do_repositorio() -> dict:
    return yaml.safe_load(CAMINHO_PADRAO.read_text(encoding="utf-8"))


def gravar(tmp_path, dados: dict):
    caminho = tmp_path / "anomaly.yaml"
    caminho.write_text(yaml.safe_dump(dados, allow_unicode=True), encoding="utf-8")
    return caminho


def test_config_do_repositorio_carrega_os_seis_sinais():
    cfg = carregar_config()
    assert set(cfg.vitais.sinais) == set(SINAIS)
    assert cfg.vitais.sinais["spo2"].baixo.critical == 85


def test_chave_desconhecida_e_rejeitada(tmp_path):
    dados = dados_do_repositorio()
    dados["vitais"]["sinais"]["fc"]["unidae"] = "bpm"  # erro de digitação
    with pytest.raises(ValidationError):
        carregar_config(gravar(tmp_path, dados))


def test_limites_de_baixo_fora_de_ordem_sao_rejeitados(tmp_path):
    dados = copy.deepcopy(dados_do_repositorio())
    dados["vitais"]["sinais"]["spo2"]["baixo"] = {"medium": 85, "high": 90, "critical": 94}
    with pytest.raises(ValidationError):
        carregar_config(gravar(tmp_path, dados))


def test_limite_fora_da_faixa_fisica_e_rejeitado(tmp_path):
    dados = dados_do_repositorio()
    dados["vitais"]["sinais"]["fc"]["alto"]["critical"] = 300
    with pytest.raises(ValidationError):
        carregar_config(gravar(tmp_path, dados))


def test_sinal_faltando_e_rejeitado(tmp_path):
    dados = dados_do_repositorio()
    del dados["vitais"]["sinais"]["temp"]
    with pytest.raises(ValidationError):
        carregar_config(gravar(tmp_path, dados))