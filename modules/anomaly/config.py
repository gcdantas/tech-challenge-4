"""Leitura e validação da configuração de anomalias (config/anomaly.yaml)."""
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts.patients import REPO_ROOT

CAMINHO_PADRAO = REPO_ROOT / "config/anomaly.yaml"
CAMINHO_SIMULADOR = REPO_ROOT / "config/simulator.yaml"

# Os seis sinais que o detector de vitais acompanha
SINAIS = ("fc", "spo2", "fr", "pas", "pad", "temp")

# O que o simulador consegue injetar nos vitais. As anomalias o sistema DEVE
# sinalizar; os distratores parecem anomalia mas NÃO devem gerar alerta
TIPOS_ANOMALIA = (
    "dessaturacao",
    "taquicardia",
    "hipotensao",
    "febre",
    "taquipneia",
    "deterioracao_multivariada",
)
TIPOS_DISTRATOR = ("artefato_pico", "perda_sinal", "leitura_invalida")

# O mesmo vale para a movimentação do leito
TIPOS_MOVIMENTO_ANOMALIA = ("saida_do_leito", "imobilidade")
TIPOS_MOVIMENTO_DISTRATOR = ("falso_quadro", "perda_deteccao", "virar_no_leito")


class Estrito(BaseModel):
    # extra="forbid": se eu errar o nome de uma chave no yaml, dá erro na hora
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Limites(Estrito):
    medium: float
    high: float
    critical: float


class SinalCfg(Estrito):
    nome: str
    unidade: str
    faixa_fisica: tuple[float, float]
    baixo: Limites | None = None
    alto: Limites | None = None

    @model_validator(mode="after")
    def validar(self):
        minimo, maximo = self.faixa_fisica
        if minimo >= maximo:
            raise ValueError("faixa_fisica precisa ter o mínimo menor que o máximo")
        if self.baixo is None and self.alto is None:
            raise ValueError("defina pelo menos um lado: baixo ou alto")

        # No lado de baixo o limite fica mais severo conforme o valor desce
        if self.baixo and not (self.baixo.medium > self.baixo.high > self.baixo.critical):
            raise ValueError("limites de baixo devem ser decrescentes (medium > high > critical)")
        # No lado de cima é o contrário
        if self.alto and not (self.alto.medium < self.alto.high < self.alto.critical):
            raise ValueError("limites de alto devem ser crescentes (medium < high < critical)")
        if self.baixo and self.alto and self.baixo.medium >= self.alto.medium:
            raise ValueError("o limite de baixo precisa ficar abaixo do limite de alto")

        # Limite fora da faixa física nunca seria alcançado por uma leitura válida
        for lado in (self.baixo, self.alto):
            if lado and not all(minimo <= v <= maximo for v in (lado.medium, lado.high, lado.critical)):
                raise ValueError("limites precisam estar dentro da faixa_fisica")
        return self


class ZScoreCfg(Estrito):
    janela_s: float = Field(gt=0)
    min_pontos: int = Field(ge=10)
    limiar: float = Field(gt=0)  # |z| a partir daqui é desvio "low"
    limiar_medium: float = Field(gt=0)
    limiar_high: float = Field(gt=0)
    escala_min: dict[str, float]

    @model_validator(mode="after")
    def conferir_limiares(self):
        if not (self.limiar < self.limiar_medium < self.limiar_high):
            raise ValueError("os limiares do z-score precisam ser crescentes: limiar < limiar_medium < limiar_high")
        return self

    @field_validator("escala_min")
    @classmethod
    def conferir_escala(cls, valores: dict[str, float]):
        if set(valores) != set(SINAIS):
            raise ValueError(f"escala_min precisa ter exatamente estes sinais: {', '.join(SINAIS)}")
        if any(v <= 0 for v in valores.values()):
            raise ValueError("escala_min precisa ser maior que zero em todos os sinais")
        return valores


class VitaisCfg(Estrito):
    duracao_min_s: float = Field(gt=0)  # tempo mínimo para confirmar um episódio com limite clínico
    duracao_min_estatistico_s: float = Field(gt=0)  # idem, para desvio só estatístico (mais exigente)
    recuperacao_s: float = Field(gt=0)  # tempo normal seguido para encerrar o episódio
    lacuna_max_s: float = Field(gt=0)  # sem leitura por mais que isso encerra o episódio do sinal
    suprimir_correlatos: bool
    sinais: dict[str, SinalCfg]
    zscore: ZScoreCfg

    @model_validator(mode="after")
    def validar(self):
        if set(self.sinais) != set(SINAIS):
            raise ValueError(f"os sinais precisam ser exatamente estes: {', '.join(SINAIS)}")
        if self.duracao_min_estatistico_s < self.duracao_min_s:
            raise ValueError("duracao_min_estatistico_s não pode ser menor que duracao_min_s")
        return self


class IsolationForestCfg(Estrito):
    ativo: bool
    semente: int
    min_treino: int = Field(ge=30)  # vetores normais necessários para o primeiro treino
    max_treino: int = Field(ge=30)  # só os mais recentes entram no treino
    refit_s: float = Field(gt=0)  # de quanto em quanto tempo o modelo é treinado de novo
    n_estimators: int = Field(ge=10)
    quantil_limiar: float = Field(gt=0.5, lt=1)
    margem_limiar: float = Field(ge=1.0)
    a_saturacao: float = Field(gt=0, lt=1)  # escore em que o excesso normalizado chega a 1
    duracao_min_s: float = Field(gt=0)
    rebase_s: float = Field(gt=0)  # episódio mais longo que isso vira o novo normal

    @model_validator(mode="after")
    def validar(self):
        if self.max_treino < self.min_treino:
            raise ValueError("max_treino não pode ser menor que min_treino")
        return self


class MedicamentoCfg(Estrito):
    nome: str
    dose_min_mg: float = Field(gt=0)  # por tomada
    dose_max_mg: float = Field(gt=0)  # por tomada
    dose_diaria_max_mg: float = Field(gt=0)
    continuo: bool  # uso contínuo: parar de repente é um risco

    @model_validator(mode="after")
    def validar(self):
        if self.dose_min_mg > self.dose_max_mg:
            raise ValueError("dose_min_mg não pode ser maior que dose_max_mg")
        if self.dose_diaria_max_mg < self.dose_max_mg:
            raise ValueError("dose_diaria_max_mg não pode ser menor que dose_max_mg")
        return self


class AumentoCfg(Estrito):
    # aumento percentual da dose diária em relação ao pedido anterior do mesmo medicamento
    medium: float = Field(gt=0)
    high: float = Field(gt=0)
    critical: float = Field(gt=0)
    janela_comparacao_h: float = Field(gt=0)  # só compara com pedido encerrado até X horas antes

    @model_validator(mode="after")
    def validar(self):
        if not (self.medium < self.high < self.critical):
            raise ValueError("os limites de aumento precisam ser crescentes: medium < high < critical")
        return self


class SuspensaoCfg(Estrito):
    min_uso_h: float = Field(ge=0)  # só vale para medicamento em uso há pelo menos isso
    janela_substituicao_h: float = Field(gt=0)  # sem novo pedido nesse tempo = suspensão abrupta


class PrescricoesCfg(Estrito):
    aumento: AumentoCfg
    suspensao: SuspensaoCfg
    medicamentos: dict[str, MedicamentoCfg]

    @field_validator("medicamentos")
    @classmethod
    def conferir_chaves(cls, valores: dict[str, MedicamentoCfg]):
        # a chave é o nome em minúsculas e sem acento, que é como o detector procura
        for chave in valores:
            if chave != chave.lower() or not chave.isascii():
                raise ValueError(f"chave de medicamento precisa ser minúscula e sem acento: {chave!r}")
        return valores


class SaidaLeitoCfg(Estrito):
    confirmar_fora_s: float = Field(gt=0)  # fora do leito por tanto tempo seguido para valer
    retorno_s: float = Field(gt=0)  # de volta ao leito por tanto tempo seguido para encerrar
    no_leito_min_s: float = Field(ge=0)  # só é "saída" se a pessoa estava no leito antes
    ausencia_high_s: float = Field(gt=0)  # fora do leito por mais que isso sobe para high
    score: float = Field(ge=0, le=1)
    score_high: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validar(self):
        if self.score_high < self.score:
            raise ValueError("score_high não pode ser menor que score")
        return self


class ImobilidadeCfg(Estrito):
    limiar_s: float = Field(gt=0)  # parado por tanto tempo vira Finding (low)
    fator_medium: float = Field(gt=1)  # em múltiplos do limiar
    fator_high: float = Field(gt=1)
    bin_s: float = Field(gt=0)  # a pose é resumida em blocos desse tamanho
    min_amostras_bin: int = Field(ge=1)  # amostras com pose para o bloco valer
    tol_angulo: float = Field(gt=0)  # graus
    tol_centro: float = Field(gt=0)  # unidades normalizadas (0 a 1)
    lacuna_max_s: float = Field(gt=0)  # sem pose por mais que isso zera a contagem

    @model_validator(mode="after")
    def validar(self):
        if not (1 < self.fator_medium < self.fator_high):
            raise ValueError("os fatores precisam ser crescentes: 1 < fator_medium < fator_high")
        return self


class MovimentoCfg(Estrito):
    saida_leito: SaidaLeitoCfg
    imobilidade: ImobilidadeCfg


class AnomaliaCfg(Estrito):
    vitais: VitaisCfg
    isolation_forest: IsolationForestCfg
    prescricoes: PrescricoesCfg
    movimento: MovimentoCfg


def carregar_config(caminho: str | Path | None = None) -> AnomaliaCfg:
    caminho = Path(caminho) if caminho else CAMINHO_PADRAO
    return AnomaliaCfg.model_validate(yaml.safe_load(caminho.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Simulador (config/simulator.yaml)
# ---------------------------------------------------------------------------


class VitaisSimCfg(Estrito):
    periodo_s: float = Field(gt=0)  # intervalo entre amostras
    duracao_min: float = Field(gt=0)
    phi: float = Field(ge=0, lt=1)  # autocorrelação do ruído (AR(1))
    arredondar: bool
    resposta_fisiologica: bool  # na dessaturação a FC e a FR sobem um pouco
    basal: dict[str, float]
    ruido: dict[str, float]
    deriva: dict[str, float]
    variacao_entre_pacientes: dict[str, float]

    @field_validator("basal", "ruido", "deriva", "variacao_entre_pacientes")
    @classmethod
    def conferir_sinais(cls, valores: dict[str, float]):
        if set(valores) != set(SINAIS):
            raise ValueError(f"precisa ter exatamente estes sinais: {', '.join(SINAIS)}")
        return valores


class InjecaoCfg(Estrito):
    tipo: str
    inicio_min: float = Field(ge=0)  # minutos desde o início da série
    duracao_min: float = Field(gt=0)
    magnitude: float = Field(default=0, ge=0)
    rampa_s: float = Field(default=0, ge=0)  # tempo para subir até a magnitude (e para voltar)

    @model_validator(mode="after")
    def validar(self):
        if self.tipo not in TIPOS_ANOMALIA + TIPOS_DISTRATOR:
            raise ValueError(f"tipo de injeção desconhecido: {self.tipo!r}")
        # perda de sinal e leitura inválida não usam magnitude
        if self.tipo not in ("perda_sinal", "leitura_invalida") and self.magnitude <= 0:
            raise ValueError(f"{self.tipo} precisa de magnitude maior que zero")
        return self


class CenarioVitaisCfg(Estrito):
    injecoes: list[InjecaoCfg] = Field(default_factory=list)


class PedidoCfg(Estrito):
    medicamento: str
    dose_mg: float = Field(gt=0)  # por tomada
    doses_por_dia: float = Field(gt=0)


class EventoCenarioCfg(Estrito):
    acao: Literal["prescrever", "suspender"]
    minuto: float  # minutos a partir do início da demonstração (pode ser negativo)
    medicamento: str
    dose_mg: float | None = Field(default=None, gt=0)
    doses_por_dia: float | None = Field(default=None, gt=0)
    anomalia: bool = False  # True = o sistema deve sinalizar este evento
    tipo_esperado: str | None = None  # qual achado se espera, quando anomalia

    @model_validator(mode="after")
    def validar(self):
        if self.acao == "prescrever" and (self.dose_mg is None or self.doses_por_dia is None):
            raise ValueError("prescrever precisa de dose_mg e doses_por_dia")
        if self.anomalia and not self.tipo_esperado:
            raise ValueError("evento com anomalia precisa de tipo_esperado")
        return self


class CenarioPrescricoesCfg(Estrito):
    pedidos_iniciais: list[PedidoCfg] = Field(default_factory=list)
    eventos: list[EventoCenarioCfg] = Field(default_factory=list)


class PrescricoesSimCfg(Estrito):
    historico_h: float = Field(gt=0)  # os pedidos de admissão começam tantas horas antes


class PoseCfg(Estrito):
    centroid_x: float = Field(ge=0, le=1)
    centroid_y: float = Field(ge=0, le=1)
    torso_angle: float = Field(ge=0, le=180)


class MovimentoSimCfg(Estrito):
    hz: float = Field(gt=0)  # amostras por segundo (o movement.parquet real do I1 tem 5)
    duracao_min: float = Field(gt=0)
    phi: float = Field(ge=0, lt=1)  # autocorrelação do ruído
    deitado: PoseCfg  # pose do paciente deitado no leito
    fora: PoseCfg  # pose de quem saiu do leito
    ruido_centro: float = Field(ge=0)
    ruido_angulo: float = Field(ge=0)
    reposicionamento_min: float = Field(gt=0)  # a cada quanto tempo o paciente muda de posição (em média)
    deslocamento_angulo: float = Field(gt=0)  # quanto a pose muda ao virar
    deslocamento_centro: float = Field(gt=0)
    rampa_s: float = Field(gt=0)  # tempo que a transição de pose leva
    clipe_curto_min: float = Field(gt=0)  # clipe real mais curto que isso é estendido com série simulada


class EventoMovimentoCfg(Estrito):
    tipo: str
    inicio_min: float = Field(ge=0)
    duracao_s: float = Field(default=0, ge=0)
    oscila: bool = False  # só saída do leito: a pessoa se debruça no leito e o in_bed pisca

    @model_validator(mode="after")
    def validar(self):
        if self.tipo not in TIPOS_MOVIMENTO_ANOMALIA + TIPOS_MOVIMENTO_DISTRATOR:
            raise ValueError(f"tipo de evento de movimento desconhecido: {self.tipo!r}")
        if self.tipo != "virar_no_leito" and self.duracao_s <= 0:
            raise ValueError(f"{self.tipo} precisa de duracao_s maior que zero")
        if self.oscila and self.tipo != "saida_do_leito":
            raise ValueError("oscila só vale para saida_do_leito")
        return self


class CenarioMovimentoCfg(Estrito):
    duracao_min: float | None = Field(default=None, gt=0)  # sem isso vale a duração padrão
    eventos: list[EventoMovimentoCfg] = Field(default_factory=list)


class CenarioCfg(Estrito):
    vitais: CenarioVitaisCfg = Field(default_factory=CenarioVitaisCfg)
    prescricoes: CenarioPrescricoesCfg = Field(default_factory=CenarioPrescricoesCfg)
    movimento: CenarioMovimentoCfg = Field(default_factory=CenarioMovimentoCfg)


class FluxoCfg(Estrito):
    velocidade: float = Field(ge=0)  # segundos da série por segundo real; 0 = sem pausas


class SimuladorCfg(Estrito):
    semente: int
    vitais: VitaisSimCfg
    prescricoes: PrescricoesSimCfg
    movimento: MovimentoSimCfg
    fluxo: FluxoCfg
    pacientes: dict[str, str] = Field(default_factory=dict)  # paciente -> cenário que ele usa
    cenarios: dict[str, CenarioCfg] = Field(default_factory=dict)

    @model_validator(mode="after")
    def conferir_pacientes(self):
        for paciente, cenario in self.pacientes.items():
            if cenario not in self.cenarios:
                raise ValueError(f"o paciente {paciente} usa o cenário {cenario!r}, que não existe em cenarios")
        return self


def carregar_config_simulador(caminho: str | Path | None = None) -> SimuladorCfg:
    caminho = Path(caminho) if caminho else CAMINHO_SIMULADOR
    return SimuladorCfg.model_validate(yaml.safe_load(caminho.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Avaliação (config/avaliacao.yaml)
# ---------------------------------------------------------------------------

CAMINHO_AVALIACAO = REPO_ROOT / "config/avaliacao.yaml"
Intervalo = tuple[float, float]


def _intervalo_valido(valor: tuple[float, float]) -> tuple[float, float]:
    if valor[0] > valor[1]:
        raise ValueError(f"intervalo com o mínimo maior que o máximo: {valor}")
    return valor


class TipoVitalAvalCfg(Estrito):
    magnitude: Intervalo
    duracao_min: Intervalo
    rampa_s: float = Field(ge=0)

    _conferir = field_validator("magnitude", "duracao_min")(_intervalo_valido)


class VarreduraCfg(Estrito):
    pacientes_por_ponto: int = Field(ge=1)
    duracao_min: float = Field(gt=0)
    evento_inicio_min: float = Field(ge=0)
    magnitudes: dict[str, list[float]]  # tipo -> magnitudes testadas


class VitaisAvalCfg(Estrito):
    duracao_min: float = Field(gt=0)
    pacientes_com_eventos: int = Field(ge=1)
    pacientes_estaveis: int = Field(ge=1)
    eventos_por_paciente: tuple[int, int]
    distratores_por_paciente: tuple[int, int]
    inicio_min: float = Field(ge=0)  # nenhum evento antes disso (a linha de base precisa se formar)
    folga_min: float = Field(ge=0)  # tempo mínimo entre eventos
    tipos: dict[str, TipoVitalAvalCfg]
    distratores: dict[str, TipoVitalAvalCfg]
    varredura: VarreduraCfg

    @model_validator(mode="after")
    def validar(self):
        for nome in self.tipos:
            if nome not in TIPOS_ANOMALIA:
                raise ValueError(f"tipo de anomalia desconhecido em tipos: {nome!r}")
        for nome in self.distratores:
            if nome not in TIPOS_DISTRATOR:
                raise ValueError(f"tipo de distrator desconhecido em distratores: {nome!r}")
        for nome in self.varredura.magnitudes:
            if nome not in self.tipos:
                raise ValueError(f"a varredura usa o tipo {nome!r}, que não está em tipos")
        return self


class PrescricoesAvalCfg(Estrito):
    duracao_min: float = Field(gt=0)
    pacientes: int = Field(ge=1)
    eventos_por_paciente: tuple[int, int]


class MovimentoAvalCfg(Estrito):
    pacientes_saida: int = Field(ge=1)
    pacientes_imobilidade: int = Field(ge=1)
    pacientes_estaveis: int = Field(ge=1)
    duracao_saida_min: float = Field(gt=0)
    duracao_imobilidade_min: float = Field(gt=0)
    saidas_por_paciente: tuple[int, int]
    saida_duracao_s: Intervalo
    imobilidade_duracao_min: Intervalo
    distratores_por_paciente: tuple[int, int]
    folga_min: float = Field(ge=0)

    _conferir = field_validator("saida_duracao_s", "imobilidade_duracao_min")(_intervalo_valido)


class AvaliacaoCfg(Estrito):
    semente: int
    tolerancia_s: dict[str, float]  # folga depois do fim do rótulo, por fonte
    vitais: VitaisAvalCfg
    prescricoes: PrescricoesAvalCfg
    movimento: MovimentoAvalCfg

    @field_validator("tolerancia_s")
    @classmethod
    def conferir_fontes(cls, valores: dict[str, float]):
        if set(valores) != {"vitals", "prescription", "movement"}:
            raise ValueError("tolerancia_s precisa ter vitals, prescription e movement")
        return valores


def carregar_config_avaliacao(caminho: str | Path | None = None) -> AvaliacaoCfg:
    caminho = Path(caminho) if caminho else CAMINHO_AVALIACAO
    return AvaliacaoCfg.model_validate(yaml.safe_load(caminho.read_text(encoding="utf-8")))
