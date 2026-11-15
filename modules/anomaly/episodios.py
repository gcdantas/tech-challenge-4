"""Agrupa amostras anormais em episódios.

Sem isso, uma dessaturação de 3 minutos geraria 18 alertas, um por amostra. O
rastreador junta as amostras ativas num episódio só e decide quando ele vira
Finding:
  - só depois de uma duração mínima (filtra artefato curto de sensor);
  - uma vez, e de novo apenas se a severidade piorar;
  - o episódio só acaba depois de `recuperacao_s` seguidos sem anormalidade, para
    uma leitura normal no meio não partir o episódio em dois.
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Episodio:
    id: str
    origem: str  # de onde veio, ex.: o nome do sinal
    inicio: float  # tempo (s) da primeira amostra anormal
    ultimo_ativo: float  # tempo da última amostra anormal
    pior_rank: int = -1
    pior_score: float = 0.0
    tipo: str = ""
    clinico: bool = False  # True quando algum limite clínico foi cruzado
    detalhes: dict[str, Any] = field(default_factory=dict)  # evidência do pior ponto
    rank_emitido: int = -1  # severidade do último Finding emitido (-1 = nenhum)
    fim: float | None = None

    @property
    def emitido(self) -> bool:
        return self.rank_emitido >= 0

    @property
    def duracao_s(self) -> float:
        return (self.fim if self.fim is not None else self.ultimo_ativo) - self.inicio


class RastreadorEpisodio:
    def __init__(self, origem: str, duracao_min_clinico_s: float, duracao_min_estatistico_s: float, recuperacao_s: float):
        self.origem = origem
        self.duracao_min_clinico_s = duracao_min_clinico_s
        self.duracao_min_estatistico_s = duracao_min_estatistico_s
        self.recuperacao_s = recuperacao_s
        self.atual: Episodio | None = None
        self.encerrados: list[Episodio] = []  # só os que chegaram a gerar Finding
        self._inativo_desde: float | None = None
        self._contador = 0

    def atualizar(
        self,
        t: float,
        ativo: bool,
        *,
        rank: int = 0,
        score: float = 0.0,
        tipo: str = "",
        clinico: bool = False,
        detalhes: dict[str, Any] | None = None,
        pode_emitir: bool = True,
    ) -> list[str]:
        """Processa a amostra de tempo `t`. Devolve "emitir", "escalar" e/ou "encerrar".

        `pode_emitir=False` segura o Finding sem perder o episódio. Quem usa o
        rastreador faz isso para suprimir desvios correlatos, e na próxima amostra
        a decisão é tomada de novo.
        """
        eventos: list[str] = []
        ep = self.atual

        if ativo:
            self._inativo_desde = None
            if ep is None:
                self._contador += 1
                ep = self.atual = Episodio(f"{self.origem}-{self._contador}", self.origem, t, t)
            ep.ultimo_ativo = t

            if rank > ep.pior_rank or (rank == ep.pior_rank and score > ep.pior_score):
                ep.pior_rank, ep.pior_score = rank, score
                ep.detalhes = dict(detalhes or {})
            # o tipo clínico (ex.: dessaturacao) vale mais que o genérico (desvio_spo2)
            if clinico:
                ep.clinico, ep.tipo = True, tipo
            elif not ep.clinico:
                ep.tipo = tipo

            minimo = self.duracao_min_clinico_s if ep.clinico else self.duracao_min_estatistico_s
            if pode_emitir and (t - ep.inicio) >= minimo:
                if not ep.emitido:
                    eventos.append("emitir")
                    ep.rank_emitido = ep.pior_rank
                elif ep.pior_rank > ep.rank_emitido:
                    eventos.append("escalar")
                    ep.rank_emitido = ep.pior_rank
            return eventos

        if ep is None:
            return eventos
        if self._inativo_desde is None:
            self._inativo_desde = t
        if t - self._inativo_desde >= self.recuperacao_s:
            eventos.append("encerrar")
            self.encerrar()
        return eventos

    def encerrar(self) -> Episodio | None:
        """Fecha o episódio aberto. Se nunca gerou Finding, era artefato e é descartado."""
        ep = self.atual
        if ep is None:
            return None
        self.atual = None
        self._inativo_desde = None
        ep.fim = ep.ultimo_ativo
        if ep.emitido:
            self.encerrados.append(ep)
        return ep

    def todos_emitidos(self) -> list[Episodio]:
        abertos = [self.atual] if self.atual is not None and self.atual.emitido else []
        return [*self.encerrados, *abertos]