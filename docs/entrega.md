# Entrega da Fase 4: roteiro do vídeo e checklist (Integrante 4)

Prazo: **17/11/2026**. O enunciado pede, no repositório, o código e o relatório
técnico; e um vídeo de até 15 minutos no YouTube ou Vimeo (público ou não
listado) mostrando:
1. exemplo prático da análise de áudio e vídeo;
2. detecção e resposta a anomalias;
3. integração dos serviços Azure;
4. fluxo final do alerta à equipe médica.

## Roteiro do vídeo (até 15 min)

Linguagem: a da seção 12 do README. Sempre "paciente sintético P001", "sinaliza
risco e apoia a equipe; a decisão é clínica", "indicador experimental de
alteração vocal" e "tempo quase real em ambiente de demonstração".

| Tempo | Bloco | Quem grava | O que aparece na tela | Pontos do enunciado |
|---|---|---|---|---|
| 0:00–1:00 | Abertura: problema e arquitetura | I4 | título; diagrama `docs/img/arquitetura.svg`; uma frase sobre fusão tardia e o P001 | — |
| 1:00–3:30 | Vídeo: fisioterapia e leito | I1 | REHAB24-6 com esqueleto, amplitude incompleta (rep. 12) e postura (rep. 18); OpenPose; leito do Charades com a zona da borda; relatório HTML | 1 |
| 3:30–6:00 | Áudio e notas no Azure | I2 | consulta gravada; transcrição do Azure Speech com "falta de ar" destacado; sentimento do Azure AI Language; indicador vocal (experimental); nota recente comparada com a anterior | 1 e 3 |
| 6:00–8:30 | Anomalias | I3 | simulador em fluxo; dessaturação às 10:45 marcada na série; dose duplicada; saída do leito vinda do vídeo; métricas da avaliação | 2 |
| 8:30–12:00 | Integração e alerta à equipe | I4 | Streamlit: linha do tempo do P001, alerta urgente ponderado (10:20), **alerta crítico pela regra (10:45:50)** com as três evidências; aba de anomalias; upload de um arquivo chamando o módulo; **e-mail chegando na caixa** da equipe | 2, 3 e 4 |
| 12:00–13:30 | Resultados e limites | I4 | tabela da seção 10.2 com a ressalva de amostra; controle negativo (SpO2 fora da janela não gera crítico); limitações principais | — |
| 13:30–14:30 | Ética e encerramento | I4 | dados públicos, sintéticos ou da equipe; nada bruto no Git; decisão clínica; link do repositório | — |

### Preparação da demonstração do I4 (antes de gravar)

```powershell
git pull
python -m pip install -r requirements.txt
python -m data.download_weights; python -m data.download_rehab24; python -m data.download_charades
python -m modules.video.run --patient P001 --input data/video/rehab24-6/PM_105-Camera18-30fps-transposed.mp4 data/video/charades/SSP8C.mp4
python -m modules.anomaly.run --input outputs/video/P001/movement.parquet --patient P001
# Áudio e texto: saídas reais do I2 copiadas para outputs/audio/P001 e outputs/text/P001,
# ou, na falta delas, o mock marcado: python -m modules.audio.run --scenario --patient P001 --mock
python -m scripts.e2e_p001          # confere o cenário antes de gravar
streamlit run app/streamlit_app.py
```

- Apague `outputs/alerts_delivery.jsonl` antes de gravar, para o e-mail do
  alerta crítico sair de novo durante a gravação.
- Deixe a caixa de entrada da equipe aberta numa aba, já fora do spam.
- Para o upload ao vivo, tenha à mão um TXT de nota (ex.: `data/notes/P001_nota_2.txt`).

### Edição (I4)

- Juntar os blocos na ordem acima; cortar esperas de processamento.
- Legenda fixa no canto: "Paciente sintético · demonstração técnica".
- Trechos do Charades e do REHAB24-6 curtos, com a citação na tela (licenças, seção 13).
- Exportar em 1080p e conferir que a duração final é de no máximo 15:00.
- Subir no YouTube como **não listado** e testar o link numa janela anônima.

## Checklist da entrega

| Item | Responsável | Situação |
|---|---|---|
| PRs #3 e #4 do I4 revisados e mesclados na `main` | I4 + Gustavo | ⬜ |
| PR de relatório (seções 1, 6 e 10, PDF, este roteiro) mesclado | I4 | ⬜ |
| `python -m pytest` passando na `main` | todos | ⬜ |
| README testado num ambiente limpo (seção 5) | I1 (M5) | ⬜ |
| Repositório acessível aos avaliadores (público, ou avaliadores convidados) | dono do repositório | ⬜ |
| PDF único do relatório: `python -m scripts.relatorio_pdf` → `docs/relatorio_tecnico.pdf`, gerado da `main` final | I4 | ⬜ |
| Blocos do vídeo gravados (I1, I2, I3, I4) | todos | ⬜ |
| Vídeo editado, até 15 min, no YouTube/Vimeo não listado | I4 | ⬜ |
| Link do vídeo no README (seção 1) e no PDF | I4 | ⬜ |
| Submissão na plataforma da FIAP: link do repositório, PDF e link do vídeo | I4 | ⬜ |
