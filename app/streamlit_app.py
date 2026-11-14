"""Dashboard Streamlit (dono: I4).

Carrega data/patients.yaml, lê todos os findings.jsonl em outputs/ (inclusive os
envios pela própria tela), roda a fusão e grava outputs/alerts.jsonl. O alerta
crítico também vai por e-mail (app/notify.py); sem SMTP no .env, vira .eml.

Abas por modalidade: vídeo (quadros, clipes e relatório do I1), áudio e notas
(transcrição e notas anotadas do I2), anomalias (séries do I3) e envio de
arquivos, que chama a análise do módulo dono (app/ingest.py).

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # `streamlit run` coloca só app/ no sys.path
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from app.dashboard import highlight_terms, md_escape, note_changes, series_layout  # noqa: E402
from app.fusion import fuse, load_fusion_config, write_alerts  # noqa: E402
from app.ingest import (  # noqa: E402
    KIND_LABEL,
    UPLOAD_KINDS,
    analyze_upload,
    list_uploads,
    remove_upload,
    uploadable_artifacts,
)
from app.loaders import (  # noqa: E402
    findings_dataframe,
    load_anomaly_series,
    load_findings,
    load_movement,
    load_notes,
    load_transcripts,
    outputs_dir,
    patients_path,
    video_frame_dirs,
    video_media,
    video_reports,
)
from app.notify import SmtpSettings, load_alerts_config, notify, read_deliveries  # noqa: E402
from contracts.patients import load_patients  # noqa: E402

load_dotenv(ROOT / ".env", override=False)

LEVEL_LABEL = {"critico": "Crítico", "urgente": "Urgente", "atencao": "Atenção"}

st.set_page_config(page_title="Monitoramento multimodal", layout="wide")
st.title("Monitoramento multimodal de pacientes")
st.caption(
    "Protótipo de demonstração técnica. O sistema sinaliza risco e apoia a equipe; a decisão é clínica."
)

out_dir = outputs_dir()
pf = load_patients(patients_path())

# --- Fusão de todos os pacientes (grava alerts.jsonl) -------------------------
all_loaded = load_findings(out_dir)
alerts = fuse(all_loaded.findings, load_fusion_config())
# E-mail do alerta crítico: um envio por alerta, mesmo com a tela recarregando.
alerts_cfg = load_alerts_config()
alerts = notify(alerts, all_loaded.findings, out_dir, cfg=alerts_cfg, settings=SmtpSettings.from_env())
deliveries = read_deliveries(out_dir)
alerts_path = write_alerts(out_dir / "alerts.jsonl", alerts)

# --- Barra lateral ------------------------------------------------------------
with st.sidebar:
    patient_id = st.selectbox("Paciente", sorted(pf.patients))
    patient = pf.patient(patient_id)
    st.write(patient.description)
    st.markdown(f"**Referência da demo:** `{pf.demo_reference.isoformat()}`")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "artefato": name,
                    "tipo": a.kind,
                    "demo_offset (s)": a.demo_offset,
                    "início na demo": pf.demo_time(a.demo_offset).strftime("%d/%m %H:%M:%S"),
                    "arquivo": a.path or "a definir",
                }
                for name, a in patient.artifacts.items()
            ]
        ),
        hide_index=True,
    )

loaded = load_findings(out_dir, patient_id)
for err in loaded.errors:
    st.error(f"Arquivo de Findings inválido: {err}")

df = findings_dataframe(loaded.findings)
if not df.empty and df["mock"].any():
    st.warning(
        "Há Findings FICTÍCIOS (mocks do M1) na tela. Gere-os de novo com "
        "`python -m scripts.make_mocks`."
    )

# --- Alertas ------------------------------------------------------------------
st.subheader("Alertas")
patient_alerts = [a for a in alerts if a.patient_id == patient_id]
if not patient_alerts:
    st.info("Nenhum alerta para este paciente.")
_LEVEL_RANK = {"critico": 0, "urgente": 1, "atencao": 2}
for a in sorted(patient_alerts, key=lambda a: (_LEVEL_RANK[a.level], -(a.score or 0.0), a.created_at)):
    msg = (
        f"**{LEVEL_LABEL[a.level]}** · {a.created_at.strftime('%d/%m %H:%M:%S')} · `{a.rule}`"
        + (f" · score {a.score:.2f}" if a.score is not None else "")
        + f"  \n{a.description}  \nFindings: {len(a.finding_ids)}"
    )
    if a.suppressed_finding_ids:
        msg += f" (+{len(a.suppressed_finding_ids)} do mesmo tipo dentro do cooldown, sem novo alerta)"
    d = deliveries.get(a.alert_id)
    if d is not None and d.sent_at is not None:
        msg += (
            f"  \nE-mail enviado à equipe às {d.sent_at.strftime('%H:%M:%S')} "
            f"({len(d.recipients)} destinatário(s))."
        )
    elif d is not None:
        msg += f"  \nE-mail não enviado ({d.error}). Cópia de contingência: `{d.eml_path}`."
    {"critico": st.error, "urgente": st.warning, "atencao": st.info}[a.level](msg)
st.caption(
    f"Alertas gravados em `{alerts_path}`. Níveis que geram e-mail: "
    f"{', '.join(alerts_cfg.email.levels) if alerts_cfg.email.enabled else 'nenhum (desligado)'}."
)

# --- Linha do tempo -----------------------------------------------------------
st.subheader("Linha do tempo multimodal")
if df.empty:
    st.info("Nenhum Finding encontrado em outputs/. Rode `python -m scripts.make_mocks`.")
else:
    tl = df.copy()
    # Eventos pontuais ganham 30 s de largura só para aparecer no gráfico.
    tl["fim_plot"] = tl[["inicio", "fim"]].max(axis=1)
    tl.loc[tl["fim_plot"] == tl["inicio"], "fim_plot"] = tl["inicio"] + pd.Timedelta(seconds=30)
    # Altair/Vega trabalham em UTC; o eixo mostra o horário local do navegador.
    tl["inicio"] = pd.to_datetime(tl["inicio"], utc=True)
    tl["fim_plot"] = pd.to_datetime(tl["fim_plot"], utc=True)
    chart = (
        alt.Chart(tl)
        .mark_bar(size=14, cornerRadius=3)
        .encode(
            x=alt.X("inicio:T", title="Horário"),
            x2="fim_plot:T",
            y=alt.Y("origem:N", title="Origem", sort=["video", "movement", "prescription", "audio", "text", "vitals"]),
            color=alt.Color(
                "severidade:N",
                scale=alt.Scale(
                    domain=["low", "medium", "high", "critical"],
                    range=["#7fb3d5", "#f5b041", "#e67e22", "#c0392b"],
                ),
            ),
            tooltip=["origem", "tipo", "severidade", "score", "descricao"],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, width="stretch")
    st.dataframe(
        df.drop(columns=["finding_id"]).assign(
            inicio=df["inicio"].map(lambda t: t.isoformat()), fim=df["fim"].map(lambda t: t.isoformat())
        ),
        hide_index=True,
    )

# --- Detalhes por modalidade -------------------------------------------------
tab_video, tab_audio, tab_anom, tab_upload = st.tabs(
    ["Vídeo (Integrante 1)", "Áudio e notas (Integrante 2)", "Anomalias (Integrante 3)", "Enviar arquivo"]
)

with tab_video:
    media = [m for m in video_media(loaded.findings, out_dir) if m["frames"] or m["clips"]]
    st.markdown("**Desvios detectados, com quadro e clipe anotados**")
    if not media:
        st.info("Nenhum quadro ou clipe de desvio encontrado. Rode o módulo de vídeo (seção 7.3).")
    for m in media:
        f = m["finding"]
        with st.expander(
            f"{f.timestamp_start.strftime('%H:%M:%S')} · {f.type} · severidade {f.severity} · score {f.score:.2f}"
        ):
            st.write(f.description)
            cols = st.columns(2)
            for img in m["frames"][:1]:
                cols[0].image(str(img), caption=img.name)
            for clip in m["clips"][:1]:
                cols[1].video(str(clip))
    frame_dirs = video_frame_dirs(out_dir, patient_id)
    if frame_dirs:
        st.markdown("**Quadros anotados por vídeo** (inclui o leito, que não gera Findings)")
        video_name = st.selectbox("Vídeo", list(frame_dirs), key="frames_video")
        imgs = frame_dirs[video_name]
        st.image([str(x) for x in imgs[:12]], caption=[x.name for x in imgs[:12]], width=220)
        if len(imgs) > 12:
            st.caption(f"Mostrando 12 de {len(imgs)} quadros.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Série de movimentação (Integrante 1 → Integrante 3)**")
        mov = load_movement(out_dir, patient_id)
        if mov is None:
            st.info("movement.parquet não encontrado.")
        else:
            mov = mov.assign(timestamp=pd.to_datetime(mov["timestamp"], utc=True))
            st.line_chart(mov.set_index("timestamp")[["centroid_x", "centroid_y"]])
            st.area_chart(mov.set_index("timestamp")["in_bed"].astype(int), height=120)
    with col2:
        st.markdown("**Relatório automático do vídeo**")
        reports = video_reports(out_dir, patient_id)
        if not reports:
            st.info("report.html não encontrado.")
        for r in reports:
            st.download_button(
                f"Baixar {r.parent.name}/{r.name}" + (" (envio)" if "uploads" in r.parts else ""),
                r.read_bytes(),
                file_name=f"relatorio_video_{patient_id}.html",
                mime="text/html",
                key=f"report_{r}",
            )

with tab_audio:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Transcrição da consulta**")
        transcripts = load_transcripts(out_dir, patient_id)
        if not transcripts:
            st.info("transcript.json não encontrado.")
        for path, tr in transcripts:
            label = "Envio pela tela" if "uploads" in path.parts else "Consulta do paciente"
            with st.expander(label + (" (mock)" if tr.get("mock") else ""), expanded=len(transcripts) == 1):
                for seg in tr.get("segments", []):
                    text = highlight_terms(seg["text"], seg.get("critical_terms", []))
                    st.markdown(f"`{seg['timestamp_start'][11:19]}` {text}")
                vocal = tr.get("vocal") or {}
                if vocal.get("status"):
                    st.caption(f"Indicador experimental de alteração vocal: {vocal['status']}.")
    with col2:
        st.markdown("**Notas de evolução anotadas**")
        notes = load_notes(out_dir, patient_id)
        if not notes:
            st.info("notes.json não encontrado. Rode o módulo de áudio e texto (seção 8).")
        for note in notes:
            when = note.get("timestamp_start", "")[:19].replace("T", " ")
            title = f"{when} · {note.get('note', 'nota')}" + (" (sintética)" if note.get("synthetic") else "")
            with st.expander(title, expanded=True):
                st.markdown(highlight_terms(note.get("text", ""), note.get("critical_terms", [])))
                st.caption(note_changes(note))

with tab_anom:
    series = load_anomaly_series(out_dir, patient_id)
    if series is None:
        st.info(
            "series_flagged.parquet ainda não existe: o simulador e os detectores do Integrante 3 "
            "não foram entregues. Os Findings de vitais, prescrições e movimentação já aparecem "
            "na linha do tempo."
        )
    else:
        time_col, numeric, flags = series_layout(series)
        if time_col is None or not numeric:
            st.warning("Formato do series_flagged.parquet não reconhecido; mostrando a tabela.")
            st.dataframe(series, hide_index=True)
        else:
            ser = series.assign(**{time_col: pd.to_datetime(series[time_col], utc=True)})
            metric = st.selectbox("Série", numeric, key="anom_metric")
            base = alt.Chart(ser).encode(x=alt.X(f"{time_col}:T", title="Horário"))
            layers = [base.mark_line().encode(y=alt.Y(f"{metric}:Q", title=metric))]
            if flags:
                marked = ser[ser[flags].any(axis=1)]
                layers.append(
                    alt.Chart(marked)
                    .mark_point(color="#c0392b", size=60, filled=True)
                    .encode(x=f"{time_col}:T", y=f"{metric}:Q", tooltip=[time_col, metric, *flags])
                )
                st.caption(f"{len(marked)} ponto(s) marcado(s) como anomalia ({', '.join(flags)}).")
            st.altair_chart(alt.layer(*layers).properties(height=280), width="stretch")

with tab_upload:
    st.markdown(
        "Envie um arquivo e escolha a qual artefato do paciente ele corresponde: o "
        "`demo_offset` desse artefato posiciona os Findings na linha do tempo. A análise é "
        "feita pelo módulo dono (vídeo: Integrante 1; áudio e nota: Integrante 2) e o "
        "resultado entra na fusão."
    )
    options = uploadable_artifacts(pf, patient_id)
    if not options:
        st.info("Este paciente não tem artefatos de vídeo, áudio ou nota no patients.yaml.")
    else:
        artifact = st.selectbox(
            "Artefato",
            list(options),
            format_func=lambda n: f"{n}: {KIND_LABEL[options[n]]}, "
            f"início {pf.artifact_time(patient_id, n).strftime('%d/%m %H:%M')}",
            key="upload_artifact",
        )
        kind = options[artifact]
        uploaded = st.file_uploader("Arquivo", type=list(UPLOAD_KINDS[kind][1]), key=f"upload_file_{kind}")
        allow_azure = False
        if kind in ("audio", "note"):
            allow_azure = st.checkbox(
                "Autorizar chamadas ao Azure (consomem a cota; sem isso, só respostas já em cache)",
                value=os.environ.get("TC_AUDIO_ALLOW_AZURE") == "1",
            )
        if kind.startswith("video"):
            st.caption("O vídeo roda o YOLOv8 localmente e pode levar alguns minutos.")
        if st.button("Analisar", disabled=uploaded is None, type="primary"):
            with st.spinner("Analisando no módulo dono..."):
                try:
                    res = analyze_upload(
                        uploaded.getvalue(), uploaded.name, patient_id, artifact, out_dir, pf,
                        allow_azure=allow_azure,
                    )
                    st.session_state["upload_msg"] = (
                        ("success", f"{uploaded.name}: {len(res.findings)} Finding(s) gerado(s); fusão atualizada.")
                        if res.ok
                        else ("error", f"{uploaded.name}: a análise falhou ({res.error}).")
                    )
                except ValueError as exc:
                    st.session_state["upload_msg"] = ("error", str(exc))
            st.rerun()  # refaz a fusão com os Findings novos
    if msg := st.session_state.pop("upload_msg", None):
        (st.success if msg[0] == "success" else st.error)(msg[1])

    uploads = list_uploads(out_dir, patient_id)
    if uploads:
        st.markdown("**Envios deste paciente**")
    for up in uploads:
        c1, c2 = st.columns([5, 1])
        status = f"{up['findings']} Finding(s)" if not up.get("error") else f"erro: {up['error']}"
        c1.markdown(
            f"`{up['created_at'][:19].replace('T', ' ')}` · {md_escape(up['filename'])} → "
            f"`{up['artifact']}` · {md_escape(status)}"
        )
        if c2.button("Remover", key=f"rm_{up['upload_id']}"):
            remove_upload(out_dir, up["upload_id"])
            st.rerun()
