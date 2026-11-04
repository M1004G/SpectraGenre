"""Step 5: Streamlit demo.  Run:  streamlit run app.py"""
import tempfile
from pathlib import Path
from string import Template

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from matplotlib.colors import LinearSegmentedColormap

from config import CLIP_SECONDS, N_MELS, SR
from evaluate import load_results, mean_std
from features import extract
from model import GenreCNN, predict_proba

# ---------------------------------------------------------------------------
# Design tokens: the only place colours, fonts and spacing are defined.
# One accent (oxide red) on warm paper; everything else is ink, grey or hairline.
# Serif for headings, sans for text, mono for numbers.
# ---------------------------------------------------------------------------
T = dict(
    bg="#F6F4EE", ink="#1B1B19", muted="#6C6A62", line="#D8D4C8",
    accent="#B3401D", neutral="#8F8C80", tint="#ECE8DC",
    serif="'Source Serif 4', Georgia, serif",
    sans="'IBM Plex Sans', 'Segoe UI', Helvetica, sans-serif",
    mono="'IBM Plex Mono', Consolas, monospace",
    s1="4px", s2="8px", s3="16px", s4="28px", s5="48px",   # spacing scale
)

CSS = Template("""<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:wght@600&display=swap');
.stApp { background: $bg; color: $ink; }
.stApp, .stApp p, .stApp label, .stApp li { font-family: $sans; }
#MainMenu, footer, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
.block-container { max-width: 1040px; padding: $s5 $s4; }

.eyebrow { font: 500 0.72rem $sans; letter-spacing: 0.09em; text-transform: uppercase; color: $muted; }
.title { font: 600 2.1rem/1.1 $serif; letter-spacing: -0.01em; margin: $s2 0 $s3; }
.lede { max-width: 62ch; color: $muted; font-size: 0.95rem; line-height: 1.55; margin: 0 0 $s4; }
.section { font: 600 1.15rem $serif; margin: $s5 0 $s2; }
.note { color: $muted; font-size: 0.82rem; line-height: 1.5; margin: $s2 0 $s3; max-width: 70ch; }

.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0 $s4; margin: $s3 0 $s2; }
.m { border-top: 1px solid $ink; padding: $s2 0 $s3; }
.m-value { font: 500 1.3rem $mono; font-variant-numeric: tabular-nums; margin-top: $s1; white-space: nowrap; }
.m-sub { color: $muted; font-size: 0.78rem; margin-top: $s1; }

table.t { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
table.t th { font: 500 0.72rem $sans; letter-spacing: 0.09em; text-transform: uppercase; color: $muted;
             text-align: left; padding: $s2 0; border-bottom: 1px solid $ink; }
table.t td { padding: 6px 0; border-bottom: 1px solid $line; }
table.t .num { font-family: $mono; text-align: right; font-variant-numeric: tabular-nums; }

button[data-baseweb="tab"] { background: transparent; color: $muted; font-family: $sans; font-size: 0.9rem; padding: $s2 $s4 $s2 0; }
button[data-baseweb="tab"][aria-selected="true"] { color: $ink; }
[data-baseweb="tab-highlight"] { background: $accent; height: 2px; }
[data-baseweb="tab-border"] { background: $line; }
[data-testid="stFileUploaderDropzone"] { background: transparent; border: 1px dashed $neutral; border-radius: 2px; }
[data-testid="stFileUploaderDropzone"] button { background: transparent; border: 1px solid $ink; border-radius: 2px; color: $ink; }
[data-testid="stAlert"] { border-radius: 2px; }
</style>""")

CMAP = LinearSegmentedColormap.from_list("paper_to_accent", [T["tint"], T["accent"]])


# ----------------------------- small UI helpers -----------------------------
def html(s):
    st.markdown(s, unsafe_allow_html=True)


def metrics(items):
    """items: list of (label, value, sub-label)"""
    cells = "".join(
        f'<div class="m"><div class="eyebrow">{l}</div><div class="m-value">{v}</div><div class="m-sub">{s}</div></div>'
        for l, v, s in items
    )
    html(f'<div class="metrics">{cells}</div>')


def table(header, rows, text_cols=1):
    th = "".join(f'<th class="{"num" if i >= text_cols else ""}">{h}</th>' for i, h in enumerate(header))
    body = "".join(
        "<tr>" + "".join(f'<td class="{"num" if i >= text_cols else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>"
        for r in rows
    )
    html(f'<table class="t"><tr>{th}</tr>{body}</table>')


def fig_ax(w, h):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=T["bg"])
    ax.set_facecolor(T["bg"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(T["line"])
    ax.tick_params(colors=T["muted"], labelsize=8, length=3, color=T["line"])
    ax.xaxis.label.set_color(T["muted"])
    ax.yaxis.label.set_color(T["muted"])
    return fig, ax


def show(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------- charts -----------------------------------
def spectrogram_chart(log_mel):
    fig, ax = fig_ax(5, 3)
    ax.imshow(log_mel, origin="lower", aspect="auto", cmap=CMAP, extent=[0, CLIP_SECONDS, 0, N_MELS])
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Mel band", fontsize=8)
    show(fig)


def prob_chart(classes, probs):
    order = np.argsort(probs)
    fig, ax = fig_ax(5, 3)
    colors = [T["accent"] if i == order[-1] else T["neutral"] for i in order]
    ax.barh([classes[i] for i in order], probs[order], color=colors, height=0.6)
    for pos, i in enumerate(order):
        ax.text(probs[i] + 0.01, pos, f"{probs[i]:.2f}", va="center", fontsize=8, color=T["muted"])
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("CNN probability", fontsize=8)
    show(fig)


def f1_chart(classes, svm_f1, cnn_f1):
    fig, ax = fig_ax(6, 3.2)
    x, w = np.arange(len(classes)), 0.38
    ax.bar(x - w / 2, svm_f1, w, color=T["neutral"], label="MFCC + SVM")
    ax.bar(x + w / 2, cnn_f1, w, color=T["accent"], label="Mel CNN")
    ax.set_xticks(x, classes, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1", fontsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=T["muted"])
    show(fig)


def confusion_chart(cm, classes):
    cm = np.array(cm)
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = fig_ax(4.6, 4.1)
    ax.imshow(norm, cmap=CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else T["ink"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Predicted genre", fontsize=8)
    ax.set_ylabel("True genre", fontsize=8)
    show(fig)


# ---------------------------------- models ----------------------------------
@st.cache_resource
def load_models():
    svm = joblib.load("models/svm.joblib") if Path("models/svm.joblib").exists() else None
    ckpts = sorted(Path("models").glob("cnn_seed*.pt"))
    cnn = None
    if ckpts:
        ck = torch.load(ckpts[0], map_location="cpu")
        net = GenreCNN(len(ck["classes"]))
        net.load_state_dict(ck["state"])
        cnn = (net, ck["mean"], ck["std"], ck["classes"])
    return svm, cnn


# ----------------------------------- tabs -----------------------------------
def classify_tab():
    svm, cnn = load_models()
    if svm is None or cnn is None:
        html('<div class="note">No trained models found. Run baseline_svm.py and train_cnn.py first.</div>')
        return
    net, mean, std, classes = cnn

    html('<div class="eyebrow" style="margin-top:16px">Input</div>')
    up = st.file_uploader("Audio clip", type=["mp3", "wav", "ogg", "flac"], label_visibility="collapsed")
    html('<div class="note">The first 30 seconds are analysed, resampled to 22.05 kHz mono. '
         'The models know eight genres only, so every clip is assigned one of them.</div>')
    if up is None:
        return

    data = up.getvalue()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / ("clip" + Path(up.name).suffix)
        path.write_bytes(data)
        y, _ = librosa.load(path, sr=SR, mono=True, duration=CLIP_SECONDS)
    log_mel, mfcc = extract(y)

    probs = predict_proba(net, log_mel[None], mean, std)[0]
    cnn_pred = classes[int(probs.argmax())]
    svm_pred = svm["classes"][int(svm["model"].predict(mfcc[None])[0])]

    st.audio(data)
    metrics([
        ("CNN prediction", cnn_pred, f"{probs.max():.0%} confidence"),
        ("SVM prediction", svm_pred, "MFCC baseline"),
        ("Models agree", "Yes" if cnn_pred == svm_pred else "No", "same top genre"),
    ])
    left, right = st.columns(2, gap="large")
    with left:
        html('<div class="eyebrow">Mel spectrogram</div>')
        spectrogram_chart(log_mel)
    with right:
        html('<div class="eyebrow">Genre probabilities</div>')
        prob_chart(classes, probs)


def results_tab():
    svm, cnn = load_results("svm"), load_results("cnn_seed")
    if not svm or not cnn:
        html('<div class="note">No results yet. Run baseline_svm.py and train_cnn.py first.</div>')
        return
    s, classes = svm[0], svm[0]["classes"]

    metrics([
        ("SVM accuracy", f'{s["accuracy"]:.3f}', "MFCC summary, RBF kernel"),
        ("SVM macro-F1", f'{s["macro_f1"]:.3f}', "unweighted mean over genres"),
        ("CNN accuracy", mean_std([r["accuracy"] for r in cnn]), f"{len(cnn)} seed(s), mean and std"),
        ("CNN macro-F1", mean_std([r["macro_f1"] for r in cnn]), f"{len(cnn)} seed(s), mean and std"),
        ("Chance level", f"{1 / len(classes):.3f}", f"{len(classes)} balanced genres"),
    ])
    html('<div class="note">All numbers are on the official FMA test split. Hyperparameters and the '
         'stopping epoch were chosen on the validation split only.</div>')

    svm_f1 = [s["per_genre_f1"][c] for c in classes]
    cnn_f1 = [float(np.mean([r["per_genre_f1"][c] for r in cnn])) for c in classes]
    html('<div class="section">Per-genre F1</div>')
    left, right = st.columns([3, 2], gap="large")
    with left:
        f1_chart(classes, svm_f1, cnn_f1)
    with right:
        table(["Genre", "SVM", "CNN"], [[c, f"{a:.3f}", f"{b:.3f}"] for c, a, b in zip(classes, svm_f1, cnn_f1)])

    html('<div class="section">Where the CNN confuses genres</div>')
    cm = np.array(cnn[0]["confusion_matrix"])
    left, right = st.columns([3, 2], gap="large")
    with left:
        confusion_chart(cm, classes)
    with right:
        html(f'<div class="note">Seed {cnn[0].get("seed", 0)}. Rows are the true genre, columns the predicted one. '
             'Cell numbers are clip counts; shading is the share of each row.</div>')
        html('<div class="eyebrow">Most frequent confusions</div>')
        pairs = sorted(((cm[i, j], classes[i], classes[j])
                        for i in range(len(classes)) for j in range(len(classes)) if i != j), reverse=True)[:6]
        table(["True genre", "Predicted as", "Clips"], [[t, p, n] for n, t, p in pairs], text_cols=2)


def main():
    st.set_page_config(page_title="MelGenre", layout="wide")
    html(CSS.substitute(T))
    html('<div class="eyebrow">Music information retrieval</div>'
         '<div class="title">MelGenre</div>'
         '<p class="lede">Genre classification on FMA-small. An MFCC and SVM baseline is compared '
         'against a small CNN trained on mel spectrograms.</p>')
    tab_classify, tab_eval = st.tabs(["Classify a clip", "Evaluation"])
    with tab_classify:
        classify_tab()
    with tab_eval:
        results_tab()


main()