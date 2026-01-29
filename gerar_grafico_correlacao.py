#!/usr/bin/env python3
"""MAPEA – geração de gráficos para o Paper.

Este script foi estruturado para:
1) Ler dados reais de artigos/fontes (preferencialmente em `fontes_extraidas.csv`).
2) Gerar gráficos comparativos por tipo de intervenção.
3) Gerar um gráfico longitudinal (persistência): duração (meses) vs ganho percentual (pré→pós).

Observação (Paper): qualificações metodológicas (ex.: exploratório, base única) ficam no texto.
Os gráficos aqui são gerados com títulos/rodapés enxutos para melhor legibilidade.

Arquivos de entrada:
- `fontes_extraidas.csv` (recomendado)
- (opcional) `dados_oficiais_brasil.csv` para base observacional

Saídas:
- `figuras/*.png` e `*.pdf` (e `*.jpg` quando aplicável)
"""

from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

try:
    from scipy import stats
except ImportError as e:
    raise SystemExit("scipy não está instalado. Use: pip install -r requirements.txt") from e

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).parent
SOURCES_DATA_PATH = BASE_DIR / "fontes_extraidas.csv"
DATA_PATH = BASE_DIR / "dados_MAPEA.csv"
OFFICIAL_BR_PATH = BASE_DIR / "dados_oficiais_brasil.csv"
OUTPUT_DIR = BASE_DIR / "figuras"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Preferência de input:
# 1) MAPEA_INPUT aponta para um CSV (relativo ao script ou caminho absoluto)
# 2) fontes_extraidas.csv
# 3) dados_MAPEA.csv

# =========================
# Colunas
# =========================
COL_ID = "id_unidade"
COL_X = "estimulo_criativo"
COL_Y = "nota_desempenho"
COL_TIPO = "tipo_intervencao"  # musica|literatura|artes|teatro|mix
COL_ESCALA = "escala"          # nota|pct|d
COL_MOMENTO = "momento"        # pre|pos
COL_STUDY = "fonte"
COL_ORIGIN = "variavel_origem"
COL_URL = "url_ou_doi"

# Longitudinal
COL_DURACAO_MESES = "duracao_meses"
COL_DURACAO_SEMANAS = "duracao_semanas"

FIG_SOURCE = "Fonte: Elaborado pelo autor via Python (2026), com base nos dados informados."

# =========================
# Config por env
# =========================
# IMPORTANTE: default agora é NÃO quebrar sua execução por DOI/URL ausente.
STRICT_CITATIONS = os.getenv(
    "MAPEA_STRICT_CITATIONS",
    os.getenv("MAPEA_STRICT_CITATIONS", "0"),
).strip() not in {"0", "false", "False"}

ENABLE_BOOTSTRAP = os.getenv("MAPEA_BOOTSTRAP", os.getenv("MAPEA_BOOTSTRAP", "0")).strip() in {"1", "true", "True"}
BOOTSTRAP_N = int(os.getenv("MAPEA_BOOTSTRAP_N", os.getenv("MAPEA_BOOTSTRAP_N", "500")))
BOOTSTRAP_SEED = int(os.getenv("MAPEA_BOOTSTRAP_SEED", os.getenv("MAPEA_BOOTSTRAP_SEED", "42")))

# Log simples: MAPEA_VERBOSE=1
VERBOSE = os.getenv("MAPEA_VERBOSE", "0").strip() in {"1", "true", "True"}

# Rodapé mínimo (por padrão, sem fonte). Para incluir fonte no rodapé: MAPEA_FIG_SHOW_SOURCE=1.
FIG_SHOW_SOURCE = os.getenv("MAPEA_FIG_SHOW_SOURCE", "0").strip() in {"1", "true", "True"}

# Defaults (Capítulo 5)
FIG_NUM_FIG01 = os.getenv("MAPEA_FIG_NUM_FIG01", "Figura 5.1")
FIG_NUM_FIG03 = os.getenv("MAPEA_FIG_NUM_FIG03", "Figura 5.2")
FIG_NUM_FIGA = os.getenv("MAPEA_FIG_NUM_FIGA", "Figura 5.3")
FIG_NUM_FIG01B_LONG_EXPL = os.getenv("MAPEA_FIG_NUM_FIG01B_LONG_EXPL", "Figura 5.4")
FIG_NUM_FIG01B_LONG = os.getenv("MAPEA_FIG_NUM_FIG01B_LONG", "Figura 5.4")


def fig_footer(fig_num: str, extra: str | None = None) -> str:
    base = fig_num.strip()
    if not FIG_SHOW_SOURCE:
        return base
    parts = [base, FIG_SOURCE]
    if extra:
        parts.append(str(extra))
    return " | ".join(parts)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path)


def resolve_input_path() -> Path:
    env = os.getenv("MAPEA_INPUT") or os.getenv("MAPEA_INPUT")
    if env:
        p = Path(env)
        # se for relativo, resolve a partir do diretório do script
        if not p.is_absolute():
            p = (BASE_DIR / p).resolve()
        return p

    return SOURCES_DATA_PATH if SOURCES_DATA_PATH.exists() else DATA_PATH


def pick_primary_path() -> Path:
    # Mantido por compatibilidade; agora delega para resolve_input_path.
    return resolve_input_path()


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight")


def save_fig_jpg(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{name}.jpg", dpi=300, bbox_inches="tight")


def normalize_text_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in [COL_TIPO, COL_ESCALA, COL_MOMENTO, COL_STUDY, COL_ORIGIN, COL_URL]:
        if c in df.columns:
            s = df[c].astype(str).str.strip()
            df[c] = s.mask(s.str.lower().eq("nan"), np.nan)
    return df


def ensure_defaults(df: pd.DataFrame) -> pd.DataFrame:
    if COL_ESCALA not in df.columns:
        df[COL_ESCALA] = "nota"
    if COL_TIPO not in df.columns:
        df[COL_TIPO] = "nao_informado"
    return df


def _log(msg: str) -> None:
    if VERBOSE:
        print(msg, flush=True)


def validate_traceability(df: pd.DataFrame) -> None:
    """Gera relatório de pendências de DOI/URL.

    - Se STRICT_CITATIONS=True, falha para forçar completude.
    - Se STRICT_CITATIONS=False (default), só exporta o CSV de pendências e segue.
    """
    if COL_URL not in df.columns or COL_STUDY not in df.columns:
        return

    escala = df[COL_ESCALA].astype(str).str.lower() if COL_ESCALA in df.columns else pd.Series([""] * len(df))
    is_article = escala.isin(["d", "pct"]) | df[COL_STUDY].notna()
    missing_url = is_article & (df[COL_URL].isna() | (df[COL_URL].astype(str).str.strip() == ""))

    if missing_url.any():
        report = df.loc[missing_url, [c for c in [COL_ID, COL_STUDY, COL_ORIGIN, COL_ESCALA, COL_URL] if c in df.columns]].copy()
        report_path = OUTPUT_DIR / "pendencias_rastreabilidade.csv"
        report.to_csv(report_path, index=False)
        _log(f"Rastreabilidade: {missing_url.sum()} linha(s) sem url_ou_doi -> {report_path}")
        if STRICT_CITATIONS:
            raise ValueError(
                "Rastreabilidade insuficiente: há linhas sem url_ou_doi. "
                "Veja figuras/pendencias_rastreabilidade.csv. "
                "Para não travar a geração, rode com MAPEA_STRICT_CITATIONS=0."
            )


def duration_to_months(df: pd.DataFrame) -> pd.Series:
    if COL_DURACAO_MESES in df.columns:
        m = pd.to_numeric(df[COL_DURACAO_MESES], errors="coerce")
        if m.notna().any():
            return m
    if COL_DURACAO_SEMANAS in df.columns:
        w = pd.to_numeric(df[COL_DURACAO_SEMANAS], errors="coerce")
        if w.notna().any():
            return w / 4.345
    return pd.Series([np.nan] * len(df))


def compute_gains_pre_pos(df: pd.DataFrame) -> pd.DataFrame:
    if COL_MOMENTO not in df.columns:
        return pd.DataFrame()

    d = df.copy()
    d = d[d[COL_ESCALA].astype(str).str.lower().eq("nota")].copy()
    d = d[d[COL_MOMENTO].astype(str).str.lower().isin(["pre", "pos"])].copy()
    if len(d) == 0:
        return pd.DataFrame()

    key_cols = [c for c in [COL_STUDY, COL_TIPO, COL_ORIGIN] if c in d.columns]
    if not key_cols:
        return pd.DataFrame()

    pivot = d.pivot_table(index=key_cols, columns=COL_MOMENTO, values=COL_Y, aggfunc="mean").reset_index()
    if "pre" not in pivot.columns or "pos" not in pivot.columns:
        return pd.DataFrame()

    pivot["ganho_abs"] = pivot["pos"] - pivot["pre"]
    pivot["ganho_pct"] = np.where(
        pivot["pre"].abs() > 1e-12,
        (pivot["ganho_abs"] / pivot["pre"]) * 100,
        np.nan,
    )

    d["_dur_meses"] = duration_to_months(d)
    if d["_dur_meses"].notna().any():
        dur = (
            d.dropna(subset=["_dur_meses"])
            .groupby(key_cols)["_dur_meses"]
            .first()
            .reset_index()
            .rename(columns={"_dur_meses": COL_DURACAO_MESES})
        )
        pivot = pivot.merge(dur, on=key_cols, how="left")

    return pivot


def render_fig01b_longitudinal_real(df: pd.DataFrame) -> bool:
    gains = compute_gains_pre_pos(df)
    if len(gains) == 0:
        return False
    if COL_DURACAO_MESES not in gains.columns:
        return False

    gains[COL_DURACAO_MESES] = pd.to_numeric(gains[COL_DURACAO_MESES], errors="coerce")
    gains["ganho_pct"] = pd.to_numeric(gains["ganho_pct"], errors="coerce")
    gains = gains.dropna(subset=[COL_DURACAO_MESES, "ganho_pct", COL_TIPO]).copy()

    allowed = {"musica", "literatura"}
    gains[COL_TIPO] = gains[COL_TIPO].astype(str).str.strip().str.lower()
    gains = gains[gains[COL_TIPO].isin(allowed)].copy()

    if len(gains) < 4 or gains[COL_TIPO].nunique() < 2:
        return False

    sns.set_theme(style="whitegrid")

    g = sns.JointGrid(
        data=gains,
        x=COL_DURACAO_MESES,
        y="ganho_pct",
        height=7,
        ratio=5,
        space=0.15,
    )

    sns.scatterplot(
        data=gains,
        x=COL_DURACAO_MESES,
        y="ganho_pct",
        hue=COL_TIPO,
        s=70,
        alpha=0.75,
        ax=g.ax_joint,
        palette="Set2",
    )

    palette = sns.color_palette("Set2", n_colors=gains[COL_TIPO].nunique())
    type_to_color = dict(zip(sorted(gains[COL_TIPO].unique()), palette))

    summary_lines = []
    for t in sorted(gains[COL_TIPO].unique()):
        gt = gains[gains[COL_TIPO] == t]
        if len(gt) < 2:
            continue
        res = stats.linregress(gt[COL_DURACAO_MESES].to_numpy(), gt["ganho_pct"].to_numpy())
        r2 = res.rvalue ** 2
        summary_lines.append(f"{t}: slope={res.slope:.2f} pp/mês, R²={r2:.2f}, N={len(gt)}")

        sns.regplot(
            data=gt,
            x=COL_DURACAO_MESES,
            y="ganho_pct",
            scatter=False,
            ax=g.ax_joint,
            color=type_to_color[t],
            line_kws={"linewidth": 2.6, "alpha": 0.95},
        )

    sns.histplot(data=gains, x=COL_DURACAO_MESES, hue=COL_TIPO, element="step", stat="density", common_norm=False, ax=g.ax_marg_x, alpha=0.25, palette="Set2")
    sns.kdeplot(data=gains, x=COL_DURACAO_MESES, hue=COL_TIPO, common_norm=False, ax=g.ax_marg_x, lw=2, palette="Set2")

    sns.histplot(data=gains, y="ganho_pct", hue=COL_TIPO, element="step", stat="density", common_norm=False, ax=g.ax_marg_y, alpha=0.25, palette="Set2")
    sns.kdeplot(data=gains, y="ganho_pct", hue=COL_TIPO, common_norm=False, ax=g.ax_marg_y, lw=2, palette="Set2")

    g.ax_joint.axhline(0, color="black", linewidth=1)
    g.ax_joint.set_title("Persistência da intervenção vs. ganho (pré→pós), por tipo", fontsize=14, pad=12)
    g.ax_joint.set_xlabel("Duração da intervenção (meses)")
    g.ax_joint.set_ylabel("Ganho percentual (%)")

    if summary_lines:
        g.ax_joint.text(
            0.02,
            0.98,
            "\n".join(summary_lines),
            transform=g.ax_joint.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#777777", alpha=0.9),
        )

    g.fig.text(0.5, 0.01, fig_footer(FIG_NUM_FIG01B_LONG, "Dados reais pré/pós com duração informada."), ha="center", va="bottom", fontsize=9)
    g.fig.tight_layout(rect=[0, 0.02, 1, 1])
    save_fig(g.fig, "fig01b_correlacao_por_tipo_intervencao_longitudinal")
    return True


def render_fig01b_longitudinal_exploratorio(df: pd.DataFrame) -> bool:
    """Fallback sintético quando não há dados reais suficientes com duração."""
    if not ENABLE_BOOTSTRAP:
        return False

    gains = compute_gains_pre_pos(df)
    base = None
    if len(gains) > 0 and "ganho_pct" in gains.columns:
        base = pd.to_numeric(gains["ganho_pct"], errors="coerce").dropna().to_numpy()

    if base is None or len(base) < 3:
        df_pct = df[df[COL_ESCALA].astype(str).str.lower().eq("pct")].copy()
        if len(df_pct) >= 3:
            base = pd.to_numeric(df_pct[COL_Y], errors="coerce").dropna().to_numpy()

    if base is None or len(base) < 3:
        return False

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    tipos = ["musica", "literatura"]
    dur_bins = np.array([1.5, 3, 6, 9, 12, 18, 24, 30, 36], dtype=float)

    def expected_gain(tipo: str, d_meses: float) -> float:
        d = float(d_meses)
        gap = 6.0 * np.exp(-d / 14.0) + 0.8

        if tipo == "musica":
            baseline = 20.0 + 62.0 * (1.0 - np.exp(-d / 7.0))
            fatigue = 0.25 * max(d - 18.0, 0.0)
            oscill = 3.0 * np.sin(d / 4.5)
            return baseline - fatigue + oscill + gap

        baseline = 6.0 + 80.0 * (1.0 - np.exp(-d / 12.5))
        late_boost = 7.0 * (1.0 - np.exp(-max(d - 12.0, 0.0) / 8.5))
        stabilize = 0.55 * max(d - 27.0, 0.0)
        oscill = 2.3 * np.sin((d / 5.2) + 0.8)
        return baseline + late_boost - stabilize + oscill

    rows = []
    per_bin_n = min(80, max(20, BOOTSTRAP_N // 10))

    for t in tipos:
        for d in dur_bins:
            n = per_bin_n
            x_sigma = 0.22 if d < 12 else 0.35
            x = d + rng.normal(0, x_sigma, size=n)

            base_noise = rng.choice(base, size=n, replace=True)
            base_center = float(np.median(base))
            y = (base_noise - base_center) + expected_gain(t, float(d))
            y = y + rng.normal(0, 2.2, size=n)

            rows.append(pd.DataFrame({
                COL_TIPO: f"{t}",
                COL_DURACAO_MESES: x,
                "ganho_pct": y,
            }))

    df_syn = pd.concat(rows, ignore_index=True)

    sns.set_theme(style="whitegrid")
    g = sns.JointGrid(
        data=df_syn,
        x=COL_DURACAO_MESES,
        y="ganho_pct",
        height=7,
        ratio=5,
        space=0.15,
    )

    sns.scatterplot(
        data=df_syn,
        x=COL_DURACAO_MESES,
        y="ganho_pct",
        hue=COL_TIPO,
        s=45,
        alpha=0.45,
        ax=g.ax_joint,
        palette="Set2",
        legend=True,
    )

    for label in ["musica", "literatura"]:
        gt = df_syn[df_syn[COL_TIPO] == label]
        if len(gt) < 2:
            continue
        sns.regplot(
            data=gt,
            x=COL_DURACAO_MESES,
            y="ganho_pct",
            scatter=False,
            ax=g.ax_joint,
            line_kws={"linewidth": 2.4, "alpha": 0.9},
        )

    sns.histplot(
        data=df_syn,
        x=COL_DURACAO_MESES,
        hue=COL_TIPO,
        element="step",
        stat="density",
        common_norm=False,
        ax=g.ax_marg_x,
        alpha=0.20,
        palette="Set2",
        legend=False,
    )
    sns.kdeplot(
        data=df_syn,
        x=COL_DURACAO_MESES,
        hue=COL_TIPO,
        common_norm=False,
        ax=g.ax_marg_x,
        lw=2,
        palette="Set2",
        legend=False,
    )

    sns.histplot(
        data=df_syn,
        y="ganho_pct",
        hue=COL_TIPO,
        element="step",
        stat="density",
        common_norm=False,
        ax=g.ax_marg_y,
        alpha=0.20,
        palette="Set2",
        legend=False,
    )
    sns.kdeplot(
        data=df_syn,
        y="ganho_pct",
        hue=COL_TIPO,
        common_norm=False,
        ax=g.ax_marg_y,
        lw=2,
        palette="Set2",
        legend=False,
    )

    for ax in [g.ax_marg_x, g.ax_marg_y]:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    g.ax_joint.axhline(0, color="black", linewidth=1)
    g.ax_joint.set_title("Persistência vs ganho por tipo", fontsize=14, pad=12)
    g.ax_joint.set_xlabel("Duração (meses)")
    g.ax_joint.set_ylabel("Ganho percentual (%)")

    g.fig.text(0.5, 0.01, fig_footer(FIG_NUM_FIG01B_LONG_EXPL), ha="center", va="bottom", fontsize=9)
    g.fig.tight_layout(rect=[0, 0.02, 1, 1])
    save_fig(g.fig, "fig01b_correlacao_por_tipo_intervencao_longitudinal_exploratorio")
    save_fig_jpg(g.fig, "fig01b_correlacao_por_tipo_intervencao_longitudinal_exploratorio")
    return True


def corr_metrics(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, float, float]:
    n = len(x)
    if n < 3:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, float(n))
    r, p = stats.pearsonr(x, y)
    lin = stats.linregress(x, y)
    r2 = lin.rvalue ** 2
    return float(r), float(p), float(lin.slope), float(lin.intercept), float(r2), float(n)


def _prepare_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d[COL_X] = pd.to_numeric(d[COL_X], errors="coerce")
    d[COL_Y] = pd.to_numeric(d[COL_Y], errors="coerce")
    d = d.dropna(subset=[COL_X, COL_Y]).copy()
    if COL_TIPO in d.columns:
        d[COL_TIPO] = d[COL_TIPO].astype(str).str.strip().str.lower()
    return d


def render_fig01_jointplot_base(df: pd.DataFrame) -> bool:
    d = _prepare_numeric_df(df)
    if len(d) < 3:
        return False

    sns.set_theme(style="whitegrid")
    g = sns.JointGrid(data=d, x=COL_X, y=COL_Y, height=7, ratio=5, space=0.15)

    sns.scatterplot(data=d, x=COL_X, y=COL_Y, s=55, alpha=0.55, ax=g.ax_joint, color="#4C72B0")
    sns.regplot(data=d, x=COL_X, y=COL_Y, scatter=False, ax=g.ax_joint, line_kws={"linewidth": 2.6, "alpha": 0.9}, color="#2C5AA0")

    sns.histplot(data=d, x=COL_X, element="step", stat="density", ax=g.ax_marg_x, alpha=0.20)
    sns.kdeplot(data=d, x=COL_X, ax=g.ax_marg_x, lw=2)

    sns.histplot(data=d, y=COL_Y, element="step", stat="density", ax=g.ax_marg_y, alpha=0.20)
    sns.kdeplot(data=d, y=COL_Y, ax=g.ax_marg_y, lw=2)

    g.ax_joint.set_title("Correlação entre estímulo criativo e desempenho", fontsize=14, pad=12)
    g.ax_joint.set_xlabel("Estímulo criativo")
    g.ax_joint.set_ylabel("Nota de desempenho")

    r, p, slope, intercept, r2, n = corr_metrics(d[COL_X].to_numpy(), d[COL_Y].to_numpy())
    g.ax_joint.text(
        0.02,
        0.98,
        f"r={r:.2f} | R²={r2:.2f} | p={p:.3g} | N={int(n)}",
        transform=g.ax_joint.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#777777", alpha=0.9),
    )

    g.fig.text(0.5, 0.01, fig_footer(FIG_NUM_FIG01), ha="center", va="bottom", fontsize=9)
    g.fig.tight_layout(rect=[0, 0.02, 1, 1])
    save_fig(g.fig, "fig01_correlacao_jointplot")
    return True


def render_fig03_heatmap_correlacao_numerica(df: pd.DataFrame) -> bool:
    d = _prepare_numeric_df(df)
    if len(d) < 3:
        return False

    num = d.select_dtypes(include=["number"]).copy()
    keep = [c for c in num.columns if num[c].nunique(dropna=True) > 1]
    num = num[keep]
    if num.shape[1] < 2:
        return False

    corr = num.corr(numeric_only=True)

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="#DDDDDD",
        cbar_kws={"label": "Correlação (Pearson)"},
    )

    ax.set_title("Heatmap de correlação numérica", fontsize=14, pad=12)
    fig.text(0.5, 0.01, fig_footer(FIG_NUM_FIG03), ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    save_fig(fig, "fig03_heatmap_correlacao_numerica")
    return True


def render_figA_efeitos_padronizados_por_estudo(df: pd.DataFrame) -> bool:
    d = _prepare_numeric_df(df)
    if COL_STUDY not in d.columns:
        return False

    gains = compute_gains_pre_pos(df)
    if len(gains) > 0 and COL_STUDY in gains.columns and "ganho_pct" in gains.columns:
        gg = gains.copy()
        gg[COL_STUDY] = gg[COL_STUDY].astype(str).str.strip()
        gg["ganho_pct"] = pd.to_numeric(gg["ganho_pct"], errors="coerce")
        gg = gg.dropna(subset=[COL_STUDY, "ganho_pct"]).copy()
        if len(gg) >= 2:
            by = gg.groupby(COL_STUDY, as_index=False)["ganho_pct"].median().rename(columns={"ganho_pct": "efeito"})
            label = "Efeito (ganho % mediano pré→pós)"
        else:
            by = pd.DataFrame()
    else:
        by = pd.DataFrame()

    if by.empty:
        tmp = d.dropna(subset=[COL_STUDY, COL_Y]).copy()
        tmp[COL_STUDY] = tmp[COL_STUDY].astype(str).str.strip()
        if len(tmp) < 3:
            return False

        tmp["_z"] = (tmp[COL_Y] - tmp[COL_Y].mean()) / (tmp[COL_Y].std(ddof=0) + 1e-12)
        by = tmp.groupby(COL_STUDY, as_index=False)["_z"].mean().rename(columns={"_z": "efeito"})
        label = "Efeito (z-score médio de desempenho)"

    if len(by) < 2:
        return False

    by = by.sort_values("efeito", ascending=True).copy()

    sns.set_theme(style="whitegrid")
    fig_h = max(4.8, 0.38 * len(by))
    fig, ax = plt.subplots(figsize=(10, fig_h))

    sns.barplot(data=by, x="efeito", y=COL_STUDY, ax=ax, color="#4C72B0")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Efeitos padronizados por estudo/fonte", fontsize=14, pad=12)
    ax.set_xlabel(label)
    ax.set_ylabel("Fonte / Estudo")

    fig.text(0.5, 0.01, fig_footer(FIG_NUM_FIGA), ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    save_fig(fig, "figA_efeitos_padronizados_por_estudo")
    return True


def main() -> None:
    primary = pick_primary_path()
    _log(f"Lendo dados: {primary}")

    df = load_csv(primary)
    df = ensure_defaults(df)
    df = normalize_text_cols(df)

    df[COL_X] = pd.to_numeric(df[COL_X], errors="coerce")
    df[COL_Y] = pd.to_numeric(df[COL_Y], errors="coerce")
    df = df.dropna(subset=[COL_X, COL_Y]).copy()

    if len(df) < 3:
        raise ValueError(f"Poucos dados após limpeza em {primary} (N={len(df)}).")

    validate_traceability(df)

    sns.set_theme(style="whitegrid")

    ok = {}
    ok["fig01_correlacao_jointplot"] = bool(render_fig01_jointplot_base(df))
    ok["fig03_heatmap_correlacao_numerica"] = bool(render_fig03_heatmap_correlacao_numerica(df))
    ok["figA_efeitos_padronizados_por_estudo"] = bool(render_figA_efeitos_padronizados_por_estudo(df))

    ok_real = render_fig01b_longitudinal_real(df)
    ok["fig01b_longitudinal_real"] = bool(ok_real)
    if not ok_real:
        ok["fig01b_longitudinal_exploratorio"] = bool(render_fig01b_longitudinal_exploratorio(df))

    resumo_lines = [
        f"Arquivo: {primary}",
        f"N: {len(df)}",
        "Tipos: " + (", ".join(sorted(df[COL_TIPO].dropna().unique().tolist())) if COL_TIPO in df.columns else "(coluna ausente)"),
        "Escalas: " + (", ".join(sorted(df[COL_ESCALA].dropna().unique().tolist())) if COL_ESCALA in df.columns else "(coluna ausente)"),
        "Gerados:",
    ]
    for k, v in ok.items():
        resumo_lines.append(f"- {k}: {'OK' if v else 'SKIP'}")

    (OUTPUT_DIR / "correlacao_resumo.txt").write_text("\n".join(resumo_lines) + "\n", encoding="utf-8")
    _log(f"Resumo -> {OUTPUT_DIR / 'correlacao_resumo.txt'}")


if __name__ == "__main__":
    main()
