#!/usr/bin/env python3
"""Gera um diagrama do modelo MAPEA (Análise Preditiva de Estímulo de Aprendizagem).

Saídas:
- figuras/MAPEA_modelo.(png|pdf|jpg)
- figuras/MAPEA_modelo_mono.(png|pdf|jpg)

Dependências:
- matplotlib

Uso:
    python3 gerar_grafico_MAPEA_modelo.py

Obs. (Paper): o rodapé dentro da imagem fica curto (ex.: "Figura 4.1").
A fonte deve ir na legenda do Word.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.patches import Circle


# Rodapé interno (curto)
FIG_NUM = os.getenv("MAPEA_FIG_NUM_MODELO", "Figura 4.1")
FOOTER_SHOW = os.getenv("MAPEA_FIG_SHOW_FOOTER_MODELO", "1").strip() not in {"0", "false", "False"}


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float
    title: str
    body: str
    facecolor: str
    edgecolor: str = "#1f2937"

    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)

    def top(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h)

    def bottom(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y)

    def left(self) -> tuple[float, float]:
        return (self.x, self.y + self.h / 2)

    def right(self) -> tuple[float, float]:
        return (self.x + self.w, self.y + self.h / 2)


def _add_box(ax, b: Box, *, title_color: str = "#111827", text_color: str = "#111827") -> None:
    patch = FancyBboxPatch(
        (b.x, b.y),
        b.w,
        b.h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.2,
        facecolor=b.facecolor,
        edgecolor=b.edgecolor,
    )
    ax.add_patch(patch)

    # Ajuste de padding vertical: menos espaço no cabeçalho e mais área útil para o corpo
    title_top_pad = 0.030
    body_top_pad = 0.070

    ax.text(
        b.x + 0.018,
        b.y + b.h - title_top_pad,
        b.title,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
        color=title_color,
    )

    ax.text(
        b.x + 0.018,
        b.y + b.h - body_top_pad,
        b.body,
        fontsize=9.0,
        va="top",
        ha="left",
        color=text_color,
        linespacing=1.16,
    )


def _arrow(
    ax,
    src: tuple[float, float],
    dst: tuple[float, float],
    text: str | None = None,
    rad: float = 0.0,
    *,
    color: str = "#111827",
    linewidth: float = 1.2,
    mutation_scale: float = 12,
    text_offset: tuple[float, float] = (0.0, 0.015),
    text_kwargs: dict | None = None,
) -> None:
    arr = FancyArrowPatch(
        src,
        dst,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)

    if text:
        mx = (src[0] + dst[0]) / 2
        my = (src[1] + dst[1]) / 2
        kw = {
            "fontsize": 8.3,
            "ha": "center",
            "va": "bottom",
            "color": color,
        }
        if text_kwargs:
            kw.update(text_kwargs)
        ax.text(mx + text_offset[0], my + text_offset[1], text, **kw)


def _arrow_elbow(
    ax,
    src: tuple[float, float],
    dst: tuple[float, float],
    *,
    via: tuple[float, float],
    text: str | None = None,
    color: str = "#111827",
    linewidth: float = 1.2,
    mutation_scale: float = 12,
    text_offset: tuple[float, float] = (0.0, 0.012),
    text_kwargs: dict | None = None,
) -> None:
    """Desenha uma seta em 2 segmentos (cotovelo)."""
    # Segmento 1 (sem ponta)
    seg1 = FancyArrowPatch(
        src,
        via,
        arrowstyle="-",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(seg1)

    # Segmento 2 (com ponta)
    seg2 = FancyArrowPatch(
        via,
        dst,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(seg2)

    if text:
        mx = (via[0] + dst[0]) / 2
        my = (via[1] + dst[1]) / 2
        kw = {
            "fontsize": 8.3,
            "ha": "center",
            "va": "bottom",
            "color": color,
            "bbox": dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
        }
        if text_kwargs:
            kw.update(text_kwargs)
        ax.text(mx + text_offset[0], my + text_offset[1], text, **kw)


def _add_center_node(
    ax,
    *,
    x: float,
    y: float,
    r: float,
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str,
    linewidth: float = 1.4,
    title_color: str = "#111827",
    text_color: str = "#111827",
) -> None:
    circle = Circle((x, y), r, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)
    ax.add_patch(circle)

    ax.text(
        x,
        y + r * 0.35,
        title,
        fontsize=11,
        fontweight="bold",
        va="center",
        ha="center",
        color=title_color,
    )
    ax.text(
        x,
        y + r * 0.10,
        body,
        fontsize=9.0,
        va="top",
        ha="center",
        color=text_color,
        linespacing=1.18,
    )


def _build_diagram(ax, *, mono: bool) -> None:
    title_color = "#111827"
    text_color = "#111827"
    arrow_color = "#111827"

    # Para impressão P&B (padrão mais acadêmico)
    if mono:
        box_face_default = "none"   # sem preenchimento
        box_edge_default = "#111827"  # contorno preto/cinza escuro
        arrow_lw = 1.8
        arrow_ms = 14
        box_lw = 1.6
        center_lw = 1.8
        label_color = "#111827"
    else:
        box_face_default = None
        box_edge_default = None
        arrow_lw = 1.2
        arrow_ms = 12
        box_lw = 1.2
        center_lw = 1.4
        label_color = "#374151"

    # Paleta
    if mono:
        # estilo monocromático acadêmico: sem preenchimento + contornos
        c1 = c2 = c3 = c4 = c5 = c6 = c7 = c8 = cb = box_face_default
        e1 = e2 = e3 = e4 = e5 = e6 = e7 = e8 = box_edge_default
    else:
        c1, e1 = "#e0f2fe", "#0ea5e9"  # dados
        c2, e2 = "#dbeafe", "#2563eb"  # segmentação
        c3, e3 = "#ecfeff", "#06b6d4"  # pipeline (reservado)
        c4, e4 = "#ecfccb", "#84cc16"  # estímulos
        c5, e5 = "#fef9c3", "#eab308"  # mecanismos
        c6, e6 = "#ede9fe", "#8b5cf6"  # modelo
        c7, e7 = "#ffe4e6", "#fb7185"  # intervenção
        c8, e8 = "#f3f4f6", "#6b7280"  # resultados
        cb = "#ffffff"  # futuro/neutro

    # --- GRID DE LAYOUT ---
    # Objetivo: distribuição uniforme horizontal (3 colunas) e vertical (topo / meio / base)
    # sem encostar no título nem no rodapé.
    y_shift = -0.020

    col_left_x = 0.07
    col_mid_x = 0.375
    col_right_x = 0.72

    top_y = 0.785 + y_shift
    mid_y = 0.545 + y_shift
    bottom_y = 0.145 + y_shift

    h_top = 0.130
    h_mid = 0.130
    h_bottom = 0.150

    w_left = 0.255
    w_mid = 0.265
    w_right = 0.245

    # --- LINHA DO TOPO (3 cards) ---
    # Mantém TODOS na mesma linha e com a mesma altura para o alinhamento ficar perfeito.
    dados = Box(
        x=col_left_x,
        y=top_y,
        w=w_left,
        h=h_top,
        title="Dados e Contexto (macro)",
        body=(
            "• INEP/SAEB/PISA/IBGE\n"
            "• Indicadores educacionais\n"
            "• Socioeconomia\n"
            "• Infraestrutura"
        ),
        facecolor=c1,
        edgecolor=e1,
    )

    estimulos = Box(
        x=col_mid_x,
        y=top_y,
        w=w_mid,
        h=h_top,
        title="Variáveis de Estímulo",
        body=(
            "• Música\n"
            "• Literatura\n"
            "• Artes/teatro/criatividade\n"
            "(preditoras)"
        ),
        facecolor=c4,
        edgecolor=e4,
    )

    modelo = Box(
        x=col_right_x,
        y=top_y,
        w=w_right,
        h=h_top,
        title="Modelo MAPEA (IA/Predição)",
        body=(
            "• Predição por perfil/região\n"
            "• Cenários (risco x proteção)\n"
            "• Recomenda intervenção\n"
            "• Explicabilidade/ética"
        ),
        facecolor=c6,
        edgecolor=e6,
    )

    # --- LINHA DO MEIO (2 cards + nó central) ---
    segmentacao = Box(
        x=col_left_x,
        y=mid_y,
        w=w_left,
        h=h_mid,
        title="Segmentação (região/perfil)",
        body=(
            "• Região/UF/Município\n"
            "• Vulnerabilidade\n"
            "• Clusters\n"
            "• Priorização"
        ),
        facecolor=c2,
        edgecolor=e2,
    )

    interv = Box(
        x=col_right_x,
        y=mid_y,
        w=w_right,
        h=h_mid,
        title="Intervenções (personalizadas)",
        body=(
            "• Plano (música/leitura)\n"
            "• Trilhas personalizadas\n"
            "• Suporte socioemocional\n"
            "• Intensidade/aderência"
        ),
        facecolor=c7,
        edgecolor=e7,
    )

    # Nó central no centro vertical da linha do meio (sobe um pouco para não encostar na faixa inferior)
    center_xy = (0.52, mid_y + h_mid * 0.55)

    # --- LINHA DE BAIXO (3 cards) ---
    ontologia = Box(
        x=col_left_x,
        y=bottom_y,
        w=w_left,
        h=h_bottom,
        title="(Futuro) Ontologia/KG (padrões)",
        body=(
            "• Conceitos (estímulo→mecanismo→ação)\n"
            "• Evidências + rastreabilidade\n"
            "• RAG/explicabilidade\n"
            "• Reuso entre estudos"
        ),
        facecolor=cb,
        edgecolor=box_edge_default if mono else "#111827",
    )

    mecanismos = Box(
        x=col_mid_x,
        y=bottom_y,
        w=w_mid,
        h=h_bottom,
        title="Mecanismos (mediadores)",
        body=(
            "• Atenção/foco\n"
            "• Memória de trabalho\n"
            "• Motivação\n"
            "• Autorregulação"
        ),
        facecolor=c5,
        edgecolor=e5,
    )

    resultados = Box(
        x=col_right_x,
        y=bottom_y,
        w=w_right,
        h=h_bottom,
        title="Resultados + Monitoramento",
        body=(
            "• Proficiência, evasão, frequência\n"
            "• Bem-estar/engajamento\n"
            "• Impacto (antes/depois)\n"
            "• Ajustes + re-treino"
        ),
        facecolor=c8,
        edgecolor=e8,
    )

    # --- nó central ---
    center_r = 0.082
    center_title = "Aluno / Cluster"
    center_body = "histórico\npreferências\nengajamento\nnecessidades"
    center_face = box_face_default if mono else "#fff7ed"
    center_edge = box_edge_default if mono else "#f97316"

    # Ajusta espessura de contorno no modo mono (melhor impressão)
    # sem alterar a assinatura pública de Box.
    def _box_with_linewidth(b: Box) -> FancyBboxPatch:
        patch = FancyBboxPatch(
            (b.x, b.y),
            b.w,
            b.h,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=box_lw,
            facecolor=b.facecolor,
            edgecolor=b.edgecolor,
        )
        ax.add_patch(patch)

        # Mesmo ajuste de padding vertical da função _add_box (mantém consistência)
        title_top_pad = 0.030
        body_top_pad = 0.070

        ax.text(
            b.x + 0.018,
            b.y + b.h - title_top_pad,
            b.title,
            fontsize=11,
            fontweight="bold",
            va="top",
            ha="left",
            color=title_color,
        )

        ax.text(
            b.x + 0.018,
            b.y + b.h - body_top_pad,
            b.body,
            fontsize=9.0,
            va="top",
            ha="left",
            color=text_color,
            linespacing=1.16,
        )

        return patch

    for b in (dados, segmentacao, estimulos, mecanismos, modelo, interv, resultados, ontologia):
        _box_with_linewidth(b)

    _add_center_node(
        ax,
        x=center_xy[0],
        y=center_xy[1],
        r=center_r,
        title=center_title,
        body=center_body,
        facecolor=center_face,
        edgecolor=center_edge,
        linewidth=center_lw,
        title_color=title_color,
        text_color=text_color,
    )

    # === SETAS (recalculadas para o grid) ===
    cx, cy = center_xy

    _arrow(
        ax,
        dados.bottom(),
        segmentacao.top(),
        None,
        rad=0.0,
        color=arrow_color,
        linewidth=arrow_lw,
        mutation_scale=arrow_ms,
    )

    _arrow(
        ax,
        (segmentacao.x + segmentacao.w, segmentacao.y + segmentacao.h * 0.55),
        (cx - center_r, cy + 0.01),
        None,
        rad=-0.20,
        color=arrow_color,
        linewidth=arrow_lw,
        mutation_scale=arrow_ms,
    )

    _arrow(
        ax,
        (estimulos.x + estimulos.w * 0.55, estimulos.y),
        (cx, cy + center_r),
        None,
        rad=0.0,
        color=arrow_color,
        linewidth=arrow_lw,
        mutation_scale=arrow_ms,
    )

    _arrow(
        ax,
        (cx, cy - center_r),
        (mecanismos.x + mecanismos.w * 0.55, mecanismos.y + mecanismos.h),
        None,
        rad=0.0,
        color=arrow_color,
        linewidth=arrow_lw,
        mutation_scale=arrow_ms,
    )

    _arrow(
        ax,
        (cx + center_r, cy + 0.015),
        (modelo.x, modelo.y + modelo.h * 0.55),
        None,
        rad=0.22,
        color=arrow_color,
        linewidth=arrow_lw,
        mutation_scale=arrow_ms,
    )

    _arrow(
        ax,
        modelo.bottom(),
        interv.top(),
        None,
        rad=0.0,
        color=arrow_color,
        linewidth=arrow_lw,
        mutation_scale=arrow_ms,
    )

    _arrow(
        ax,
        (interv.x + interv.w * 0.85, interv.y + interv.h * 0.15),
        (resultados.x + resultados.w * 0.65, resultados.y + resultados.h),
        None,
        rad=-0.18,
        color=arrow_color,
        linewidth=arrow_lw,
        mutation_scale=arrow_ms,
    )

    # Feedback: Resultados → Aluno
    # Mantém a seta direta (curva suave), e remove a alternativa em cotovelo
    # que estava cruzando outras ligações no diagrama.
    _arrow(
        ax,
        (resultados.x + resultados.w * 0.25, resultados.y + resultados.h),
        (cx + center_r * 0.15, cy - center_r * 0.65),
        None,
        rad=0.28,
        color=arrow_color,
        linewidth=arrow_lw,
        mutation_scale=arrow_ms,
    )

    _arrow(
        ax,
        (ontologia.x + ontologia.w, ontologia.y + ontologia.h * 0.85),
        (segmentacao.x + segmentacao.w * 0.25, segmentacao.y),
        None,
        rad=0.30,
        color=arrow_color,
        linewidth=arrow_lw,
        mutation_scale=arrow_ms,
    )

    # Rodapé: sobe um pouco para não encostar na faixa inferior
    ax.text(
        0.05,
        0.060,
        "Foco no indivíduo/grupo: o perfil do estudante/turma é o núcleo do MAPEA (entrada de novas variáveis + recomendação + feedback).",
        fontsize=9,
        color="#374151",
        ha="left",
        va="bottom",
    )
    ax.text(
        0.05,
        0.040,
        "Evolução: v1 conceitual → v2 personalização por perfil/região → v3 ontologia/KG + IA (RAG/explicabilidade).",
        fontsize=9,
        color="#374151",
        ha="left",
        va="bottom",
    )


def _render(*, mono: bool, out_png: Path, out_pdf: Path, out_jpg: Path) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "Preditivo de Estímulo de Aprendizagem (MAPEA)",
        fontsize=18,
        fontweight="bold",
        x=0.5,
        y=0.965,
        color="#111827",
    )

    _build_diagram(ax, mono=mono)

    # Rodapé interno curto (USP): apenas o número da figura.
    if FOOTER_SHOW:
        ax.text(
            0.5,
            0.015,
            FIG_NUM,
            fontsize=10,
            color="#111827" if mono else "#374151",
            ha="center",
            va="bottom",
        )

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_jpg, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    out_dir = Path(__file__).resolve().parent / "figuras"
    out_dir.mkdir(parents=True, exist_ok=True)

    _render(
        mono=False,
        out_png=out_dir / "MAPEA_modelo.png",
        out_pdf=out_dir / "MAPEA_modelo.pdf",
        out_jpg=out_dir / "MAPEA_modelo.jpg",
    )

    _render(
        mono=True,
        out_png=out_dir / "MAPEA_modelo_mono.png",
        out_pdf=out_dir / "MAPEA_modelo_mono.pdf",
        out_jpg=out_dir / "MAPEA_modelo_mono.jpg",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
