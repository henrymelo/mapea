# MAPEA — geração de gráficos para o Modelo MAPEA do Paper - USP‑ESALQ

**Paper:** MAPEA: Modelo de Análise Preditiva de Estímulos de Aprendizagem

**Autor:** Henrique Renaldo de Melo

**Orientador:** Ricardo Janes

Este diretório contém os scripts Python usados para gerar as figuras do Capítulo 5 do Paper do programa de pós-graduação (USP‑ESALQ) — o estudo MAPEA (Modelo APEA: Análise Preditiva de Estímulo de Aprendizagem). O objetivo é reproduzir os gráficos analíticos e o diagrama do modelo MAPEA (análises estatísticas, comparações por tipo de intervenção e visualização de persistência longitudinal).

## Modelo MAPEA

![Modelo MAPEA](paper/mapEA_modelo.png)

**Paper MAPEA (PDF):** [TCC Revisado — Henrique Renaldo de Melo](paper/%5BTCC%20Revisado%5D%20-%20Henrique%20Renaldo%20de%20Melo.pdf)

Resumo do propósito
- MAPEA é um estudo desenvolvido no contexto da pós-graduação em Data Science e Analytics 2025 da USP‑ESALQ.
- O pipeline aqui é leve: lê uma base tabular (CSV), produz visualizações (PNG/PDF) em `figuras/` e gera um diagrama conceitual do modelo MAPEA.
- Os scripts são pensados para serem reprodutíveis localmente — o repositório que você publicar pode conter apenas os scripts e as bases necessárias (ou amostras anônimas das mesmas).

Arquivos principais
- `gerar_grafico_correlacao.py` — script principal que gera os gráficos de correlação e o gráfico longitudinal (fig01b_correlacao_por_tipo_intervencao_longitudinal). Procura os CSVs na seguinte ordem:
  1. `MAPEA_INPUT` (variável de ambiente) — caminho para um CSV (absoluto ou relativo ao diretório do script)
  2`fontes_extraidas.csv` (recomendado)
  3`dados_MAPEA.csv` (fallback)

- `gerar_grafico_MAPEA_modelo.py` — gera o diagrama conceitual MAPEA em cores e versão monocromática e salva em `figuras/`.
- `requirements.txt` — dependências necessárias (pandas, numpy, matplotlib, seaborn, scipy, etc.).

Dados que acompanham o script (o que subir)
- Ideal: `fontes_extraidas.csv` (base primária usada para as figuras). Se disponível, inclua essa base (ou uma amostra anônima) no repositório.
- Alternativa/fallback: `dados_MAPEA.csv`.
- Opcional: `dados_oficiais_brasil.csv` (usado para bases observacionais auxiliares).

Arquivos auxiliares e templates (incluídos no diretório)
- `fontes_extraidas_template.csv` — template para a base `fontes_extraidas.csv` (ex.: colunas e formato esperados).
- `dados_MAPEA_template.csv` / `dados_MAPEA_exemplo.csv` — templates/exemplos para `dados_MAPEA.csv`.
- `dados_oficiais_brasil_template.csv` — template para `dados_oficiais_brasil.csv`.

Saídas geradas pelos scripts (em `figuras/`)
- Figuras: `fig01_correlacao_jointplot.*`, `fig01b_correlacao_por_tipo_intervencao.*`, `fig01b_correlacao_por_tipo_intervencao_longitudinal_exploratorio.*`, `fig03_heatmap_correlacao_numerica.*`, `figA_efeitos_padronizados_por_estudo.*`, `MAPEA_modelo.*`, etc.
- Relatórios auxiliares: `figuras/pendencias_rastreabilidade.csv` (linhas sem DOI/URL), `figuras/correlacao_resumo.txt`.

Instalação e execução (exemplo rápido)
1. Criar ambiente virtual e instalar dependências:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Executar os scripts (saídas vão para `figuras/`):

```bash
# usa a CSV padrão (fontes_extraidas.csv ou dados_MAPEA.csv conforme presença)
python3 gerar_grafico_correlacao.py
python3 gerar_grafico_MAPEA_modelo.py
```

3. Forçar um CSV específico (por exemplo, se você tem a base no Downloads):

```bash
export MAPEA_INPUT="fontes_extraidas.csv"
python3 gerar_grafico_correlacao.py
```

Variáveis de ambiente úteis
- `MAPEA_INPUT` — caminho para o CSV de entrada (relativo ao script ou absoluto).
- `MAPEA_BOOTSTRAP=1` — força a geração exploratória sintética quando não há dados reais suficientes (útil para ilustrações).
- `MAPEA_STRICT_CITATIONS=0` — evita que o script interrompa a execução por falta de DOI/URL.
- `MAPEA_FIG_SHOW_SOURCE=1` — inclui a fonte no rodapé das figuras.
- `MAPEA_VERBOSE=1` — modo verboso (logs simples).

Observações e boas práticas
- Preferência por não comitar a pasta `figuras/` gerada; ela é reprodutível localmente. Comite apenas scripts, `requirements.txt` e os CSVs (ou amostras anonimizadas).
- Se o CSV original contém dados pessoais ou sensíveis, anônimize antes de subir ao GitHub.


```
# Artefatos
figuras/
.venv/
__pycache__/
*.pyc
```

Licença e créditos
- Código e figuras geradas são parte do trabalho do autor (Paper, USP‑ESALQ).
- Mantenha a referência ao autor.
- Desenvolvido por Henrique Renaldo de Melo, 2025.
- Licença: MIT License (veja o arquivo LICENSE para detalhes).
- GitHub: https://github.com/henrymelo/mapea
- LinkedIn: https://www.linkedin.com/in/hrmelo