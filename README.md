# ◈ AntiBot Lab

**Signal over suspicion.** A visually rich Streamlit workbench for advertising traffic research, anomalous publisher/traffic-source discovery, and controlled synthetic bot experiments.

## Launch

```bash
pip install -r requirements.txt
streamlit run app.py
```

### GitHub Codespaces

Open **Code → Codespaces → Create codespace on main**, allow the devcontainer to install dependencies, and run `streamlit run app.py --server.address 0.0.0.0 --server.port 8501`. Open forwarded port 8501 (private by default).

## Seven workspaces

- **Command center** — traffic overview, source concentration and key performance indicators.
- **Traffic explorer** — event browser, click rates and campaign analysis.
- **Anomaly studio** — Isolation Forest, Local Outlier Factor, One-Class SVM, empirical-CDF tail scores, IP graph-overlap heuristics.
- **Graph intelligence** — source and encoded-IP relationships.
- **Model arena** — source-disjoint model benchmarking; Logistic Regression, Random Forest, HistGradientBoosting plus unsupervised detectors.
- **Red-team simulator** — synthetic click storms, device rotation and low-and-slow activity with known injected labels.
- **Research notes** — provenance, ground-truth policy and limitations.

## Data provenance matters

The default demo is entirely synthetic; labels there only identify simulated bots. The processed iPinYou x1 RTB benchmark is downloadable on demand for **non-commercial** research; it has **no verified human/bot labels**. A publisher/domain is a traffic-source proxy, not confirmed publisher account ownership. For real unlabeled traffic, the dashboard reports anomaly rankings, not proven fraud or fabricated PR-AUC metrics. Optional uploaded CSVs may carry user-supplied labels, explicitly identified as unaudited.

The iPinYou processed benchmark does not preserve reliable wall-clock event timestamps. Don't interpret encoded hour categories as hours or claim temporal burst identification from them. The graph-overlap score is a heuristic, **not a graph neural network**. The synthetic simulator is for controlled comparisons, not evidence of actual fraud rates.

## Testing

```bash
python -m pytest -q
python -m compileall -q app.py fraudlab
```

This repository includes a Codespaces devcontainer, GitHub Actions tests and MIT-licensed source code. Third-party dataset licenses are separate.

Data references: https://huggingface.co/datasets/reczoo/iPinYou_x1 · https://contest.ipinyou.com/
