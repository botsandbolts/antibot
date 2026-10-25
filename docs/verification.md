# AntiBot Lab runtime verification

We test **synthetic traffic generation**, per-source feature engineering, anomaly scores, held-out benchmarking, and an **offline encoded-schema ZIP fixture** matching the published iPinYou x1 columns. The ZIP fixture verifies the loader and label-provenance policy, but it is not the 252 MB real archive.

The real dataset must be downloaded interactively from the dashboard. Real events have unknown bot identity; artificial attacks are explicitly marked `is_injected`.

Codespaces automatically starts Streamlit on port 8501 via `scripts/launch.sh`. Open the private forwarded port to view the dashboard. The deployment is not publicly hosted by default.

A complete live smoke test includes: launch app, inspect synthetic Command Center and Model Arena, download iPinYou x1, verify loaded rows/encoded columns, then evaluate anomaly rankings **without fabricated fraud metrics**.
