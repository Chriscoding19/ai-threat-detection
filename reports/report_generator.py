"""
report_generator.py
--------------------
Generates a professional HTML report of the threat detection analysis.
"""

import os
import json
from datetime import datetime


def generate_html_report(results: list, label_names: list,
                          dataset_stats: dict, output_dir: str = "reports") -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = os.path.join(output_dir, f"threat_detection_report_{timestamp}.html")

    best = max((r for r in results if "accuracy" in r), key=lambda r: r["accuracy"])

    per_class_html = ""
    if "classification_report" in best:
        for cls in label_names:
            if cls in best["classification_report"]:
                m = best["classification_report"][cls]
                per_class_html += f"""
                <tr>
                    <td>{cls}</td>
                    <td>{m['precision']:.3f}</td>
                    <td>{m['recall']:.3f}</td>
                    <td>{m['f1-score']:.3f}</td>
                    <td>{int(m['support'])}</td>
                </tr>"""

    models_html = ""
    for r in results:
        if "accuracy" in r:
            f1 = r.get("f1_weighted", r.get("f1_binary", 0))
            models_html += f"""
            <tr>
                <td><strong>{r['model']}</strong></td>
                <td>{r['accuracy']:.4f} ({r['accuracy']*100:.1f}%)</td>
                <td>{f1:.4f}</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI Threat Detection Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0d1117; color: #c9d1d9; line-height: 1.6; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 2rem; }}
  h1 {{ color: #58a6ff; font-size: 2rem; margin-bottom: 0.3rem; }}
  h2 {{ color: #79c0ff; font-size: 1.2rem; margin: 2rem 0 1rem; border-bottom: 1px solid #30363d; padding-bottom: 0.5rem; }}
  .meta {{ color: #8b949e; margin-bottom: 2rem; font-size: 0.9rem; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
  .summary-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1.2rem; text-align: center; }}
  .summary-card .number {{ font-size: 2rem; font-weight: bold; }}
  .summary-card .label  {{ font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }}
  table {{ width: 100%; border-collapse: collapse; background: #161b22; border-radius: 8px; overflow: hidden; border: 1px solid #30363d; margin-bottom: 2rem; }}
  th {{ background: #21262d; color: #8b949e; text-align: left; padding: 0.75rem 1rem; font-size: 0.8rem; text-transform: uppercase; }}
  td {{ padding: 0.75rem 1rem; border-top: 1px solid #21262d; font-family: monospace; font-size: 0.9rem; }}
  .figures {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }}
  .figure {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }}
  .figure img {{ width: 100%; display: block; }}
  .figure-caption {{ padding: 0.5rem 1rem; font-size: 0.8rem; color: #8b949e; text-align: center; }}
  .attack-badge {{ display: inline-block; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: bold; }}
  .badge-normal {{ background: #0d2a14; color: #3fb950; }}
  .badge-dos    {{ background: #2a0d0d; color: #f85149; }}
  .badge-scan   {{ background: #2a1d0d; color: #d29922; }}
  .badge-exfil  {{ background: #0d1a2a; color: #388bfd; }}
  .badge-brute  {{ background: #1a0d2a; color: #bc8cff; }}
  .highlight {{ color: #3fb950; font-weight: bold; }}
</style>
</head>
<body>
<div class="container">
  <h1>🧠 AI-Based Threat Detection</h1>
  <div class="meta">
    Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} &nbsp;|&nbsp;
    Dataset: {dataset_stats.get('total', 0):,} traffic records &nbsp;|&nbsp;
    Models: Random Forest, Gradient Boosting, Isolation Forest, Ensemble
  </div>

  <h2>Dataset Overview</h2>
  <div class="summary-grid">
    <div class="summary-card"><div class="number" style="color:#58a6ff">{dataset_stats.get('total',0):,}</div><div class="label">Total Records</div></div>
    <div class="summary-card"><div class="number" style="color:#3fb950">{dataset_stats.get('normal',0):,}</div><div class="label">Normal Traffic</div></div>
    <div class="summary-card"><div class="number" style="color:#f85149">{dataset_stats.get('attacks',0):,}</div><div class="label">Attack Records</div></div>
    <div class="summary-card"><div class="number" style="color:#d29922">{len(label_names)}</div><div class="label">Classes</div></div>
    <div class="summary-card"><div class="number highlight">{best['accuracy']*100:.1f}%</div><div class="label">Best Accuracy</div></div>
  </div>

  <h2>Attack Classes Detected</h2>
  <p style="margin-bottom:1rem; color:#8b949e">The system classifies traffic into 5 categories using ML models trained on {dataset_stats.get('total',0):,} labelled records.</p>
  <table>
    <thead><tr><th>Class</th><th>Description</th><th>Key Indicators</th></tr></thead>
    <tbody>
      <tr><td><span class="attack-badge badge-normal">NORMAL</span></td><td>Legitimate network traffic</td><td>Normal packet sizes, business hours, low connection rate</td></tr>
      <tr><td><span class="attack-badge badge-dos">DoS</span></td><td>Denial of Service flood attack</td><td>conn_rate &gt; 100, tiny packets, massive packet count</td></tr>
      <tr><td><span class="attack-badge badge-scan">Port Scan</span></td><td>Network reconnaissance scan</td><td>unique_ports &gt; 100, short duration, SYN-only flags</td></tr>
      <tr><td><span class="attack-badge badge-exfil">Data Exfiltration</span></td><td>Unauthorized data theft</td><td>bytes_sent &gt; 100KB, off-hours, high bytes_ratio</td></tr>
      <tr><td><span class="attack-badge badge-brute">Brute Force</span></td><td>Credential stuffing / password guessing</td><td>failed_logins &gt; 10, same port, rapid connections</td></tr>
    </tbody>
  </table>

  <h2>Model Performance</h2>
  <table>
    <thead><tr><th>Model</th><th>Accuracy</th><th>F1 Score</th></tr></thead>
    <tbody>{models_html}</tbody>
  </table>

  <h2>Best Model — Per-Class Metrics ({best['model']})</h2>
  <table>
    <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
    <tbody>{per_class_html}</tbody>
  </table>

  <h2>Visualisations</h2>
  <div class="figures">
    <div class="figure">
      <img src="figures/class_distribution.png" alt="Class Distribution">
      <div class="figure-caption">Dataset class distribution</div>
    </div>
    <div class="figure">
      <img src="figures/model_comparison.png" alt="Model Comparison">
      <div class="figure-caption">Accuracy and F1 score by model</div>
    </div>
    <div class="figure">
      <img src="figures/feature_importance.png" alt="Feature Importance">
      <div class="figure-caption">Top features — Random Forest</div>
    </div>
    <div class="figure">
      <img src="figures/anomaly_scores.png" alt="Anomaly Scores">
      <div class="figure-caption">Isolation Forest anomaly score distribution</div>
    </div>
  </div>
  <div class="figures">
    <div class="figure">
      <img src="figures/confusion_Random_Forest.png" alt="RF Confusion Matrix">
      <div class="figure-caption">Random Forest confusion matrix</div>
    </div>
    <div class="figure">
      <img src="figures/confusion_Ensemble_RF__GB__IsoForest.png" alt="Ensemble Confusion Matrix">
      <div class="figure-caption">Ensemble confusion matrix</div>
    </div>
  </div>
</div>
</body>
</html>"""

    with open(filename, "w") as f:
        f.write(html)
    print(f"  HTML report saved: {filename}")
    return filename
