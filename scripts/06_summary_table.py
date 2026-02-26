# -*- coding: utf-8 -*-
"""
06_summary_table.py — Сводная таблица R-фактора для публикации (Markdown + LaTeX).

Результат:
  output/tables/summary_table.md
  output/tables/summary_table.tex
"""
import os
import sys
import csv

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import OUTPUT_DIR, R_UNITS


def load_stats(csv_path):
    """Загрузить annual_stats.csv."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) if k != "year" else int(float(v))
                         for k, v in row.items()})
    return rows


def generate_markdown(rows, out_path):
    """Сгенерировать Markdown-таблицу."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    lines = []
    lines.append(f"# Годовая статистика R-фактора RUSLE ({rows[0]['year']}–{rows[-1]['year']})")
    lines.append("")
    lines.append(f"Единицы измерения: {R_UNITS}")
    lines.append("")
    lines.append("| Год | Min | P5 | P25 | Медиана | Среднее | P75 | P95 | Max | σ | CV, % |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for r in rows:
        lines.append(
            f"| {r['year']:.0f} | {r['min']:.0f} | {r['p5']:.0f} | {r['p25']:.0f} | "
            f"{r['median']:.0f} | {r['mean']:.0f} | {r['p75']:.0f} | {r['p95']:.0f} | "
            f"{r['max']:.0f} | {r['std']:.0f} | {r['cv']:.0f} |"
        )

    # Итоговая строка
    means = [r["mean"] for r in rows]
    overall_mean = np.mean(means)
    overall_std = np.std(means)
    lines.append(
        f"| **Среднее** | {np.mean([r['min'] for r in rows]):.0f} | "
        f"{np.mean([r['p5'] for r in rows]):.0f} | "
        f"{np.mean([r['p25'] for r in rows]):.0f} | "
        f"{np.mean([r['median'] for r in rows]):.0f} | "
        f"**{overall_mean:.0f}** | "
        f"{np.mean([r['p75'] for r in rows]):.0f} | "
        f"{np.mean([r['p95'] for r in rows]):.0f} | "
        f"{np.mean([r['max'] for r in rows]):.0f} | "
        f"{overall_std:.0f} | "
        f"{overall_std/overall_mean*100:.0f} |"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Markdown: {out_path}")


def generate_latex(rows, out_path):
    """Сгенерировать LaTeX-таблицу."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Годовая статистика R-фактора RUSLE по данным IMERG V07 (калиброванный), "
                 f"{rows[0]['year']}--{rows[-1]['year']}" + r"}")
    lines.append(r"\label{tab:rfactor_annual}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{c r r r r r r r r r r}")
    lines.append(r"\hline")
    lines.append(r"Год & Min & P5 & P25 & Мед. & Сред. & P75 & P95 & Max & $\sigma$ & CV,\% \\")
    lines.append(r"\hline")

    for r in rows:
        lines.append(
            f"{r['year']:.0f} & {r['min']:.0f} & {r['p5']:.0f} & {r['p25']:.0f} & "
            f"{r['median']:.0f} & {r['mean']:.0f} & {r['p75']:.0f} & {r['p95']:.0f} & "
            f"{r['max']:.0f} & {r['std']:.0f} & {r['cv']:.0f} \\\\"
        )

    lines.append(r"\hline")
    means = [r["mean"] for r in rows]
    overall_mean = np.mean(means)
    overall_std = np.std(means)
    lines.append(
        r"\textbf{Сред.} & "
        f"{np.mean([r['min'] for r in rows]):.0f} & "
        f"{np.mean([r['p5'] for r in rows]):.0f} & "
        f"{np.mean([r['p25'] for r in rows]):.0f} & "
        f"{np.mean([r['median'] for r in rows]):.0f} & "
        f"\\textbf{{{overall_mean:.0f}}} & "
        f"{np.mean([r['p75'] for r in rows]):.0f} & "
        f"{np.mean([r['p95'] for r in rows]):.0f} & "
        f"{np.mean([r['max'] for r in rows]):.0f} & "
        f"{overall_std:.0f} & "
        f"{overall_std/overall_mean*100:.0f} \\\\"
    )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  LaTeX: {out_path}")


def main():
    print("=" * 60)
    print("Сводная таблица R-фактора для публикации")
    print("=" * 60)

    csv_path = os.path.join(OUTPUT_DIR, "tables", "annual_stats.csv")
    if not os.path.exists(csv_path):
        print(f"  Ошибка: не найден {csv_path}")
        print("  Сначала запустите: python scripts/01_compute_stats.py")
        return

    rows = load_stats(csv_path)

    generate_markdown(rows, os.path.join(OUTPUT_DIR, "tables", "summary_table.md"))
    generate_latex(rows, os.path.join(OUTPUT_DIR, "tables", "summary_table.tex"))


if __name__ == "__main__":
    main()
