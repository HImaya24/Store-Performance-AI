from __future__ import annotations
import os, io, base64, datetime as dt
from typing import List, Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader, select_autoescape
from fpdf import FPDF

from ..utils.security import sanitize_text, sanitize_recommendations
from ..utils.rai import bias_flags, explainability_context
from ..utils.kpi import format_kpis
from ..config import OUTPUT_DIR

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _trend_plot_png(trends: List[Dict[str, Any]]) -> bytes:
    if not trends:
        return b""
    df = pd.DataFrame(trends)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    plt.figure()
    plt.plot(df['date'], df['sales'], label='Sales')
    plt.title('Sales Trend')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf.read()

def _save_png(path: str, png_bytes: bytes) -> Optional[str]:
    if not png_bytes:
        return None
    with open(path, 'wb') as f:
        f.write(png_bytes)
    return path

def generate_reports(*, store_id: str, kpis: Dict[str, Any], trends: List[Dict[str, Any]], anomalies: List[Dict[str, Any]], insights_text: str, recommendations: List[str], formats: List[str]) -> List[str]:
    ts = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join(OUTPUT_DIR, f"{store_id}_{ts}")
    _ensure_dir(outdir)

    # Plot
    png = _trend_plot_png(trends)
    chart_path = _save_png(os.path.join(outdir, 'trend.png'), png)

    # Sanitize & RAI
    insights_text = sanitize_text(insights_text)
    recommendations = sanitize_recommendations(recommendations)
    flags = bias_flags(insights_text + ' ' + ' '.join(recommendations))
    explanations = explainability_context(kpis, recommendations)
    kpis_fmt = format_kpis(kpis)

    files = []
    if 'html' in formats:
        files.append(_render_html(outdir, store_id, kpis_fmt, anomalies, insights_text, recommendations, explanations, chart_path))
    if 'md' in formats:
        files.append(_render_md(outdir, store_id, kpis_fmt, anomalies, insights_text, recommendations, explanations))
    if 'pdf' in formats:
        files.append(_render_pdf(outdir, store_id, kpis_fmt, anomalies, insights_text, recommendations, explanations, flags, chart_path))

    return files

def _render_html(outdir, store_id, kpis_fmt, anomalies, insights_text, recommendations, explanations, chart_path):
    env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), 'templates')), autoescape=select_autoescape())
    tpl = env.get_template('report.html')
    html = tpl.render(
        store_id=store_id,
        generated_at=dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        executive_summary=insights_text,
        kpis_fmt=kpis_fmt,
        anomalies=anomalies or [],
        recommendations=recommendations or [],
        explanations=explanations or [],
        chart_src=f"trend.png" if chart_path else None
    )
    out = os.path.join(outdir, 'report.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    # copy css
    css_src = os.path.join(os.path.dirname(__file__), 'templates', 'styles.css')
    css_dst = os.path.join(outdir, 'styles.css')
    if os.path.exists(css_src):
        with open(css_src, 'r', encoding='utf-8') as s, open(css_dst, 'w', encoding='utf-8') as d:
            d.write(s.read())
    return out

def _render_md(outdir, store_id, kpis_fmt, anomalies, insights_text, recommendations, explanations):
    out = os.path.join(outdir, 'report.md')
    lines = [f"# Store Performance Report — {store_id}", '', f"Generated: {dt.datetime.now():%Y-%m-%d %H:%M:%S}", '', '## Executive Summary', insights_text or 'N/A', '', '## KPIs']
    for k, v in kpis_fmt.items():
        lines.append(f"- **{k}**: {v}")
    if anomalies:
        lines += ['', '## Anomalies']
        for a in anomalies:
            lines.append(f"- {a.get('date')} — {a.get('metric')} ({a.get('severity')}): {a.get('note','')}")
    if recommendations:
        lines += ['', '## Recommendations']
        for i, r in enumerate(recommendations):
            why = explanations[i] if i < len(explanations) else ''
            lines.append(f"- {r}  \\n  _Why_: {why}")
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return out

class _PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Store Performance Report', ln=True, align='C')

def _render_pdf(outdir, store_id, kpis_fmt, anomalies, insights_text, recommendations, explanations, bias_flags_list, chart_path):
    pdf = _PDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, f'Store: {store_id}', ln=True)
    pdf.cell(0, 8, f'Generated: {dt.datetime.now():%Y-%m-%d %H:%M:%S}', ln=True)
    pdf.ln(4)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Executive Summary', ln=True)
    pdf.set_font('Arial', '', 11)
    for line in (insights_text or '').split('\n'):
        pdf.multi_cell(0, 6, line)
    pdf.ln(2)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'KPIs', ln=True)
    pdf.set_font('Arial', '', 11)
    for k, v in kpis_fmt.items():
        pdf.cell(0, 6, f"{k}: {v}", ln=True)
    pdf.ln(2)

    if anomalies:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Anomalies', ln=True)
        pdf.set_font('Arial', '', 11)
        for a in anomalies:
            pdf.multi_cell(0, 6, f"- {a.get('date')} — {a.get('metric')} ({a.get('severity')}): {a.get('note','')}")
        pdf.ln(2)

    if chart_path and os.path.exists(chart_path):
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Sales Trend', ln=True)
        pdf.image(chart_path, w=170)
        pdf.ln(2)

    if recommendations:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Recommendations', ln=True)
        pdf.set_font('Arial', '', 11)
        for i, r in enumerate(recommendations):
            why = explanations[i] if i < len(explanations) else ''
            pdf.multi_cell(0, 6, f"- {r}\n  Why: {why}")
    if bias_flags_list:
        pdf.ln(2)
        pdf.set_font('Arial', 'I', 10)
        pdf.multi_cell(0, 6, 'Responsible AI: Possible bias phrases detected — ' + ', '.join(bias_flags_list))

    out = os.path.join(outdir, 'report.pdf')
    pdf.output(out)
    return out
