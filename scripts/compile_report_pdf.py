#!/usr/bin/env python3
"""
compile_report_pdf.py - Converts CivicResolve_AI_Final_Report.md into a professional PDF
using marked, KaTeX for mathematical equations, and Google Chrome headless printing.
"""

import subprocess
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "report"
MD_FILE = REPORT_DIR / "CivicResolve_AI_Final_Report.md"
HTML_FILE = REPORT_DIR / "report_print.html"
PDF_FILE = REPORT_DIR / "CivicResolve_AI_Final_Report.pdf"

def main():
    print(f"Reading markdown from: {MD_FILE}")
    if not MD_FILE.exists():
        print(f"Error: {MD_FILE} does not exist!")
        sys.exit(1)

    # 1. Run marked to get HTML body
    cmd_marked = ["npx", "-y", "marked", "-i", str(MD_FILE)]
    result = subprocess.run(cmd_marked, capture_output=True, text=True, cwd=str(REPORT_DIR))
    if result.returncode != 0:
        print(f"Marked error: {result.stderr}")
        sys.exit(1)

    html_body = result.stdout

    # 2. Wrap HTML body with academic print stylesheet and KaTeX math renderer
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CivicResolve AI - Final Academic Project Report</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
<style>
@page {{
    size: A4;
    margin: 14mm 13mm 14mm 13mm;
    @bottom-right {{
        content: counter(page);
    }}
}}

body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Liberation Sans", sans-serif;
    color: #1a202c;
    background-color: #ffffff;
    line-height: 1.48;
    font-size: 9.8pt;
    margin: 0;
    padding: 0;
}}

/* Typography */
h1 {{
    font-size: 23pt;
    font-weight: 800;
    color: #0f172a;
    text-align: center;
    margin-top: 22pt;
    margin-bottom: 11pt;
    letter-spacing: -0.5px;
}}

h2 {{
    font-size: 14.5pt;
    font-weight: 700;
    color: #1e3a8a;
    border-bottom: 2px solid #3b82f6;
    padding-bottom: 4pt;
    margin-top: 22pt;
    margin-bottom: 9pt;
    page-break-after: avoid;
}}

h3 {{
    font-size: 11.8pt;
    font-weight: 600;
    color: #1e293b;
    margin-top: 15pt;
    margin-bottom: 5pt;
    page-break-after: avoid;
}}

h4 {{
    font-size: 10.8pt;
    font-weight: 600;
    color: #334155;
    margin-top: 11pt;
    margin-bottom: 4pt;
    page-break-after: avoid;
}}

p, ul, ol {{
    margin-top: 0;
    margin-bottom: 7pt;
    text-align: justify;
}}

li {{
    margin-bottom: 3pt;
}}

hr {{
    border: 0;
    height: 1px;
    background: #cbd5e1;
    margin: 20pt 0;
}}

/* Blockquotes & Callouts */
blockquote {{
    background: #f8fafc;
    border-left: 4px solid #3b82f6;
    margin: 12pt 0;
    padding: 8pt 14pt;
    font-style: italic;
    color: #334155;
}}

/* Code Blocks */
pre {{
    background: #0f172a;
    color: #f8fafc;
    padding: 10pt 12pt;
    border-radius: 6px;
    font-size: 8.5pt;
    line-height: 1.45;
    overflow-x: auto;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
    margin: 10pt 0;
    page-break-inside: avoid;
}}

code {{
    background: #f1f5f9;
    color: #0f172a;
    padding: 1.5pt 4pt;
    border-radius: 3px;
    font-size: 9pt;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
}}

pre code {{
    background: transparent;
    color: inherit;
    padding: 0;
    border-radius: 0;
}}

/* Tables */
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12pt 0;
    font-size: 9pt;
    page-break-inside: avoid;
}}

th, td {{
    border: 1px solid #cbd5e1;
    padding: 5pt 7pt;
    text-align: left;
    vertical-align: top;
}}

th {{
    background-color: #f1f5f9;
    font-weight: 700;
    color: #0f172a;
}}

tr:nth-child(even) td {{
    background-color: #f8fafc;
}}

/* Images & Figures */
img {{
    max-width: 88%;
    height: auto;
    display: block;
    margin: 8pt auto 2pt auto;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    page-break-inside: avoid;
}}

em {{
    display: block;
    text-align: center;
    font-size: 8.5pt;
    color: #64748b;
    margin-bottom: 12pt;
}}

.katex-display {{
    margin: 12pt 0 !important;
    text-align: center;
}}
</style>
<script>
document.addEventListener("DOMContentLoaded", function() {{
    renderMathInElement(document.body, {{
        delimiters: [
            {{left: '$$', right: '$$', display: true}},
            {{left: '$', right: '$', display: false}}
        ],
        throwOnError: false
    }});
}});
</script>
</head>
<body>
{html_body}
</body>
</html>
"""

    with open(HTML_FILE, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated printable HTML: {HTML_FILE}")

    # 3. Print to PDF via Google Chrome
    cmd_chrome = [
        "google-chrome",
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=5000",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={PDF_FILE}",
        str(HTML_FILE)
    ]
    print("Executing Chrome headless print-to-pdf...")
    res_chrome = subprocess.run(cmd_chrome, capture_output=True, text=True)
    if res_chrome.returncode != 0:
        print(f"Chrome error: {res_chrome.stderr}")
        sys.exit(1)

    if PDF_FILE.exists() and PDF_FILE.stat().st_size > 0:
        print(f"Successfully generated PDF: {PDF_FILE} ({PDF_FILE.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"Failed to generate valid PDF at {PDF_FILE}")
        sys.exit(1)

if __name__ == "__main__":
    main()
