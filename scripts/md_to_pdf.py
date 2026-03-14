import os
import sys
import base64
import re
import subprocess
try:
    import markdown
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "markdown"])
    import markdown

md_file = r"D:\Artur\Yandex.Disk\РАЗРАБОТКА\code\rfactor-analysis\docs\analysis.md"
html_file = r"D:\Artur\Yandex.Disk\РАЗРАБОТКА\code\rfactor-analysis\docs\analysis.html"
pdf_file = r"D:\Artur\Yandex.Disk\РАЗРАБОТКА\code\rfactor-analysis\docs\analysis.pdf"

print("Reading markdown...")
with open(md_file, "r", encoding="utf-8") as f:
    text = f.read()

docs_dir = os.path.dirname(md_file)

# Inject base64 directly into markdown before compiling to HTML
def get_base64_image(match):
    path = match.group(2)
    if path.startswith("http") or path.startswith("data:"):
        return match.group(0)
    
    abs_path = os.path.join(docs_dir, path)
    if os.path.exists(abs_path):
        with open(abs_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
        ext = os.path.splitext(abs_path)[1].lower()
        mime = "image/png"
        if ext in [".jpg", ".jpeg"]: mime = "image/jpeg"
        alt_text = match.group(1)
        # Make a direct HTML tag to be 100% sure markdown parser handles data-uri
        return f'<img src="data:{mime};base64,{encoded_string}" alt="{alt_text}" />'
    else:
        print(f"Warning: Image not found {abs_path}")
        return match.group(0)

text = re.sub(r'!\[(.*?)\]\((.*?)\)', get_base64_image, text)

print("Converting to HTML...")
html = markdown.markdown(text, extensions=['tables', 'fenced_code'])

css = """
@page {
    size: A4;
    margin: 20mm;
}
body {
    font-family: 'Times New Roman', Times, serif;
    font-size: 14pt;
    line-height: 1.5;
    color: #000;
}
h1, h2, h3, h4 {
    font-family: 'Arial', sans-serif;
    color: #111;
    page-break-after: avoid;
    margin-top: 1.5em;
}
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 20px auto;
    page-break-inside: avoid;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
    page-break-inside: avoid;
    font-size: 11pt;
}
th, td {
    border: 1px solid #000;
    padding: 8px;
    text-align: left;
}
th { background-color: #e5e5e5; }
code {
    background-color: #f6f8fa;
    padding: 2px 4px;
    font-family: Consolas, monospace;
    font-size: 11pt;
    border: 1px solid #eaeaea;
    border-radius: 3px;
}
pre {
    background-color: #f6f8fa;
    padding: 10px;
    border: 1px solid #eaeaea;
    border-radius: 4px;
    page-break-inside: avoid;
    overflow-x: auto;
}
pre code { border: none; padding: 0; }
"""

full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>{css}</style>
<script>
  window.MathJax = {{
    tex: {{
      inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
      displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
    }}
  }};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body>
{html}
</body>
</html>"""

with open(html_file, "w", encoding="utf-8") as f:
    f.write(full_html)

print("Printing to PDF using Edge browser engine...")
edge_paths = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
]

edge_exe = next((p for p in edge_paths if os.path.exists(p)), None)

if edge_exe:
    html_url = "file:///" + html_file.replace("\\", "/")
    cmd = [
        edge_exe, 
        "--headless", 
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--virtual-time-budget=5000",
        f"--print-to-pdf={pdf_file}", 
        html_url
    ]
    subprocess.run(cmd, timeout=30)
    if os.path.exists(pdf_file):
        print(f"Successfully generated high-quality PDF at: {pdf_file}")
    else:
        print("Failed to generate PDF.")
else:
    print("Microsoft Edge not found. Cannot convert HTML to PDF.")
