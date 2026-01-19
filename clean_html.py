#!/usr/bin/env python3
"""
Clean MSO (Microsoft Office) cruft from Word-exported HTML.
Preserves content and basic structure, strips Office-specific markup.
"""

import re
import sys
from pathlib import Path

def clean_mso_html(html_content):
    """Remove Microsoft Office specific markup from HTML."""

    # Remove XML namespaces from html tag
    html_content = re.sub(r'<html[^>]*>', '<html lang="en">', html_content, flags=re.IGNORECASE)

    # IMPORTANT: Keep content inside <![if !vml]>...<![endif]> blocks (non-VML fallbacks with img tags)
    # Extract the content, remove the conditional wrapper
    html_content = re.sub(r'<!\[if !vml\]>(.*?)<!\[endif\]>', r'\1', html_content, flags=re.DOTALL | re.IGNORECASE)

    # Remove VML conditional blocks (Office vector markup we don't need)
    html_content = re.sub(r'<!\[if gte vml 1\]>.*?<!\[endif\]>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r'<!--\[if gte vml 1\]>.*?<!\[endif\]-->', '', html_content, flags=re.DOTALL | re.IGNORECASE)

    # Remove other conditional comments (mso, IE specific, etc.)
    html_content = re.sub(r'<!\[if[^\]]*\]>.*?<!\[endif\]>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r'<!--\[if[^\]]*\]>.*?<!\[endif\]-->', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r'<!--\[if[^\]]*\]>', '', html_content, flags=re.IGNORECASE)
    html_content = re.sub(r'<!\[endif\]-->', '', html_content, flags=re.IGNORECASE)
    html_content = re.sub(r'<!\[endif\]>', '', html_content, flags=re.IGNORECASE)

    # Remove Office-specific XML tags
    html_content = re.sub(r'<o:p>.*?</o:p>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r'<o:[^>]*/?>', '', html_content, flags=re.IGNORECASE)
    html_content = re.sub(r'</o:[^>]*>', '', html_content, flags=re.IGNORECASE)
    html_content = re.sub(r'<w:[^>]*/?>', '', html_content, flags=re.IGNORECASE)
    html_content = re.sub(r'</w:[^>]*>', '', html_content, flags=re.IGNORECASE)
    html_content = re.sub(r'<m:[^>]*/?>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r'</m:[^>]*>', '', html_content, flags=re.IGNORECASE)
    html_content = re.sub(r'<v:[^>]*>.*?</v:[^>]*>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r'<v:[^>]*/>', '', html_content, flags=re.IGNORECASE)

    # Remove mso-* style properties (keep other styles)
    def clean_style(match):
        style = match.group(1)
        # Split by semicolon, filter out mso-* properties
        props = [p.strip() for p in style.split(';') if p.strip()]
        clean_props = [p for p in props if not p.strip().startswith('mso-')]
        if clean_props:
            return f"style='{'; '.join(clean_props)}'"
        return ''

    html_content = re.sub(r"style='([^']*)'", clean_style, html_content)
    html_content = re.sub(r'style="([^"]*)"', lambda m: clean_style(type('obj', (object,), {'group': lambda self, n: m.group(n).replace('"', "'")})()) , html_content)

    # Remove empty style attributes
    html_content = re.sub(r"\s*style=''", '', html_content)
    html_content = re.sub(r'\s*style=""', '', html_content)

    # Remove class="Mso*" classes but keep other classes
    def clean_class(match):
        classes = match.group(1).split()
        clean_classes = [c for c in classes if not c.startswith('Mso')]
        if clean_classes:
            return f"class='{' '.join(clean_classes)}'"
        return ''

    html_content = re.sub(r"class='([^']*)'", clean_class, html_content)
    html_content = re.sub(r'class="([^"]*)"', lambda m: clean_class(type('obj', (object,), {'group': lambda self, n: m.group(n)})()) , html_content)

    # Remove empty class attributes
    html_content = re.sub(r"\s*class=''", '', html_content)
    html_content = re.sub(r'\s*class=""', '', html_content)

    # Remove lang attributes (not needed, clutters markup)
    # Handle unquoted values like lang=EN-US (note: \w+ doesn't match hyphen)
    html_content = re.sub(r'\s*lang=[\w-]+', '', html_content)
    html_content = re.sub(r"\s*lang='[^']*'", '', html_content)
    html_content = re.sub(r'\s*lang="[^"]*"', '', html_content)

    # Remove Word-specific meta tags
    html_content = re.sub(r'<meta[^>]*(ProgId|Generator|Originator)[^>]*>', '', html_content, flags=re.IGNORECASE)

    # Remove Word-specific link tags
    html_content = re.sub(r'<link[^>]*(File-List|Edit-Time-Data|themeData|colorSchemeMapping)[^>]*>', '', html_content, flags=re.IGNORECASE)

    # Clean up empty spans (spans with no attributes and only whitespace or nothing)
    for _ in range(5):  # Multiple passes to catch nested empty spans
        html_content = re.sub(r'<span>(\s*)</span>', r'\1', html_content)
        html_content = re.sub(r'<span>\s*<span>', '<span>', html_content)
        html_content = re.sub(r'</span>\s*</span>', '</span>', html_content)

    # Remove data-theme attribute from html tag if present
    html_content = re.sub(r'\s*data-theme=\w+', '', html_content)

    # Clean up multiple consecutive whitespace/newlines (but preserve some structure)
    html_content = re.sub(r'\n\s*\n\s*\n', '\n\n', html_content)

    # Remove empty paragraphs
    html_content = re.sub(r'<p[^>]*>\s*</p>', '', html_content)

    # Add proper doctype and clean head
    if '<!DOCTYPE' not in html_content.upper():
        html_content = '<!DOCTYPE html>\n' + html_content

    return html_content


def add_clean_styles(html_content):
    """Add clean CSS styles to replace removed MSO styles."""

    clean_css = """
<style>
/* Clean styles to replace MSO formatting */
body {
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 900px;
    margin: 0 auto;
    padding: 20px 40px;
}

h1, h2, h3, h4 {
    color: #000;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
}

h1 { font-size: 1.8em; }
h2 { font-size: 1.5em; }
h3 { font-size: 1.25em; }

p {
    margin: 1em 0;
    text-align: justify;
}

ul, ol {
    margin: 1em 0;
    padding-left: 2em;
}

li {
    margin: 0.5em 0;
}

table {
    border-collapse: collapse;
    margin: 1em 0;
}

td, th {
    padding: 8px 12px;
    vertical-align: top;
}

figure {
    margin: 1.5em 0;
    text-align: center;
}

figcaption {
    font-size: 0.9em;
    color: #666;
    margin-top: 0.5em;
}

img {
    max-width: 100%;
    height: auto;
}

a {
    color: #0066cc;
}

blockquote, .executive-summary {
    background: #f8f9fa;
    border-left: 3px solid #666;
    padding: 10px 15px;
    margin: 1.5em 0;
}

strong, b {
    font-weight: 600;
}

code {
    background: #f4f4f4;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: monospace;
}

@media (max-width: 768px) {
    body {
        padding: 15px 20px;
    }
}
</style>
"""

    # Insert clean CSS after <head> tag
    html_content = re.sub(r'(<head[^>]*>)', r'\1\n' + clean_css, html_content, flags=re.IGNORECASE)

    return html_content


def main():
    if len(sys.argv) < 2:
        print("Usage: python clean_html.py input.html [output.html]")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else input_file

    print(f"Reading {input_file}...")
    html_content = input_file.read_text(encoding='utf-16')

    original_size = len(html_content)
    print(f"Original size: {original_size:,} characters")

    print("Cleaning MSO markup...")
    html_content = clean_mso_html(html_content)

    print("Adding clean styles...")
    html_content = add_clean_styles(html_content)

    cleaned_size = len(html_content)
    reduction = (1 - cleaned_size / original_size) * 100
    print(f"Cleaned size: {cleaned_size:,} characters ({reduction:.1f}% reduction)")

    # Count images to verify
    img_count = len(re.findall(r'<img', html_content, re.IGNORECASE))
    print(f"Images preserved: {img_count}")

    print(f"Writing {output_file}...")
    output_file.write_text(html_content, encoding='utf-8')

    print("Done!")


if __name__ == '__main__':
    main()
