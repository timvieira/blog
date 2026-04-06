#!/usr/bin/env python3
"""
Minimal static blog builder.

Reads markdown (.md) files from content/, renders them to HTML,
handles Jupyter notebook embedding, and generates an archive index.

Dependencies: markdown, jinja2, nbconvert, pygments
"""

import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import jinja2
import markdown
from pygments.formatters import HtmlFormatter

CONTENT_DIR = Path("content")
OUTPUT_DIR = Path("output/blog")
TEMPLATE_DIR = Path(".")
STATIC_DIRS = ["images", "figures", "downloads", "css"]
SITE_NAME = "Graduate Descent"
EXTRA_TEMPLATE_VARS = {}  # satellite posts can set e.g. {"css_url": "/blog/css/blog.css"}
SATELLITE = False  # when True, single published post becomes root index (no archive page)

def slugify(title):
    """Convert a title to a URL slug, matching Pelican's default behavior."""
    import unicodedata
    slug = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    slug = slug.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)     # remove non-word chars
    slug = re.sub(r"[-\s]+", "-", slug)       # collapse whitespace/hyphens
    slug = slug.strip("-")
    return slug
AUTHOR = "Tim Vieira"
# Absolute URL only used in Atom feed (required by spec)
FEED_URL = "https://timvieira.github.io/blog"

# Matches the pelican-style metadata header lines (key: value)
META_RE = re.compile(r"^([A-Za-z_-]+)\s*:\s*(.+)$")

# Matches {% notebook path.ipynb cells[start:end] %}
NOTEBOOK_RE = re.compile(
    r"\{%\s*notebook\s+(\S+)\s+cells\[(\d*):(\d*)\]\s*%\}"
)


def parse_post(filepath):
    """Parse a markdown file into metadata dict and body string."""
    meta = {}
    lines = filepath.read_text(encoding="utf-8").splitlines()

    # Parse header lines (key: value) until first blank line
    body_start = 0
    for i, line in enumerate(lines):
        m = META_RE.match(line)
        if m:
            meta[m.group(1).lower()] = m.group(2).strip()
        elif line.strip() == "":
            body_start = i + 1
            # Keep consuming blank lines
            while body_start < len(lines) and lines[body_start].strip() == "":
                body_start += 1
            break
        else:
            break

    body = "\n".join(lines[body_start:])
    return meta, body


def render_notebook(nb_path, start=0, end=None):
    """Convert a Jupyter notebook (or slice of it) to HTML."""
    from nbconvert import HTMLExporter
    from nbconvert.preprocessors import Preprocessor
    from traitlets import Integer
    import nbformat

    class SliceIndex(Integer):
        default_value = None
        def validate(self, obj, value):
            if value is None:
                return value
            return super().validate(obj, value)

    class SubCell(Preprocessor):
        start = SliceIndex(0, config=True)
        end = SliceIndex(None, config=True)
        def preprocess(self, nb, resources):
            nb.cells = nb.cells[self.start:self.end]
            return nb, resources

    nb = nbformat.read(str(nb_path), as_version=4)

    # Slice cells
    nb.cells = nb.cells[start:end]

    exporter = HTMLExporter()
    exporter.template_name = "classic"
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True

    html, resources = exporter.from_notebook_node(nb)

    # Strip the full HTML wrapper — extract just the body content
    # The classic template wraps in <body> tags
    body_match = re.search(r"<body[^>]*>(.*)</body>", html, re.DOTALL)
    if body_match:
        html = body_match.group(1)

    return html


def _protect_math(body):
    """Protect LaTeX math blocks from markdown processing.

    Replaces content inside $$...$$ and $...$ with placeholders,
    so markdown doesn't mangle characters like * and _.
    """
    placeholders = []

    def save(m):
        placeholders.append(m.group(0))
        return f"\x00MATH{len(placeholders) - 1}\x00"

    # Protect display math ($$...$$), bare \begin{}...\end{}, then inline ($...$)
    body = re.sub(r'\$\$.*?\$\$', save, body, flags=re.DOTALL)
    body = re.sub(r'\\begin\{[^}]+\}.*?\\end\{[^}]+\}', save, body, flags=re.DOTALL)
    body = re.sub(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', save, body, flags=re.DOTALL)
    return body, placeholders


def _restore_math(html, placeholders):
    """Restore LaTeX math from placeholders."""
    for i, original in enumerate(placeholders):
        html = html.replace(f"\x00MATH{i}\x00", original)
    return html


def _extract_macros(body):
    """Extract <macros>...</macros> block, return (body_without_macros, macros_dict).

    Parses LaTeX \\newcommand and \\def into a dict suitable for MathJax tex.macros.
    """
    m = re.search(r'<macros>(.*?)</macros>', body, re.DOTALL)
    if not m:
        return body, {}
    block = m.group(1)
    body = body[:m.start()] + body[m.end():]
    macros = {}

    def _match_braced(s, start):
        """Return content between matched braces starting at s[start]='{'. """
        assert s[start] == '{'
        depth, i = 1, start + 1
        while i < len(s) and depth > 0:
            if s[i] == '{': depth += 1
            elif s[i] == '}': depth -= 1
            i += 1
        return s[start + 1 : i - 1], i

    i = 0
    while i < len(block):
        # \newcommand{\name}[nargs]{expansion} or \def\name{expansion}
        cm = re.match(r'\\(?:newcommand|renewcommand)\{\\(\w+)\}', block[i:])
        if cm:
            i += cm.end()
            nargs = None
            if i < len(block) and block[i] == '[':
                j = block.index(']', i)
                nargs = int(block[i + 1 : j])
                i = j + 1
            expansion, i = _match_braced(block, i)
            macros[cm.group(1)] = [expansion, nargs] if nargs else expansion
            continue
        dm = re.match(r'\\def\\(\w+)', block[i:])
        if dm:
            i += dm.end()
            expansion, i = _match_braced(block, i)
            macros[dm.group(1)] = expansion
            continue
        i += 1

    return body, macros


def _convert_sidenotes(body):
    """Convert <footnote>...</footnote> to sidenote markup."""
    return re.sub(
        r'<footnote>(.*?)</footnote>',
        r'<label class="sidenote-number"></label><span class="margin-note">\1</span>',
        body,
        flags=re.DOTALL,
    )


def _convert_simple_footnotes(body):
    """Convert legacy [ref]...[/ref] tags to markdown [^N] footnote syntax."""
    footnotes = []

    def replace_ref(match):
        n = len(footnotes) + 1
        footnotes.append(f"[^{n}]: {match.group(1).strip()}")
        return f"[^{n}]"

    body = re.sub(r'\[ref\](.*?)\[/ref\]', replace_ref, body, flags=re.DOTALL)
    if footnotes:
        body = body.rstrip() + "\n\n" + "\n\n".join(footnotes)
    return body


def _render_md_in_html_blocks(body):
    """Pre-render markdown inside HTML block tags.

    The standard markdown processor skips markdown inside HTML blocks.
    This finds content between block-level HTML tags and renders it
    inline, so images, emphasis, links, etc. all work inside HTML.
    """
    block_tags = r'address|article|aside|blockquote|center|details|dialog|dd|div|dl|dt|fieldset|figcaption|figure|footer|form|h[1-6]|header|hgroup|hr|li|main|nav|ol|p|pre|section|summary|table|ul'

    def render_inner(m):
        open_tag = m.group(1)
        inner = m.group(2)
        close_tag = m.group(3)
        md = markdown.Markdown(extensions=["extra"])
        rendered = md.convert(inner.strip())
        return f"{open_tag}\n{rendered}\n{close_tag}"

    return re.sub(
        rf'(<(?:{block_tags})(?:\s[^>]*)?>)(.*?)(</(?:{block_tags})>)',
        render_inner,
        body,
        flags=re.DOTALL,
    )


def _protect_mermaid(body):
    """Replace ```mermaid...``` blocks with placeholders before markdown rendering."""
    mermaid_blocks = []
    def replace(m):
        idx = len(mermaid_blocks)
        mermaid_blocks.append(m.group(1))
        return f"MERMAID_PLACEHOLDER_{idx}"
    body = re.sub(r'```mermaid\n(.*?)```', replace, body, flags=re.DOTALL)
    return body, mermaid_blocks


def _restore_mermaid(html, mermaid_blocks):
    """Restore mermaid placeholders as <div class="mermaid"> elements."""
    for i, block in enumerate(mermaid_blocks):
        html = html.replace(
            f"MERMAID_PLACEHOLDER_{i}",
            f'<div class="mermaid">\n{block}</div>',
        )
        # Also handle case where markdown wraps it in a <p>
        html = html.replace(
            f"<p>MERMAID_PLACEHOLDER_{i}</p>",
            f'<div class="mermaid">\n{block}</div>',
        )
    return html


def render_markdown(body):
    """Render markdown string to HTML with math-friendly settings."""
    body = _convert_sidenotes(body)
    body = _convert_simple_footnotes(body)
    body, placeholders = _protect_math(body)
    body, mermaid_blocks = _protect_mermaid(body)
    body = _render_md_in_html_blocks(body)
    md = markdown.Markdown(
        extensions=["extra", "codehilite", "toc"],
        extension_configs={
            "codehilite": {"css_class": "highlight"},
        },
    )
    html = md.convert(body)
    html = _restore_mermaid(html, mermaid_blocks)
    html = _restore_math(html, placeholders)
    return html


def process_post(filepath):
    """Process a single post: parse, handle notebooks, render to HTML."""
    meta, body = parse_post(filepath)

    is_draft = meta.get("status", "").lower() == "draft"

    # Extract LaTeX macros for MathJax config
    body, tex_macros = _extract_macros(body)

    # Check for notebook embedding (supports multiple {% notebook %} tags)
    nb_matches = list(NOTEBOOK_RE.finditer(body))
    if nb_matches:
        parts = []
        last_end = 0
        for nb_match in nb_matches:
            # Render any markdown before this notebook tag
            pre = body[last_end:nb_match.start()].strip()
            if pre:
                parts.append(render_markdown(pre))
            nb_src = nb_match.group(1)
            start = int(nb_match.group(2)) if nb_match.group(2) else 0
            end = int(nb_match.group(3)) if nb_match.group(3) else None
            nb_path = CONTENT_DIR / nb_src
            if not nb_path.exists():
                print(f"  WARNING: notebook not found: {nb_path}", file=sys.stderr)
                return None
            print(f"  Converting notebook: {nb_src} [{start}:{end}]")
            parts.append(render_notebook(nb_path, start=start, end=end))
            last_end = nb_match.end()
        # Render any markdown after the last notebook tag
        post = body[last_end:].strip()
        if post:
            parts.append(render_markdown(post))
        content = "\n".join(parts)
    else:
        content = render_markdown(body)

    # Parse date
    date_str = meta.get("date", "")
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        print(f"  WARNING: bad date '{date_str}' in {filepath}", file=sys.stderr)
        return None

    # Build slug from title (matching Pelican's behavior for URL compatibility)
    slug = slugify(meta.get("title", filepath.stem))

    # Tags
    tags = [t.strip() for t in meta.get("tags", "").split(",") if t.strip()]

    old_url = f"post/{date:%Y}/{date:%m}/{date:%d}/{slug}/"
    url = f"{slug}/"
    depth = len(Path(url).parts)
    root = "/".join([".."] * depth)

    # Rewrite absolute self-links to relative (only in href/src attributes,
    # not in plain text like bibtex entries where URLs should stay absolute)
    for prefix in ["https://timvieira.github.io/blog/", "http://timvieira.github.io/blog/"]:
        content = re.sub(
            r'(href|src)="' + re.escape(prefix),
            r'\1="' + root + "/",
            content,
        )

    return {
        "title": meta.get("title", slug),
        "date": date,
        "date_str": date.strftime("%b %d, %Y"),
        "tags": tags,
        "slug": slug,
        "content": content,
        "url": url,
        "old_url": old_url,
        "draft": is_draft,
        "tex_macros": tex_macros,
        "external_url": meta.get("external_url", ""),
    }


def build():
    # Load template
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=False,
    )
    template = env.get_template("template.html")

    # Clean and create output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    # Process all posts
    posts = []
    md_files = sorted(CONTENT_DIR.glob("*.md"))
    print(f"Processing {len(md_files)} markdown files...")
    for filepath in md_files:
        print(f"  {filepath.name}")
        post = process_post(filepath)
        if post:
            posts.append(post)

    # Sort by date, newest first
    posts.sort(key=lambda p: p["date"], reverse=True)

    # Separate drafts from published posts
    drafts = [p for p in posts if p["draft"]]
    published = [p for p in posts if not p["draft"]]
    print(f"\n{len(published)} posts, {len(drafts)} drafts")

    # Collect all unique tags (from published posts only)
    seen_tags = {}
    for post in published:
        for tag in post["tags"]:
            if tag not in seen_tags:
                seen_tags[tag] = True
    all_tags = list(seen_tags.keys())

    # Render all posts (including drafts), skip external-link posts
    for post in posts:
        if post["external_url"]:
            continue
        post_dir = OUTPUT_DIR / post["url"]
        post_dir.mkdir(parents=True, exist_ok=True)
        # post lives at e.g. post/2021/03/20/slug/index.html — 5 levels deep
        depth = len(Path(post["url"]).parts)
        root = "/".join([".."] * depth)
        html = template.render(
            page_type="article",
            post=post,
            posts=published,
            site_name=SITE_NAME,
            root=root,
            author=AUTHOR,
            all_tags=all_tags,
            **EXTRA_TEMPLATE_VARS,
        )
        (post_dir / "index.html").write_text(html, encoding="utf-8")

        # Create redirect from old dated URL to the canonical slug URL
        old_dir = OUTPUT_DIR / post["old_url"]
        old_dir.mkdir(parents=True, exist_ok=True)
        redirect_url = "/blog/" + post["url"]
        redirect_html = (
            f'<!DOCTYPE html><html><head>'
            f'<meta http-equiv="refresh" content="0; url={redirect_url}">'
            f'<link rel="canonical" href="{redirect_url}">'
            f'</head><body>'
            f'Redirecting to <a href="{redirect_url}">{post["title"]}</a>'
            f'</body></html>'
        )
        (old_dir / "index.html").write_text(redirect_html, encoding="utf-8")

    # Root index: for satellite repos with one post, use the article itself
    if SATELLITE and len(published) == 1:
        post = dict(published[0])
        # Rewrite relative paths: article content uses ../ (relative to slug/),
        # but the root index is one level up, so strip one ../ prefix
        post["content"] = re.sub(
            r'(href|src)="\.\.\/',
            r'\1="./',
            post["content"],
        )
        html = template.render(
            page_type="article",
            post=post,
            posts=published,
            site_name=SITE_NAME,
            root=".",
            author=AUTHOR,
            all_tags=all_tags,
            **EXTRA_TEMPLATE_VARS,
        )
        (OUTPUT_DIR / "index.html").write_text(html, encoding="utf-8")
    else:
        # Render archive index (lives at output root)
        html = template.render(
            page_type="archive",
            posts=published,
            site_name=SITE_NAME,
            root=".",
            author=AUTHOR,
            all_tags=all_tags,
            **EXTRA_TEMPLATE_VARS,
        )
        (OUTPUT_DIR / "index.html").write_text(html, encoding="utf-8")

    # Render drafts index
    if drafts:
        drafts_dir = OUTPUT_DIR / "drafts"
        drafts_dir.mkdir(parents=True, exist_ok=True)
        html = template.render(
            page_type="archive",
            posts=drafts,
            site_name=SITE_NAME,
            root="..",
            author=AUTHOR,
            all_tags=all_tags,
            **EXTRA_TEMPLATE_VARS,
        )
        (drafts_dir / "index.html").write_text(html, encoding="utf-8")

    # Copy static assets
    for static_dir in STATIC_DIRS:
        src = CONTENT_DIR / static_dir
        if src.exists():
            dst = OUTPUT_DIR / static_dir
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"Copied {static_dir}/")

    favicon = CONTENT_DIR / "favicon.png"
    if favicon.exists():
        shutil.copy2(favicon, OUTPUT_DIR / "favicon.png")

    # Generate pygments CSS
    formatter = HtmlFormatter(cssclass="highlight")
    pygments_css = formatter.get_style_defs(".highlight")
    (OUTPUT_DIR / "pygments.css").write_text(pygments_css, encoding="utf-8")

    # Generate Atom feed
    build_feed(published)

    print(f"\nDone. Output in {OUTPUT_DIR}/")


def build_feed(posts, max_entries=20):
    """Generate a minimal Atom feed."""
    entries = posts[:max_entries]
    feed_lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<feed xmlns="http://www.w3.org/2005/Atom">',
        f"  <title>{xml_escape(SITE_NAME)}</title>",
        f'  <link href="{FEED_URL}/" rel="alternate"/>',
        f'  <link href="{FEED_URL}/atom.xml" rel="self"/>',
        f"  <id>{FEED_URL}/</id>",
        f"  <updated>{entries[0]['date'].strftime('%Y-%m-%dT%H:%M:%SZ')}</updated>",
    ]
    for post in entries:
        entry_id = f"{FEED_URL}/{post['url']}"
        link = post["external_url"] if post["external_url"] else entry_id
        feed_lines.extend([
            "  <entry>",
            f"    <title>{xml_escape(post['title'])}</title>",
            f'    <link href="{link}" rel="alternate"/>',
            f"    <id>{entry_id}</id>",
            f"    <updated>{post['date'].strftime('%Y-%m-%dT%H:%M:%SZ')}</updated>",
            f"    <author><name>{xml_escape(AUTHOR)}</name></author>",
            "  </entry>",
        ])
    feed_lines.append("</feed>")
    (OUTPUT_DIR / "atom.xml").write_text("\n".join(feed_lines), encoding="utf-8")


if __name__ == "__main__":
    build()
