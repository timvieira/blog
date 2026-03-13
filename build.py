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

import jinja2
import markdown
from pygments.formatters import HtmlFormatter

CONTENT_DIR = Path("content")
OUTPUT_DIR = Path("output/blog")
TEMPLATE_DIR = Path(".")
STATIC_DIRS = ["images", "figures", "downloads"]
SITE_NAME = "Graduate Descent"

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


def render_markdown(body):
    """Render markdown string to HTML with math-friendly settings."""
    md = markdown.Markdown(
        extensions=["extra", "codehilite", "toc"],
        extension_configs={
            "codehilite": {"css_class": "highlight"},
        },
    )
    return md.convert(body)


def process_post(filepath):
    """Process a single post: parse, handle notebooks, render to HTML."""
    meta, body = parse_post(filepath)

    # Skip drafts
    if meta.get("status", "").lower() == "draft":
        return None

    # Check for notebook embedding
    nb_match = NOTEBOOK_RE.search(body)
    if nb_match:
        nb_src = nb_match.group(1)
        start = int(nb_match.group(2)) if nb_match.group(2) else 0
        end = int(nb_match.group(3)) if nb_match.group(3) else None
        nb_path = CONTENT_DIR / nb_src
        if not nb_path.exists():
            print(f"  WARNING: notebook not found: {nb_path}", file=sys.stderr)
            return None
        print(f"  Converting notebook: {nb_src} [{start}:{end}]")
        content = render_notebook(nb_path, start=start, end=end)
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

    url = f"post/{date:%Y}/{date:%m}/{date:%d}/{slug}/"
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
    print(f"\n{len(posts)} posts (drafts excluded)")

    # Collect all unique tags (preserving order of first appearance)
    seen_tags = {}
    for post in posts:
        for tag in post["tags"]:
            if tag not in seen_tags:
                seen_tags[tag] = True
    all_tags = list(seen_tags.keys())

    # Render each post
    for post in posts:
        post_dir = OUTPUT_DIR / post["url"]
        post_dir.mkdir(parents=True, exist_ok=True)
        # post lives at e.g. post/2021/03/20/slug/index.html — 5 levels deep
        depth = len(Path(post["url"]).parts)
        root = "/".join([".."] * depth)
        html = template.render(
            page_type="article",
            post=post,
            posts=posts,
            site_name=SITE_NAME,
            root=root,
            author=AUTHOR,
            all_tags=all_tags,
        )
        (post_dir / "index.html").write_text(html, encoding="utf-8")

    # Render archive index (lives at output root)
    html = template.render(
        page_type="archive",
        posts=posts,
        site_name=SITE_NAME,
        root=".",
        author=AUTHOR,
        all_tags=all_tags,
    )
    (OUTPUT_DIR / "index.html").write_text(html, encoding="utf-8")

    # Copy static assets
    for static_dir in STATIC_DIRS:
        src = CONTENT_DIR / static_dir
        if src.exists():
            dst = OUTPUT_DIR / static_dir
            shutil.copytree(src, dst)
            print(f"Copied {static_dir}/")

    favicon = CONTENT_DIR / "favicon.png"
    if favicon.exists():
        shutil.copy2(favicon, OUTPUT_DIR / "favicon.png")

    # Generate pygments CSS
    formatter = HtmlFormatter(cssclass="highlight")
    pygments_css = formatter.get_style_defs(".highlight")
    (OUTPUT_DIR / "pygments.css").write_text(pygments_css, encoding="utf-8")

    # Generate Atom feed
    build_feed(posts)

    print(f"\nDone. Output in {OUTPUT_DIR}/")


def build_feed(posts, max_entries=20):
    """Generate a minimal Atom feed."""
    entries = posts[:max_entries]
    feed_lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<feed xmlns="http://www.w3.org/2005/Atom">',
        f"  <title>{SITE_NAME}</title>",
        f'  <link href="{FEED_URL}/" rel="alternate"/>',
        f'  <link href="{FEED_URL}/atom.xml" rel="self"/>',
        f"  <id>{FEED_URL}/</id>",
        f"  <updated>{entries[0]['date'].strftime('%Y-%m-%dT%H:%M:%SZ')}</updated>",
    ]
    for post in entries:
        url = f"{FEED_URL}/{post['url']}"
        feed_lines.extend([
            "  <entry>",
            f"    <title>{post['title']}</title>",
            f'    <link href="{url}" rel="alternate"/>',
            f"    <id>{url}</id>",
            f"    <updated>{post['date'].strftime('%Y-%m-%dT%H:%M:%SZ')}</updated>",
            f"    <author><name>{AUTHOR}</name></author>",
            "  </entry>",
        ])
    feed_lines.append("</feed>")
    (OUTPUT_DIR / "atom.xml").write_text("\n".join(feed_lines), encoding="utf-8")


if __name__ == "__main__":
    build()
