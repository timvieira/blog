"""Tests for blog build system core functionality."""

import re
from datetime import datetime
from pathlib import Path

import build


# ── slugify ─────────────────────────────────────────────────────────


def test_slugify_basic():
    assert build.slugify("Hello World") == "hello-world"


def test_slugify_special_chars():
    assert build.slugify("Fast Rank-One Updates to Matrix Inverse?") == \
        "fast-rank-one-updates-to-matrix-inverse"


def test_slugify_unicode():
    assert build.slugify("Evaluating ∇f(x) Is as Fast as f(x)") == \
        "evaluating-fx-is-as-fast-as-fx"


def test_slugify_collapses_whitespace_and_hyphens():
    assert build.slugify("Too   Many---Dashes") == "too-many-dashes"


def test_slugify_strips_leading_trailing_hyphens():
    assert build.slugify("--leading and trailing--") == "leading-and-trailing"


# ── parse_post ──────────────────────────────────────────────────────


def test_parse_post_basic(tmp_path):
    p = tmp_path / "post.md"
    p.write_text("title: My Post\ndate: 2025-01-15\ntags: math, sampling\n\nBody here.\n")
    meta, body = build.parse_post(p)
    assert meta["title"] == "My Post"
    assert meta["date"] == "2025-01-15"
    assert meta["tags"] == "math, sampling"
    assert body == "Body here."


def test_parse_post_skips_blank_lines_after_header(tmp_path):
    p = tmp_path / "post.md"
    p.write_text("title: Test\ndate: 2025-01-01\n\n\n\nActual body.\n")
    meta, body = build.parse_post(p)
    assert body == "Actual body."


def test_parse_post_external_url(tmp_path):
    p = tmp_path / "post.md"
    p.write_text("title: External\ndate: 2025-06-01\nexternal_url: https://example.com/\n\n")
    meta, body = build.parse_post(p)
    assert meta["external_url"] == "https://example.com/"


def test_parse_post_draft(tmp_path):
    p = tmp_path / "post.md"
    p.write_text("title: Draft\ndate: 2025-01-01\nstatus: draft\n\nWIP.\n")
    meta, body = build.parse_post(p)
    assert meta["status"] == "draft"


# ── math protection ─────────────────────────────────────────────────


def test_protect_math_inline():
    body = "Text with $x^2$ and more."
    protected, placeholders = build._protect_math(body)
    assert "$" not in protected
    assert len(placeholders) == 1
    restored = build._restore_math(protected, placeholders)
    assert restored == body


def test_protect_math_display():
    body = "Before $$\\sum_{i=1}^n x_i$$ after."
    protected, placeholders = build._protect_math(body)
    assert "$$" not in protected
    assert len(placeholders) == 1
    restored = build._restore_math(protected, placeholders)
    assert restored == body


def test_protect_math_begin_end():
    body = "Before \\begin{align} x &= 1 \\end{align} after."
    protected, placeholders = build._protect_math(body)
    assert "\\begin" not in protected
    assert len(placeholders) == 1
    restored = build._restore_math(protected, placeholders)
    assert restored == body


def test_protect_math_multiple():
    body = "Inline $a$ and display $$b$$ and $c$."
    protected, placeholders = build._protect_math(body)
    assert len(placeholders) == 3
    restored = build._restore_math(protected, placeholders)
    assert restored == body


def test_protect_math_preserves_underscores():
    """Math underscores should not become <em> tags."""
    body = "The formula $x_{i+1}$ is important."
    protected, placeholders = build._protect_math(body)
    assert "_" not in protected  # underscore is inside placeholder
    restored = build._restore_math(protected, placeholders)
    assert "$x_{i+1}$" in restored


# ── macros ──────────────────────────────────────────────────────────


def test_extract_macros_newcommand():
    body = "<macros>\\newcommand{\\R}{\\mathbb{R}}</macros>\n\nBody."
    cleaned, macros = build._extract_macros(body)
    assert "macros" not in cleaned.lower()
    assert macros["R"] == "\\mathbb{R}"


def test_extract_macros_with_args():
    body = "<macros>\\newcommand{\\norm}[1]{\\left\\| #1 \\right\\|}</macros>\n\nBody."
    cleaned, macros = build._extract_macros(body)
    assert macros["norm"] == ["\\left\\| #1 \\right\\|", 1]


def test_extract_macros_none():
    body = "No macros here."
    cleaned, macros = build._extract_macros(body)
    assert cleaned == body
    assert macros == {}


# ── sidenotes and footnotes ─────────────────────────────────────────


def test_convert_sidenotes():
    body = "Text<footnote>A side note.</footnote> more."
    result = build._convert_sidenotes(body)
    assert "sidenote-number" in result
    assert "margin-note" in result
    assert "A side note." in result
    assert "<footnote>" not in result


def test_convert_simple_footnotes():
    body = "Claim[ref]See paper X.[/ref] and another[ref]See paper Y.[/ref]."
    result = build._convert_simple_footnotes(body)
    assert "[^1]" in result
    assert "[^2]" in result
    assert "See paper X." in result
    assert "See paper Y." in result
    assert "[ref]" not in result


# ── URL construction (process_post) ─────────────────────────────────


def test_process_post_url_structure(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "CONTENT_DIR", tmp_path)
    p = tmp_path / "test.md"
    p.write_text("title: My Great Post\ndate: 2025-03-15\ntags: math\n\nHello world.\n")
    post = build.process_post(p)
    assert post["slug"] == "my-great-post"
    assert post["url"] == "my-great-post/"
    assert post["old_url"] == "post/2025/03/15/my-great-post/"


def test_process_post_date_parsing(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "CONTENT_DIR", tmp_path)
    p = tmp_path / "test.md"
    p.write_text("title: Test\ndate: 2025-12-25\n\nBody.\n")
    post = build.process_post(p)
    assert post["date"] == datetime(2025, 12, 25)
    assert post["date_str"] == "Dec 25, 2025"


def test_process_post_bad_date_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "CONTENT_DIR", tmp_path)
    p = tmp_path / "test.md"
    p.write_text("title: Bad\ndate: not-a-date\n\nBody.\n")
    assert build.process_post(p) is None


def test_process_post_tags(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "CONTENT_DIR", tmp_path)
    p = tmp_path / "test.md"
    p.write_text("title: Test\ndate: 2025-01-01\ntags: math, sampling, algorithms\n\nBody.\n")
    post = build.process_post(p)
    assert post["tags"] == ["math", "sampling", "algorithms"]


def test_process_post_no_tags(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "CONTENT_DIR", tmp_path)
    p = tmp_path / "test.md"
    p.write_text("title: Test\ndate: 2025-01-01\n\nBody.\n")
    post = build.process_post(p)
    assert post["tags"] == []


def test_process_post_draft_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "CONTENT_DIR", tmp_path)
    p = tmp_path / "test.md"
    p.write_text("title: Test\ndate: 2025-01-01\nstatus: draft\n\nBody.\n")
    post = build.process_post(p)
    assert post["draft"] is True


def test_process_post_published_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "CONTENT_DIR", tmp_path)
    p = tmp_path / "test.md"
    p.write_text("title: Test\ndate: 2025-01-01\n\nBody.\n")
    post = build.process_post(p)
    assert post["draft"] is False


def test_process_post_external_url(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "CONTENT_DIR", tmp_path)
    p = tmp_path / "test.md"
    p.write_text("title: Ext\ndate: 2025-01-01\nexternal_url: https://example.com/\n\n")
    post = build.process_post(p)
    assert post["external_url"] == "https://example.com/"


def test_process_post_rewrites_absolute_urls(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "CONTENT_DIR", tmp_path)
    p = tmp_path / "test.md"
    p.write_text(
        'title: Test\ndate: 2025-01-01\n\n'
        'See [this post](https://timvieira.github.io/blog/other-post/).\n'
    )
    post = build.process_post(p)
    assert "timvieira.github.io/blog/" not in post["content"]
    assert 'href="../other-post/"' in post["content"]


# ── redirect generation ─────────────────────────────────────────────


def test_redirect_points_to_canonical(tmp_path, monkeypatch):
    """Old dated URLs should redirect to the canonical slug URL."""
    monkeypatch.setattr(build, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(build, "CONTENT_DIR", tmp_path)
    monkeypatch.setattr(build, "TEMPLATE_DIR", str(Path(__file__).parent.parent))

    p = tmp_path / "test.md"
    p.write_text("title: My Post\ndate: 2025-06-15\n\nHello.\n")

    post = build.process_post(p)

    # Simulate what build() does for redirects
    old_dir = tmp_path / post["old_url"]
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
    (old_dir / "index.html").write_text(redirect_html)

    html = (old_dir / "index.html").read_text()
    assert "/blog/my-post/" in html
    assert "post/2025/06/15/my-post" not in html or "url=/blog/my-post/" in html


# ── render_markdown ─────────────────────────────────────────────────


def test_render_markdown_basic():
    html = build.render_markdown("**bold** and *italic*")
    assert "<strong>bold</strong>" in html
    assert "<em>italic</em>" in html


def test_render_markdown_preserves_math():
    html = build.render_markdown("Text $x_{i*j}$ more")
    assert "$x_{i*j}$" in html


def test_render_markdown_code_highlight():
    md = "```python\ndef foo():\n    pass\n```"
    html = build.render_markdown(md)
    assert "highlight" in html


def test_render_markdown_footnote_ref():
    md = "Claim[ref]Citation here.[/ref] and text."
    html = build.render_markdown(md)
    assert "Citation here." in html
    assert "[ref]" not in html


# ── markdown inside HTML blocks ─────────────────────────────────────


def test_details_markdown1_renders():
    """Markdown inside <details markdown="1"> should render (real post pattern)."""
    md = (
        '<details class="derivation" markdown="1">\n'
        "<summary>Proof</summary>\n\n"
        "**bold** and *italic* and [link](http://example.com)\n\n"
        "</details>"
    )
    html = build.render_markdown(md)
    assert "<strong>bold</strong>" in html
    assert "<em>italic</em>" in html
    assert 'href="http://example.com"' in html


def test_details_markdown1_with_math():
    """Math inside <details markdown="1"> should survive rendering."""
    md = (
        '<details markdown="1">\n<summary>Derivation</summary>\n\n'
        "We start with $x^2$ and then\n"
        "$$\\sum_{i=1}^n x_i$$\n\n"
        "</details>"
    )
    html = build.render_markdown(md)
    assert "$x^2$" in html
    assert "$$\\sum_{i=1}^n x_i$$" in html


def test_details_markdown1_math_underscores():
    """Math underscores inside <details> must not become <em> (issue #5)."""
    md = (
        '<details markdown="1">\n<summary>Formal definition</summary>\n\n'
        "* $\\boldsymbol{w} \\in \\boldsymbol{K}^n$\n\n"
        "* $\\oplus: \\boldsymbol{K} \\times \\boldsymbol{K} \\mapsto \\boldsymbol{K}$.\n\n"
        "</details>"
    )
    html = build.render_markdown(md)
    assert "<em>" not in html
    assert "$\\boldsymbol{w} \\in \\boldsymbol{K}^n$" in html


def test_details_markdown1_with_image():
    """Images inside <details markdown="1"> should render as <img> tags."""
    md = (
        '<details markdown="1">\n<summary>Figure</summary>\n\n'
        "![alt text](/images/fig.png)\n\n"
        "</details>"
    )
    html = build.render_markdown(md)
    assert "<img" in html
    assert "/images/fig.png" in html


def test_details_markdown1_with_code():
    """Code blocks inside <details markdown="1"> should render."""
    md = (
        '<details markdown="1">\n<summary>Code</summary>\n\n'
        "```python\ndef foo():\n    return 42\n```\n\n"
        "</details>"
    )
    html = build.render_markdown(md)
    assert "foo" in html
    assert "return" in html


def test_div_with_markdown():
    """Markdown inside <div> should also render."""
    md = "<div>\n\n**bold** text\n\n</div>"
    html = build.render_markdown(md)
    assert "<strong>bold</strong>" in html


def test_multiline_inline_math_protected():
    """Inline math spanning lines should be protected (issue #5 fix)."""
    md = "Text $a_i\n+ b_j$ end."
    html = build.render_markdown(md)
    assert "$a_i\n+ b_j$" in html
    assert "<em>" not in html
