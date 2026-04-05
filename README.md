Graduate Descent
================

[http://timvieira.github.io/blog](http://timvieira.github.io/blog)

## Build

```bash
make html     # build site to output/blog/
make serve    # build + serve locally at http://localhost:8000/blog/
```

Dependencies: `python3`, `markdown`, `jinja2`, `nbconvert`, `pygments`

## Post metadata

Posts are markdown files in `content/`. Each file starts with key-value
metadata lines (Pelican-style), followed by a blank line, then the body:

```
title: My Post Title
date: 2024-01-15
tags: foo, bar, baz
status: draft

Post body starts here...
```

Supported fields:

| Field          | Required | Description                                          |
|----------------|----------|------------------------------------------------------|
| `title`        | yes      | Post title                                           |
| `date`         | yes      | Publication date (`YYYY-MM-DD`)                      |
| `tags`         | no       | Comma-separated list of tags                         |
| `status`       | no       | Set to `draft` to exclude from the main archive      |
| `external_url` | no       | Link to external content instead of rendering a page |

## Body features

**Notebook embedding** — include Jupyter notebook cells inline:

```
{% notebook path/to/notebook.ipynb cells[0:5] %}
```

The path is relative to `content/`. You can embed multiple notebooks in one
post, with markdown between them.

**LaTeX math** — use `$...$` for inline and `$$...$$` for display math.
MathJax renders them in the browser. Math blocks are protected from markdown
processing.

**Macros** — define LaTeX macros in a `<macros>` block:

```html
<macros>
\newcommand{\R}{\mathbb{R}}
\def\argmin{\operatorname{argmin}}
</macros>
```

These are passed to MathJax's `tex.macros` config.

**Sidenotes** — `<footnote>text</footnote>` renders as a margin sidenote.

**Legacy footnotes** — `[ref]text[/ref]` is converted to standard markdown
footnotes (`[^N]`).

**Mermaid diagrams** — fenced code blocks with `mermaid` language are rendered
as diagrams.

## Satellite mode

For standalone post repos (outside this main blog repo), a CLI tool (`blog.py`)
handles building, serving, and deploying. Configure via `[tool.blog]` in
`pyproject.toml` or in `blog.toml`:

```toml
[tool.blog]
content = "content"       # source directory (default: "content")
output = "docs"           # build output (default: "docs")
static = ["images"]       # extra dirs to copy from content/
build = true              # set false for hand-crafted HTML
port = 8000               # dev server port
preprocess = [            # shell commands to run before build
    "python generate_figures.py",
]
```

Commands: `blog build`, `blog dev` (build + watch + serve), `blog deploy`.
