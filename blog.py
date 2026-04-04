#!/usr/bin/env python3
"""Build and dev server for satellite blog posts.

Usage:
    blog dev       # build + watch + serve
    blog build     # build only
    blog deploy    # build + commit output/ + push

Reads config from [tool.blog] in pyproject.toml, or from blog.toml.
All fields are optional with sensible defaults:

    [tool.blog]
    content = "content"     # source directory
    output = "output"       # build output directory
    static = []             # extra dirs to copy from content/ (e.g. ["images"])
    build = true            # false for hand-crafted HTML (no build.py step)
    port = 8000             # dev server port
"""

import subprocess
import sys
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

BLOG_DIR = Path(__file__).resolve().parent
BLOG_CONTENT_DIR = BLOG_DIR / "content"
BLOG_CSS_URL = "/blog/css/blog.css"

SATELLITE_TEMPLATE_VARS = {"css_url": BLOG_CSS_URL}

DEFAULTS = {
    "content": "content",
    "output": "docs",
    "static": [],
    "build": True,
    "port": 8000,
}


def load_config():
    """Load config from pyproject.toml [tool.blog] or blog.toml."""
    config = dict(DEFAULTS)

    pyproject = Path("pyproject.toml")
    blog_toml = Path("blog.toml")

    if pyproject.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        config.update(data.get("tool", {}).get("blog", {}))
    elif blog_toml.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        with open(blog_toml, "rb") as f:
            config.update(tomllib.load(f))

    return config


def do_build(config):
    """Run the blog build system for this satellite post."""
    sys.path.insert(0, str(BLOG_DIR))
    import build

    build.CONTENT_DIR = Path(config["content"])
    build.OUTPUT_DIR = Path(config["output"])
    build.TEMPLATE_DIR = BLOG_DIR
    build.STATIC_DIRS = list(config["static"])
    build.EXTRA_TEMPLATE_VARS = SATELLITE_TEMPLATE_VARS
    build.SATELLITE = config.get("satellite", True)
    build.build()


def serve(directory, port=8000):
    """Serve a directory over HTTP, proxying /blog/* to the blog's content.

    Auto-increments port if the requested one is busy.
    Blocks forever (Ctrl-C to stop).
    """
    directory = str(Path(directory).resolve())

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=directory, **kw)

        def translate_path(self, path):
            if path.startswith("/blog/"):
                return str(BLOG_CONTENT_DIR / path[len("/blog/"):])
            return super().translate_path(path)

        def log_message(self, fmt, *args):
            pass

    while True:
        try:
            httpd = HTTPServer(("", port), Handler)
            break
        except OSError:
            port += 1

    print(f"Serving at http://localhost:{port}/")
    httpd.serve_forever()


def watch(watch_paths, build_fn, extensions=None):
    """Watch paths for changes and call build_fn with debouncing.

    Returns the watchdog Observer (already started).
    """
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    if extensions is None:
        extensions = {'.md', '.ipynb', '.html', '.css', '.js'}

    lock = threading.Lock()
    timer = [None]

    class Rebuild(FileSystemEventHandler):
        def on_any_event(self, event):
            if event.event_type not in {'modified', 'created', 'deleted', 'moved'}:
                return
            if event.is_directory:
                return
            path = Path(event.dest_path if event.event_type == 'moved' else event.src_path)
            if path.suffix not in extensions:
                return
            if '.ipynb_checkpoints' in str(path):
                return
            with lock:
                if timer[0] is not None:
                    timer[0].cancel()
                timer[0] = threading.Timer(0.5, self._rebuild, args=[path.name])
                timer[0].start()

        def _rebuild(self, name):
            print(f"\n--- Rebuilding ({name} changed) ---")
            try:
                build_fn()
            except Exception as e:
                print(f"\n*** Build error: {e} ***\n")

    handler = Rebuild()
    observer = Observer()
    for p in watch_paths:
        p = Path(p)
        if p.is_dir():
            observer.schedule(handler, str(p), recursive=True)
        elif p.exists():
            observer.schedule(handler, str(p.parent), recursive=False)
    observer.start()
    return observer


# --- CLI commands ---

def cmd_build(config):
    """Build the site."""
    if config["build"]:
        do_build(config)
    else:
        print("No build step configured.")


def cmd_dev(config):
    """Build, watch for changes, and serve."""
    if config["build"]:
        do_build(config)
        watch_paths = [config["content"], BLOG_DIR / "template.html"]
        watch(watch_paths, lambda: do_build(config))
    serve(config["output"], port=config["port"])


def cmd_deploy(config):
    """Build and deploy to GitHub Pages."""
    if not config["build"]:
        subprocess.run(["git", "push"], check=True)
        print("Pushed.")
        return

    do_build(config)

    output = config["output"]
    subprocess.run(["git", "add", f"{output}/"], check=True)

    result = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if result.returncode == 0:
        print("No changes to deploy.")
        return

    subprocess.run(
        ["git", "commit", "-m", "Rebuild output for GitHub Pages"],
        check=True,
    )
    subprocess.run(["git", "push"], check=True)
    print("Deployed.")


def main():
    commands = {"build": cmd_build, "dev": cmd_dev, "deploy": cmd_deploy}

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f"Usage: blog <{'|'.join(commands)}>")
        sys.exit(1)

    config = load_config()
    commands[sys.argv[1]](config)


if __name__ == "__main__":
    main()
