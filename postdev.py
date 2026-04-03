"""Dev tools for satellite blog posts hosted outside the main blog repo.

Provides a dev server with CSS proxying, file watching with auto-rebuild,
and a deploy helper that rewrites CSS references to the hosted blog URL.

Usage from a satellite post repo:

    import sys; from pathlib import Path
    sys.path.insert(0, str(Path.home() / "projects/blog/main"))
    import postdev

    # Dev server (proxies /css/* to blog's CSS directory)
    postdev.serve("output", port=8000)

    # File watcher with debounced rebuild
    postdev.watch(["content"], build_fn=my_build)

    # Deploy: rewrite CSS refs to absolute URL
    postdev.rewrite_blog_css("output")
"""

import re
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

BLOG_DIR = Path(__file__).resolve().parent
BLOG_CSS_DIR = BLOG_DIR / "content" / "css"
BLOG_CSS_URL = "https://timvieira.github.io/blog/css/blog.css"


def serve(directory, port=8000):
    """Serve a directory over HTTP, proxying /css/* to the blog's CSS.

    Auto-increments port if the requested one is busy.
    Blocks forever (Ctrl-C to stop).
    """
    directory = str(Path(directory).resolve())

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=directory, **kw)

        def translate_path(self, path):
            if path.startswith("/css/"):
                return str(BLOG_CSS_DIR / path[5:])
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

    Args:
        watch_paths: list of directories/files to watch.
        build_fn: callable to invoke on changes.
        extensions: set of file extensions to trigger on
                    (default: .md, .ipynb, .html, .css, .js).

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


def rewrite_blog_css(directory):
    """Rewrite relative CSS references in HTML files to the absolute blog URL."""
    for html_file in Path(directory).rglob("*.html"):
        text = html_file.read_text()
        fixed = re.sub(
            r'href="[^"]*?/css/blog\.css"',
            f'href="{BLOG_CSS_URL}"',
            text,
        )
        if fixed != text:
            html_file.write_text(fixed)
