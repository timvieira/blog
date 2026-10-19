# Blog Build System

Static blog builder: `python3 build.py` (or `make html`). Serves locally with `make serve`.

## Rules

- **Pushing to master publishes the live site.** GitHub Actions (`.github/workflows/publish.yml`) builds and deploys to https://timvieira.github.io/blog/ on every push to master, including merged PRs. Only `git push` when requested.
- **`blog.py` is shared.** It is symlinked as `~/.local/bin/blog` and used by satellite projects (e.g. steering-embeddings, table-theorem, conditional-poisson-sampling), whose live pages also load `/blog/css/blog.css`. Don't break its CLI or move that CSS.
- **Rebuild after each change.** Run `make html` after every modification to verify the output.
- **Tags must never move.** On state change (active, disabled, hover), only modify color, background-color, opacity, border-color. Never modify font-weight, font-size, padding, or margin.
- **Suggest commit checkpoints** after meaningful changes to avoid losing good states.
