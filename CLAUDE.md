# Blog Build System

Static blog builder: `python3 build.py` (or `make html`). Serves locally with `make serve`.

## Rules

- **NEVER push.** No `git push`, no `make push`, no deployment. Only the user pushes.
- **Rebuild after each change.** Run `make html` after every modification to verify the output.
- **Tags must never move.** On state change (active, disabled, hover), only modify color, background-color, opacity, border-color. Never modify font-weight, font-size, padding, or margin.
- **Suggest commit checkpoints** after meaningful changes to avoid losing good states.
