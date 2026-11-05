"""
Draw the derivation-tree figures for the weighted CFG post (content/wcfg.md),
redrawn from Opedal et al. (2023), "An Exploration of Left-Corner
Transformations" (arXiv 2311.16258).  Run from the repo root:

    python tools/wcfg_figures.py

Writes content/images/wcfg-*.svg.
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / 'content' / 'images'
INK = '#333'
SERIF = "Georgia, 'Times New Roman', serif"


class SVG:
    def __init__(self, w, h):
        self.w, self.h, self.parts = w, h, []

    def add(self, s):
        self.parts.append(s)

    def line(self, x1, y1, x2, y2, width=1):
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{INK}" stroke-width="{width}"/>')

    def text(self, x, y, s, size=15, anchor='middle', italic=False):
        style = ' font-style="italic"' if italic else ''
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{INK}" text-anchor="{anchor}"{style}>{s}</text>')

    def save(self, name, title):
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
               f'font-family="{SERIF}" role="img">\n<title>{title}</title>\n'
               + '\n'.join(self.parts) + '\n</svg>\n')
        (OUT / name).write_text(svg)


def lc(top, bottom):
    "The left-corner nonterminal written as a slashed fraction, e.g. S/NP."
    return (f'<tspan font-size="80%" baseline-shift="35%">{top}</tspan>'
            f'<tspan font-size="90%">/</tspan>'
            f'<tspan font-size="80%" baseline-shift="-25%">{bottom}</tspan>')


# A tree is (label, [children]) or ('ROOF', text) for a roofed yield, or a plain
# string for a terminal leaf.
def draw_tree(svg, t, x0, y0, gap=20, dy=46, char=7.6):
    """Lay out leaves left to right (each as wide as its text) and center each
    parent over its children; draw edges and labels.  Returns the right edge."""
    pos = {}
    cursor = [x0]

    def width(t):
        if isinstance(t, tuple) and t[0] == 'ROOF':
            return max(40, char * len(t[1]))
        if isinstance(t, str):
            return max(16, char * len(t))
        return None

    def layout(t, path, depth):
        w = width(t)
        if w is not None:
            x = cursor[0] + w / 2
            cursor[0] += w + gap
            pos[path] = (x, y0 + depth * dy)
        else:
            xs = [layout(c, path + (i,), depth + 1)[0] for i, c in enumerate(t[1])]
            pos[path] = ((xs[0] + xs[-1]) / 2, y0 + depth * dy)
        return pos[path]

    layout(t, (), 0)

    def render(t, path):
        x, y = pos[path]
        if isinstance(t, tuple) and t[0] == 'ROOF':
            w = width(t)
            svg.add(f'<polygon points="{x:.1f},{y - 18:.1f} {x - w / 2:.1f},{y + 2:.1f} {x + w / 2:.1f},{y + 2:.1f}" '
                    f'fill="none" stroke="{INK}" stroke-width="0.9"/>')
            svg.text(x, y + 18, t[1], size=14)
            return
        if isinstance(t, str):
            svg.text(x, y + 5, t, size=14, italic=(t == 'ε'))
            return
        label, kids = t
        if label.startswith('FROZEN:'):
            label = label[len('FROZEN:'):]
            svg.text(x, y - 12, '~', size=15)
        for i, c in enumerate(kids):
            cx, cy = pos[path + (i,)]
            if isinstance(c, tuple) and (c[0] == 'ROOF' or c[0].startswith('FROZEN:')):
                top = cy - 24
            else:
                top = cy - 12
            svg.line(x, y + 6, cx, top, width=0.9)
            render(c, path + (i,))
        svg.text(x, y + 5, label, size=15)

    render(t, ())
    return cursor[0]


def motivating_example():
    """Figure 1 of the paper: a derivation with left recursion (left) and the
    corresponding derivation after the left-corner transformation (right)."""
    S = '<tspan font-family="cursive">𝒮</tspan>'
    left = (S, [
        ('NP', [
            ('PossP', [('NP', [('ROOF', 'My sister')]), "'s"]),
            ('NN', [('ROOF', 'diploma')]),
        ]),
        ('VP', [('ROOF', 'arrived')]),
    ])
    right = (S, [
        ('FROZEN:NP', [('ROOF', 'My sister')]),
        (lc(S, 'NP'), [
            "'s",
            (lc(S, 'PossP'), [
                ('NN', [('ROOF', 'diploma')]),
                (lc(S, 'NP'), [
                    ('VP', [('ROOF', 'arrived')]),
                    (lc(S, S), ['ε']),
                ]),
            ]),
        ]),
    ])
    svg = SVG(760, 300)
    r = draw_tree(svg, left, 20, 30)
    r2 = draw_tree(svg, right, r + 40, 30)
    svg.w = int(r2 + 10)
    svg.save('wcfg-left-corner-example.svg', 'A derivation with left recursion, and after the left-corner transformation')


def left_tree_only():
    "Figure 1 (left) on its own: an example derivation."
    S = '<tspan font-family="cursive">𝒮</tspan>'
    left = (S, [
        ('NP', [
            ('PossP', [('NP', [('ROOF', 'My sister')]), "'s"]),
            ('NN', [('ROOF', 'diploma')]),
        ]),
        ('VP', [('ROOF', 'arrived')]),
    ])
    svg = SVG(340, 240)
    svg.w = int(draw_tree(svg, left, 10, 30) + 5)
    svg.save('wcfg-example-derivation.svg', 'An example derivation of "My sister\'s diploma arrived"')


def generic_rule():
    """The inductive case of the definition of derivations: a node X whose
    children are subderivations rooted at alpha_1, ..., alpha_K."""
    svg = SVG(260, 130)
    x, y = 130, 26
    svg.text(x, y, 'X', size=17)
    kids = [(50, 'α', '1'), (210, 'α', 'K')]
    for cx, a, i in kids:
        svg.line(x, y + 6, cx, y + 40, width=0.9)
        svg.add(f'<text x="{cx:.1f}" y="{y + 52:.1f}" font-size="16" fill="{INK}" text-anchor="middle" font-style="italic">{a}'
                f'<tspan font-size="60%" dy="3" font-style="italic">{i}</tspan></text>')
        svg.add(f'<polygon points="{cx:.1f},{y + 64:.1f} {cx - 26:.1f},{y + 96:.1f} {cx + 26:.1f},{y + 96:.1f}" '
                f'fill="none" stroke="{INK}" stroke-width="0.9"/>')
    svg.text(x, y + 52, '⋯', size=16)
    svg.save('wcfg-derivation-node.svg', 'A derivation built by the rule X → α1 ⋯ αK')


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    generic_rule()
    left_tree_only()
    motivating_example()
