# Senior Data Science Interview Bank

**882 questions across 7 banks** — a complete senior data-scientist interview kit as a static website: coding and code review, EDA, statistics & probability, visual diagnosis (124 generated figures), ML depth, and production judgment. Answer keys are collapsible; difficulties are tagged E (screen) / M (core) / H (senior signal).

## Preview locally

```bash
python -m http.server 8000
# open http://localhost:8000
```

## Structure

```
index.html                  landing page
banks/                      the seven interactive banks (self-contained HTML)
banks/figures/              124 PNGs for the visual-diagnosis bank
src/                        markdown sources for every bank
.nojekyll                   disable Jekyll on GitHub Pages
```

## License

Add the license of your choice (MIT is a common pick for interview material you intend to share).
