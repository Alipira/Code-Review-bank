# Senior Data Science Interview Bank

**882 questions across 7 banks** — a complete senior data-scientist interview kit as a static website: coding and code review, EDA, statistics & probability, visual diagnosis (124 generated figures), ML depth, and production judgment. Answer keys are collapsible; difficulties are tagged E (screen) / M (core) / H (senior signal).

## Publish with GitHub Pages

1. Create a new repository and push these files to the `main` branch (keep `index.html` at the repo root).
2. In the repo: **Settings → Pages → Build and deployment → Source: Deploy from a branch → Branch: `main` / `/ (root)` → Save**.
3. Your site appears at `https://<username>.github.io/<repo>/` within a minute or two.

The `.nojekyll` file is required — it tells Pages to serve files as-is (no Jekyll processing).

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
