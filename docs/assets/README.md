# Brand Assets (`docs/assets/`)

Multi-platform banners, icons, emojis, and favicons for **group-by-vibes**, generated reproducibly from [`source/banner_master.png`](source/banner_master.png) via [`generate_assets.py`](generate_assets.py).

## Asset Inventory

| File | Dimensions | Background | Recommended Usage |
| :--- | :--- | :--- | :--- |
| [`banner-github.png`](banner-github.png) | `1280x420` | `#8C8DFE` periwinkle | GitHub `README.md` header (light & dark mode safe) |
| [`banner-github-transparent.png`](banner-github-transparent.png) | `1280x420` | Transparent | Light-background docs, slides, and web pages |
| [`social-preview-github.png`](social-preview-github.png) | `1280x640` | `#8C8DFE` periwinkle | GitHub repository Social Preview (`Settings -> General`) |
| [`banner-linkedin-personal.png`](banner-linkedin-personal.png) | `1584x396` | `#8C8DFE` periwinkle | LinkedIn personal profile cover (left safe zone for avatar) |
| [`banner-linkedin-company.png`](banner-linkedin-company.png) | `1128x191` | `#8C8DFE` periwinkle | LinkedIn company/project page banner |
| [`icon-512.png`](icon-512.png) | `512x512` | `#8C8DFE` periwinkle | GitHub org/LinkedIn avatar, PyPI/marketplace icon |
| [`icon-512-transparent.png`](icon-512-transparent.png) | `512x512` | Transparent | High-res brand mark, presentations, badges |
| [`icon-256-transparent.png`](icon-256-transparent.png) | `256x256` | Transparent | MkDocs header `logo` (`light` & `dark` compatible) |
| [`emoji-128.png`](emoji-128.png) | `128x128` | Transparent | Slack, Discord, and GitHub custom emoji (`~24 KB`) |
| [`mkdocs-favicon-64.png`](mkdocs-favicon-64.png) | `64x64` | Transparent | MkDocs `favicon` PNG |
| [`favicon.ico`](favicon.ico) | `16/32/48/64` | Transparent | Multi-resolution web `favicon.ico` |

## MkDocs Configuration Example

Place or link the assets inside your MkDocs `docs_dir` (e.g., `docs/assets/`), then add to `mkdocs.yml`:

```yaml
theme:
  name: material
  logo: assets/icon-256-transparent.png
  favicon: assets/mkdocs-favicon-64.png
```

## Regenerating Assets

All derived files are generated deterministically from [`source/banner_master.png`](source/banner_master.png) using [`generate_assets.py`](generate_assets.py):

```bash
uv run --no-project --with pillow --with numpy python docs/assets/generate_assets.py
```
