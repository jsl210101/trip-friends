"""Build the static browser version of Triper into docs/.

The app runs with stlite (Streamlit on Pyodide), so GitHub Pages can host it
with no server. Every source file is inlined into docs/index.html, so the page
stays in sync with the code after you rerun this script:

    python scripts/build_stlite.py
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

STLITE_VERSION = "1.12.2"
ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


def collect_files() -> dict[str, str]:
    files = {"app.py": (ROOT / "app.py").read_text(encoding="utf-8")}
    for path in sorted((ROOT / "triper").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            files[path.relative_to(ROOT).as_posix()] = path.read_text(encoding="utf-8")
    return files


def theme_config() -> dict[str, object]:
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    out: dict[str, object] = {f"theme.{k}": v for k, v in config.get("theme", {}).items()}
    out["browser.gatherUsageStats"] = False
    out["client.toolbarMode"] = "viewer"  # hide the Deploy button in the static build
    return out


TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="X-UA-Compatible" content="IE=edge" />
  <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
  <title>Triper · 小组旅行规划 / Group trip planner</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/browser@__VERSION__/build/stlite.css" />
  <style>
    body { margin: 0; background: #f6f3ec; color: #1c2b2a; font-family: system-ui, -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }
    #triper-loading { position: fixed; inset: 0; display: flex; align-items: center; justify-content: center; text-align: center; padding: 2rem; background: #f6f3ec; z-index: 10; }
    #triper-loading .box { max-width: 32rem; }
    #triper-loading h1 { color: #0f6e6b; margin: 0 0 .6rem; font-size: 2rem; }
    #triper-loading p { margin: .3rem 0; line-height: 1.5; }
    #triper-loading .spin { width: 2.2rem; height: 2.2rem; margin: 1rem auto; border: 4px solid #e7efe9; border-top-color: #0f6e6b; border-radius: 50%; animation: s 1s linear infinite; }
    @keyframes s { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <div id="triper-loading"><div class="box">
    <h1>Triper</h1>
    <div class="spin"></div>
    <p>正在浏览器里加载 Python 和 Streamlit，第一次打开大约需要 10–60 秒，请稍候。</p>
    <p>Loading Python and Streamlit in your browser. The first visit takes about 10–60 seconds.</p>
    <p style="font-size:.85rem;color:#5b6b69">所有计算都在本机浏览器中完成，不需要 API key。 / Everything runs locally in your browser; no API key needed.</p>
  </div></div>
  <div id="root"></div>
  <script type="module">
    import { mount } from "https://cdn.jsdelivr.net/npm/@stlite/browser@__VERSION__/build/stlite.js";
    const files = __FILES__;
    mount({
      requirements: [],
      entrypoint: "app.py",
      files,
      streamlitConfig: __CONFIG__,
    }, document.getElementById("root"));
    const hide = () => { const el = document.getElementById("triper-loading"); if (el) el.remove(); };
    const obs = new MutationObserver(() => {
      if (document.querySelector('[data-testid="stApp"] [data-testid="stMarkdownContainer"]')) { hide(); obs.disconnect(); }
    });
    obs.observe(document.body, { childList: true, subtree: true });
    setTimeout(() => { hide(); obs.disconnect(); }, 180000);
  </script>
</body>
</html>
"""


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    files = collect_files()
    files[".streamlit/config.toml"] = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    # Escape "</" so inline source cannot close the <script> tag.
    files_json = json.dumps(files, ensure_ascii=False, indent=1).replace("</", "<\\/")
    html = (
        TEMPLATE.replace("__VERSION__", STLITE_VERSION)
        .replace("__CONFIG__", json.dumps(theme_config()))
        .replace("__FILES__", files_json)
    )
    (DOCS / "index.html").write_text(html, encoding="utf-8")
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Wrote {DOCS / 'index.html'} with {len(files)} files (stlite {STLITE_VERSION}).")


if __name__ == "__main__":
    main()
