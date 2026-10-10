"""Standalone, escaped HTML comparison; no server or frontend dependencies."""

from html import escape


def render_comparison(data: dict) -> str:
    columns = [("Японский оригинал", data["source"], "ja")]
    for candidate in data["candidates"]:
        info = candidate["status"]
        runners = ", ".join(f"{r['name']} / {r['model'] or 'default'}" for r in info["runners"])
        label = f"{candidate['run_id']} · {runners} · {info['workflow']}"
        if not candidate["complete"]:
            label += " · НЕ ЗАВЕРШЁН"
        columns.append((label, candidate["translation"], "ru"))
    sections = "".join(
        f'<section><h2>{escape(label)}</h2><article lang="{language}">'
        f"{escape(text)}</article></section>"
        for label, text, language in columns
    )
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>Глава {data["chapter"]} · Сравнение переводов</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f1e9;color:#262822;font-family:system-ui,sans-serif}}
header{{padding:24px 32px;border-bottom:1px solid #c8c9ba}}h1{{font-size:24px;margin:0 0 8px}}
header p{{margin:0;color:#555d4e}}main{{display:grid;grid-template-columns:repeat({len(columns)},minmax(300px,1fr));gap:16px;padding:20px;align-items:start}}
section{{background:#fffef9;border:1px solid #d5d8c8;border-radius:8px;min-width:0}}
h2{{font-size:13px;overflow-wrap:anywhere;margin:0;padding:16px;border-bottom:1px solid #e1e3d8;color:#3c543e}}
article{{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.85;font-family:Georgia,serif;font-size:17px;padding:22px}}
@media(max-width:700px){{main{{grid-template-columns:1fr}}header{{padding:20px}}}}
</style></head><body><header><h1>Глава {data["chapter"]} — {escape(data["title"])}</h1>
<p>Один исходник · независимые запуски · ручное сравнение</p></header><main>{sections}</main></body></html>"""
