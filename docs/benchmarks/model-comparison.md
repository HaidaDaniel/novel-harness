# Ten-model translation benchmark

This public summary preserves aggregate benchmark conclusions without shipping
source chapters, full translations, scorecard text, runtime traces, or provider
credentials.

## Protocol

- five source chapters;
- 10 purposively selected source-aligned fragments (two per chapter);
- 100 saved candidate translations across ten variants;
- anonymized labels during scoring;
- AI evaluation, not independent human evaluation;
- Codex/Luna came from a saved run with different chunk geometry;
- provider prices frozen at the experiment date.

The selected fragments were diagnostic rather than random or held-out. Scores and
defect counts are observations from this scorecard, not complete-book claims.

## Aggregate result

See [`summary.json`](summary.json). Costs are USD for the five-chapter saved run
where known; `null` means the saved evidence did not support a comparable cost.

| Variant | Quality | Naturalness | Preferred | Score <=2 | Major | Minor | Cost USD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM-5.3-flash | 3.8 | 3.5 | 2 | 0 | 1 | 4 | 0.1836 |
| Codex / GPT-6 Luna | 3.7 | 4.7 | 3 | 0 | 4 | 2 | null |
| DeepSeek v4-pro | 3.6 | 3.9 | 2 | 2 | 4 | 0 | 0.3850 |
| DeepSeek v4.1-flash | 3.5 | 3.1 | 1 | 1 | 2 | 4 | 0.1212 |
| Longcat 2.5 preview free | 3.2 | 3.5 | 1 | 2 | 4 | 4 | 0.0000 |
| MiMo v2.6-flash | 3.0 | 2.8 | 0 | 2 | 7 | 5 | 0.0275 |
| Qwen3.8-27B R / xhigh | 3.0 | 2.8 | 0 | 2 | 6 | 2 | null |
| Qwen3.8-flash | 2.9 | 3.6 | 0 | 5 | 7 | 3 | 0.1558 |
| Space Bunny | 2.7 | 3.5 | 1 | 6 | 11 | 6 | 0.0000 |
| Qwen3.8-27B B / no-thinking | 2.3 | 2.6 | 0 | 7 | 13 | 5 | null |

Latency and cost are not universal leaderboard measurements. Small score
differences are not statistical significance claims, and no product default is
promoted by this document.
