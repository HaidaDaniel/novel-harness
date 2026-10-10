# Local Qwen benchmark notes

The saved local Qwen comparison recorded B, R, PR, and T sampler/reasoning
profiles with 800-character chunks and 400-character overlap. The portable
example replaces private endpoint, model path, and host details with loopback
and `YOUR_QWEN_MODEL_ID`.

| Profile | Thinking | Temperature | Top-p | Min-p | Presence penalty | Output cap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B | off | 0.2 | 0.95 | 0.05 | 0.0 | 32,000 |
| R | on, xhigh | 0.2 | 0.95 | 0.05 | 0.0 | 65,536 |
| PR | on, xhigh | 0.7 | 0.8 | 0.0 | 1.5 | 65,536 |
| T | on, xhigh | 1.0 | 0.95 | 0.0 | 0.0 | 65,536 |

On the saved comparison sample, R scored above B while B used fewer completion
tokens and was the practical draft profile. This is profile evidence for this
harness, not a universal ranking or a book-wide publication-quality floor.
