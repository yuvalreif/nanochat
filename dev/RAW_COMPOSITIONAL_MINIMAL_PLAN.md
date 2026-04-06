# Raw Compositional Minimal Plan

Baseline:
- Repo: `karpathy/nanochat`
- Commit: `a445144d3905c6845fda2d3cab8e63248a70cd32`
- Branch for this work: `codex/raw-compositional-plan`

Goal:
- Add a benchmark-friendly version of the compositional token approach.
- Stay close to the original nanochat control flow.
- Measure success partly by how few files and concepts are added relative to upstream.

Core constraint:
- Use the raw parquet training path.
- Do not rely on a pretokenized training-only pipeline.
- Do not import the current `vocab_diet/*` subsystem into this repo.

## Minimal runtime surface

Keep the implementation centered on these upstream files:
- `nanochat/tokenizer.py`
- `nanochat/dataloader.py`
- `nanochat/gpt.py`
- `scripts/base_train.py`
- `scripts/base_eval.py`
- `nanochat/core_eval.py`

Add at most one small helper module for compositional metadata / sequence matching / surface reconstruction.

## Runtime contract

Tokenizer-time:
- Keep the existing Rust BPE tokenizer as the base tokenizer.
- Load one compact metadata artifact next to the tokenizer.
- Metadata contains:
  - modifier groups and sizes
  - per-token direct mappings for simple case/space/punctuation cases
  - a small sequence trie for multi-token collapses such as attached function words or punctuation
  - inverse reconstruction data for decode / sampling

Raw dataloader:
- Continue reading raw parquet documents.
- Replace plain `encode(...)` usage with an optional compositional encoding path.
- Yield either:
  - baseline `(x, y)`, or
  - compositional `((x_ids, x_mods), (y_ids, y_mods))`

Model:
- Keep one GPT trunk.
- Input embedding becomes `token_embed + summed_modifier_embed`.
- Base-token logits remain the primary LM head.
- Add a small conditioned modifier head for target modifier prediction.
- Loss becomes:
  - base-token CE
  - plus per-group modifier CE conditioned on the target base token

Eval / sampling:
- Prompt encoding must use the same compositional tokenizer path as train.
- Decoding sampled outputs requires a small surface reconstructor from `(base_id, modifiers)` back to text.
- Keep baseline eval flow; only swap tokenization / decoding where needed.

## Non-goals

Do not bring over:
- bundle/cache orchestration
- raw dual-stream vs pretokenized dual-stream mode matrix
- Rust fused runtime backend
- tokenizer analysis dashboards
- extended-vocab auxiliary head
- ambiguous modifier-loss variants
- type-value embeddings
- multiple alternative conditioning architectures

## Complexity budget

Target:
- 1 new helper module
- 5 to 6 edited upstream files
- no new training entrypoint if avoidable
- no new parallel data format required for normal training

If a design requires more than this, it should be treated as suspect and simplified.

## First implementation slice

1. Add a compact compositional metadata loader in the tokenizer path.
2. Add a compositional raw dataloader path that preserves baseline behavior when disabled.
3. Extend GPT with modifier embeddings and modifier loss.
4. Thread the new batch shape through `base_train`.
5. Add the minimum encode/decode support needed for `base_eval` and `core_eval`.
