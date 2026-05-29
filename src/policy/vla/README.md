# `src/policy/vla/`

Placeholder for vision-language-action policy families (OpenVLA, RT-2, π0, ...).

When adding a new family here, mirror `src/policy/cosmos/`:
- `model.py` — backbone + action head + value head
- `dataset.py` — adapter from our annotation schema to whatever tensor shapes the backbone expects
- `trainer.py` — training loop
- `REFERENCES.md` — upstream files we ported, commit hashes, licenses

Re-use anything in `src/policy/common/` (goal space, 5D action representation, annotation iterators, reward) rather than duplicating it.
