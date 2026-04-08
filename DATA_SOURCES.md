# Data Sources

This environment is intentionally deterministic and ships with a bundled mini-dataset so the grader does not depend on external APIs or large downloads.

## What we used

- USDA FoodData Central as the grounding reference for ingredient naming conventions and realistic unit choices (`g`, `ml`, `piece`).
- Public recipe-corpus research as inspiration for which dish families are common enough to model, without ingesting copyrighted long-form instructions.

## What we did not ship

- No remote dataset fetches at runtime.
- No third-party recipe text copied into the repo.
- No dependency on a live supplier or menu API.

## Modeling choice

We authored compact recipe graphs ourselves and paired them with explicit ingredient quantities, costs, shelf-life windows, and substitutions so the environment stays fast, reproducible, and submission-safe.

