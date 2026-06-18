# Borzoi K562 Label Schema

This repository uses `q` as a historical shorthand for a Mini-Borzoi teacher
reference gene score. It is not a statistical q-value, FDR, or multiple-testing
quantity.

## Current v1 Labels

For one gene/window, the v1 scalar is:

```text
q = log1p(mean selected K562 RNA-seq teacher coverage over mapped bins)
```

The selected bins are usually gene-body bins from the Mini-Borzoi output core.
The teacher prediction is generated from official Mini-Borzoi K562 assets and
then aggregated with `BorzoiOutputMapper`.

Backward-compatible aliases:

- `q`
- `y_ref`
- `teacher_ref_expr`
- `borzoi_ref_score`
- `q_teacher`

The v1 reference cache keeps these columns:

- `q_teacher`: the final selected scalar used by old student training
- `q_teacher_raw_mean`: raw mean over selected bins before `log1p`
- `q_teacher_raw_sum`: raw sum over selected bins
- `q_teacher_log1p_mean`: `log1p(q_teacher_raw_mean)`
- `aggregation_mode`: aggregation recipe, usually `gene_body_log1p_mean`
- `target_index`: selected Mini-Borzoi target index

Old code reading `q_teacher` remains valid.

## Current Delta Labels

For one variant-gene pair:

```text
delta_q = q_alt - q_ref
```

The current delta cache stores this as:

- `q_ref_teacher`
- `q_alt_teacher`
- `delta_teacher`
- `abs_delta_teacher`
- `sign_teacher`

The current synthetic delta labels are teacher-fidelity labels only. They are
not biological validation, eQTL labels, or clinical effect labels.

## Planned v2 Rich Labels

Pocket-Borzoi v2 keeps the v1 scalar but adds richer teacher supervision:

- `q_old`: old v1 scalar for compatibility
- `primary_0_q_fold0`
- `primary_0_q_fold1`
- `primary_0_q_mean`
- `primary_0_q_std`
- `aux_<k>_q_mean`
- `profile_pca_*`: compressed primary binned-profile labels
- `aux_pca_*`: compressed auxiliary target labels
- `middle_proj_*`: target-relevant compressed teacher middle/head-input labels

The inference target remains the primary reference score. Auxiliary labels are
training-only distillation signals.
