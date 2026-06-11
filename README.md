# Pocket Borzoi

Pocket Borzoi is a research project about distilling the SNP perturbation behavior of large sequence-to-function models into tiny local experts. The main target is not to retrain a new personal-genomics model from cohort RNA-seq. Instead, we want a compact model that can quickly approximate how a large teacher such as Borzoi responds to a ref/alt sequence change for a specific query like tissue, track, gene, or endpoint.

## Project Goal

We care about the counterfactual effect of a variant on a teacher model:

- input: a reference sequence, an alternate sequence, and a query
- teacher: a large model such as Borzoi or AlphaGenome
- output: the predicted delta induced by the SNP or small variant

In short, this project aims to distill the variant-effect operator rather than distill the full teacher.

## Core Framing

The guiding story for this repository is:

1. Large genomic foundation models are powerful but expensive to run.
2. Many practical users only need a narrow query, such as one tissue or one gene-level endpoint.
3. That narrow query should be compressible into a small expert model that runs locally.

This suggests a query-conditioned or tissue-specific expert bank:

- one expert per tissue, endpoint, or task slice
- a lightweight router that selects the right expert
- optional fallback to the full teacher when uncertainty is high

## What This Project Is

- Distillation of teacher SNP-effect predictions
- Focus on compact, local, query-specific inference
- A systems and modeling story around expert routing and specialization
- A practical path toward fast local variant scoring

## What This Project Is Not

- Not a general reimplementation of Borzoi
- Not an individual-level expression prediction project in the SAGE-net sense
- Not a cohort-specific fine-tuning paper on personal genomes
- Not a broad multi-modal framework paper in the first iteration

Those directions may be interesting later, but they are intentionally out of scope for the first paper.

## Working Hypothesis

For many biologically meaningful queries, the expensive teacher response

`delta = q(T(x_alt)) - q(T(x_ref))`

can be approximated well by a much smaller student that only models the local counterfactual operator for that query.

## Proposed First Version

The first version of the project should stay narrow and executable:

- choose one main teacher: Borzoi
- optionally add AlphaGenome later as a secondary teacher
- focus on SNP or small variant perturbations
- start with a small set of tissues or endpoints rather than all tracks
- train compact experts on teacher-generated perturbation labels
- evaluate both fidelity to the teacher and usefulness on downstream biological validation

## Candidate Research Questions

- How small can a local expert be while preserving teacher-quality SNP effect estimates?
- When does a tissue-specific expert outperform a single shared student?
- What metadata or context is enough for accurate local variant-effect prediction?
- When should the system trust the expert versus defer to the full teacher?

## Early Experimental Plan

1. Define a clean query space.
   Start with tissue-specific or gene-specific SNP effect prediction instead of full teacher outputs.

2. Build a perturbation label bank.
   Use Borzoi to generate ref/alt deltas for curated sequence windows and selected queries.

3. Train small experts.
   Compare shared students, tissue-specific students, and routed expert-bank variants.

4. Evaluate fidelity.
   Measure correlation, ranking quality, calibration, and hard-example behavior against teacher outputs.

5. Evaluate biological relevance.
   Test whether distilled scores preserve enrichment or discrimination on eQTL- or disease-relevant variant sets.

6. Study systems tradeoffs.
   Report model size, latency, memory, and local usability relative to the teacher.

## Success Criteria

An initial submission would be compelling if we can show:

- strong agreement with Borzoi on selected variant-effect queries
- large reductions in runtime and model footprint
- evidence that specialized experts are better than a single tiny generic student
- a clean story that the method distills counterfactual variant effects, not full-sequence function prediction

## Immediate Repository Roadmap

- add project structure for data, labeling, training, and evaluation
- define the first teacher-query task configuration
- implement teacher perturbation data generation
- implement a minimal student baseline
- add expert-bank and routing experiments
- write reproducible evaluation scripts and experiment tracking

## Status

This repository is currently in the planning and bootstrap stage. The README is intended to serve as the project north star while we build the first experimental pipeline.
