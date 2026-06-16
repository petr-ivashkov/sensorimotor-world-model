# Embedding analysis (Fig. 6, Fig. 10, Table 2)

Exports frozen encoder embeddings for the seed-0 trained models and analyses their geometry
and physical-state content.

```text
4 environments x 3 methods (inverse, sigreg, forward_only) = 12 exports
```

Run the `train` stage (seed 0) first. The exporter reads the randomized transition splits
`*_train_randomized_25k.h5` / `*_eval_randomized_5k.h5` from `data/external/`
(see the repo README and `scripts/`).

## Export embeddings

```bash
cd experiments/embedding_analysis
python generate_config.py --overwrite
python export_embeddings.py --config config.yaml --verify-only   # check paths

# one export at a time: run.sh <env> <method>
./run.sh reacher inverse
```

Outputs are written to `experiments/embedding_analysis/outputs/<env>/<method>/`
(`train_embeddings.pt`, `eval_embeddings.pt`, `metadata.json`). Each `*_embeddings.pt` holds
the frozen embeddings plus the non-pixel arrays (actions, physical-state metadata) needed for
probing.

## Figures and table

```bash
jupyter notebook analyze_learned_representations.ipynb   # Fig. 6 (ours) + Fig. 10 (SIGReg)
jupyter notebook physical_quantity_probe_table.ipynb     # Table 2 (linear / MLP probe R^2)
```

Both notebooks read from `outputs/` and display all figures and tables inline.