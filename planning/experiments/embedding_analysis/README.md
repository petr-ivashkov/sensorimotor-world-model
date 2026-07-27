# Embedding analysis (Fig. 6, Fig. 10, Table 2)

Exports frozen encoder embeddings for the seed-0 trained models and analyses their geometry
and physical-state content.

The generated config contains the seed-0 inverse, SIGReg, forward-only, and
IDR+SIGReg runs for all four environments. `export.sub` queues only the four new
IDR+SIGReg exports; existing embeddings are not resubmitted.

Run the `train` stage (seed 0) first. The exporter reads the randomized transition splits
`*_train_randomized_25k.h5` / `*_eval_randomized_5k.h5` from `data/external/`
(see the repo README and `scripts/`).

## Export embeddings

```bash
cd experiments/embedding_analysis
python generate_config.py --overwrite
python export_embeddings.py --config config.yaml --verify-only --method idr_sigreg

# one new export locally: run.sh <env> idr_sigreg
./run.sh reacher idr_sigreg

# all four new exports on Condor
condor_submit_bid 100 export.sub
```

Outputs are written to `experiments/embedding_analysis/outputs/<env>/<method>/`
(`train_embeddings.pt`, `eval_embeddings.pt`, `metadata.json`). Each `*_embeddings.pt` holds
the frozen embeddings plus the non-pixel arrays (actions, physical-state metadata) needed for
probing.

## Figures and table

```bash
jupyter notebook analyze_learned_representations.ipynb
jupyter notebook physical_quantity_probe_table.ipynb
```

Both notebooks read from `outputs/` and apply the same PCA and physical-probe
protocol to IDR, SIGReg, forward-only, and IDR+SIGReg.
