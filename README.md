# sloptimizers

Between the modded-nanogpt speedrun's optimization track and other optimizer competitions, there's a lot of work on making optimizers faster, but the field still lacks a good metric for how sample efficient an optimizer is. We run some simple tests to see how different optimizers behave when you train on a fixed amount of data.

Kaitlyn Wang, Alan Xiao, Revant Kasichainula

This is still ongoing.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/val_loss_vs_size_dark.png">
  <img src="assets/val_loss_vs_size.png" alt="Best validation loss vs. model size for different optimizers">
</picture>

## What we've tried so far

- Adam variants (AdamW, AdamHD)
- L1 / Huber weight decay
- Second-order and preconditioned optimizers (SOAP, Muon, layer-wise Gauss-Newton)
- Norm-constrained updates (Hyperball)
- SAM and Nexus

## Code

PyTorch implementations of the optimizers are in `sloptimizers/`, with a tiny training example in `examples/`.

```bash
pip install -e .
python examples/train_tiny.py --optimizer muoneq-r
```

## License

MIT
