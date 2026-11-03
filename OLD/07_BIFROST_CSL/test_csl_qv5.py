"""Smoke, causality, and accounting checks for valerois_csl_qv5.py."""

import torch

from valerois_csl_qv5 import ValeroisCSLQv5LM


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ValeroisCSLQv5LM(vocab_size=257, dim=128, num_layers=3, chunk_size=8, num_memory_heads=2, memory_head_dim=32).to(device)
    model.eval()
    ids = torch.randint(0, 257, (2, 29), device=device)

    with torch.no_grad():
        full = model(ids)
        changed = ids.clone()
        changed[:, 17:] = torch.randint(0, 257, changed[:, 17:].shape, device=device)
        causal = model(changed)

        # Later inputs must never alter earlier logits.
        torch.testing.assert_close(full[:, :17], causal[:, :17], rtol=0.0, atol=2e-5)

        # step() must match the parallel forward path for a complete chunk only.
        # The final incomplete chunk is not in memory yet by design.
        state = model.init_state(ids.size(0), device)
        stepped = []
        for token in ids.unbind(dim=1):
            logit, state = model.step(token, state)
            stepped.append(logit)
        stepped = torch.stack(stepped, dim=1)
        torch.testing.assert_close(full, stepped, rtol=2e-4, atol=2e-4)

    params = sum(p.numel() for p in model.parameters())
    cost = model.blocks[0].cost(seq_len=32_768, batch_size=1)
    print(f"CSL-QV5 smoke test passed on {device}.")
    print(f"parameters: {params:,}")
    print(f"per-layer memory attention score elements: {cost.memory_attention:,}")
    print(f"per-layer inference KV elements: {cost.memory_kv_elements:,}")


if __name__ == "__main__":
    main()
