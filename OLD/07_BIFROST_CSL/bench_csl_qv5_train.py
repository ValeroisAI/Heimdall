"""Measure real CSL-QV5 training throughput on the target GPU; no synthetic speed claims."""

import argparse
import time

import torch
import torch.nn.functional as F

from valerois_csl_qv5 import ValeroisCSLQv5LM


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=32_768)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark needs a CUDA/ROCm PyTorch GPU build.")
    device = torch.device("cuda")  # PyTorch deliberately calls the ROCm device CUDA too.
    dtype = torch.bfloat16
    model = ValeroisCSLQv5LM(
        vocab_size=args.vocab_size,
        dim=args.dim,
        num_layers=args.layers,
        chunk_size=args.chunk_size,
    ).to(device=device, dtype=dtype)
    if args.compile:
        model = torch.compile(model, dynamic=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    inputs = torch.randint(args.vocab_size, (args.batch_size, args.seq_len + 1), device=device)

    def train_step() -> float:
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs[:, :-1])
        loss = F.cross_entropy(logits.float().reshape(-1, args.vocab_size), inputs[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        return loss.item()

    for _ in range(args.warmup):
        train_step()
    synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(args.steps):
        loss = train_step()
    synchronize()
    elapsed = time.perf_counter() - start

    tokens = args.steps * args.batch_size * args.seq_len
    params = sum(p.numel() for p in model.parameters())
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"parameters: {params / 1e6:.1f}M")
    print(f"final loss: {loss:.4f}")
    print(f"throughput: {tokens / elapsed:,.0f} tokens/s")
    print(f"step time: {elapsed * 1000 / args.steps:.1f} ms")
    print(f"peak allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
