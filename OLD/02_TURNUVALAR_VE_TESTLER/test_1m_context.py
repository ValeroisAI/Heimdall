import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("/home/arashi/Desktop/v3")
sys.path.append("/home/arashi/Desktop/v3/overnight_lab")

from layers.valerois_gcam_v2 import ValeroisGCAMv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_1m_needle_test():
    print("=" * 80)
    print(" 🧪 RUNNING 1,000,000 TOKEN CONTEXT NEEDLE-IN-A-HAYSTACK TEST")
    print(f"[*] Device: {DEVICE}")
    print("=" * 80)

    d_model = 512
    head_dim = 64
    chunk_size = 512
    total_tokens = 1_000_000
    needle_pos = 900_000 # 900K

    num_chunks = total_tokens // chunk_size
    needle_chunk_idx = needle_pos // chunk_size

    print(f"[*] Total Tokens    : {total_tokens:,}")
    print(f"[*] Chunk Size      : {chunk_size}")
    print(f"[*] Total Chunks    : {num_chunks:,}")
    print(f"[*] Needle Chunk    : {needle_chunk_idx} (Token {needle_pos:,})")

    # Load layer
    layer = ValeroisGCAMv2(d_model=d_model, head_dim=head_dim, chunk_size=chunk_size).to(DEVICE)
    layer.eval()

    # Create a distinct Needle key-value pattern
    torch.manual_seed(42)
    needle_vector = torch.randn(1, 1, d_model, device=DEVICE)
    needle_vector = needle_vector / needle_vector.norm()

    # Query vector looking for the needle
    query_vector = needle_vector.clone()

    state = None
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    # Stream chunks
    retrieval_similarities = []

    print("[*] Streaming 1M tokens chunk by chunk...")
    with torch.no_grad():
        for chunk_idx in range(num_chunks):
            # Generate random background noise (haystack)
            if chunk_idx == needle_chunk_idx:
                # Insert needle at the beginning of this chunk
                chunk = torch.randn(1, chunk_size, d_model, device=DEVICE) * 0.1
                chunk[:, 0, :] = needle_vector * 5.0 # Strong needle signal
                print(f"  [+] Needle inserted at chunk {chunk_idx} (Token {chunk_idx * chunk_size:,})")
            else:
                chunk = torch.randn(1, chunk_size, d_model, device=DEVICE) * 0.1

            # Forward chunk
            out, state = layer(chunk, state=state)

            if chunk_idx % 250 == 0 or chunk_idx == num_chunks - 1:
                elapsed = time.time() - t0
                tok_s = ((chunk_idx + 1) * chunk_size) / max(elapsed, 0.001)
                vram_mb = torch.cuda.memory_allocated() / (1024**2)
                state_mb = (state.nelement() * state.element_size()) / (1024**2)
                print(f"  [Chunk {chunk_idx:04d}/{num_chunks} | {elapsed:.1f}s] Speed: {tok_s:,.0f} tok/s | VRAM: {vram_mb:.1f} MB | State: {state_mb:.3f} MB")

    total_time = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    state_size_kb = (state.nelement() * state.element_size()) / 1024

    print("\n" + "=" * 80)
    print(" 🎯 1,000,000 TOKEN STREAMING COMPLETED!")
    print(f"[*] Elapsed Time     : {total_time:.2f} seconds ({total_tokens/total_time:,.0f} tok/s)")
    print(f"[*] Peak GPU VRAM    : {peak_vram:.2f} MB")
    print(f"[*] O(1) State Memory: {state_size_kb:.2f} KB (Katman Başına)")
    print("=" * 80)

    # Now ask question at position 1,000,000:
    print("\n[*] Asking question at token 1,000,000 using final recurrent state...")
    with torch.no_grad():
        query_chunk = query_vector.repeat(1, chunk_size, 1)
        out_query, _ = layer(query_chunk, state=state)
        # Check cosine similarity between output and needle vector
        sim = F.cosine_similarity(out_query[:, 0, :], needle_vector[:, 0, :]).item()
        print(f"[*] Cosine Similarity between Query Output and 900K Needle: {sim:.4f}")
        
    return total_time, peak_vram, state_size_kb, sim

if __name__ == "__main__":
    run_1m_needle_test()
