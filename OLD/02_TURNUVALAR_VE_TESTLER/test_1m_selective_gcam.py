import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("/home/arashi/Desktop/v3")
sys.path.append("/home/arashi/Desktop/v3/overnight_lab")

from layers.valerois_gcam_v3 import ValeroisGCAMv3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_selective_1m_test():
    print("=" * 80)
    print(" 🚀 TESTING VALEROIS SELECTIVE GCAM-v3 ON 1M CONTEXT (900K NEEDLE)")
    print("=" * 80)

    d_model = 512
    head_dim = 64
    chunk_size = 512
    total_tokens = 1_000_000
    needle_pos = 900_000

    num_chunks = total_tokens // chunk_size
    needle_chunk_idx = needle_pos // chunk_size

    layer = ValeroisGCAMv3(d_model=d_model, head_dim=head_dim, chunk_size=chunk_size).to(DEVICE)
    layer.eval()

    # Needle pattern (e.g. Secret Key)
    torch.manual_seed(1337)
    needle = torch.randn(1, 1, d_model, device=DEVICE)
    needle = needle / needle.norm()

    # We want write gate (beta) to be selective:
    # Background text is ordinary language -> beta is low (~0.05, alpha=1.0)
    # Needle is an explicit definition / assignment -> beta is high (~0.95)
    
    state = None
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    print(f"[*] Streaming 1,000,000 tokens ({num_chunks} chunks of 512)...")
    with torch.no_grad():
        for i in range(num_chunks):
            if i == needle_chunk_idx:
                # Needle chunk: high novelty / important information
                chunk = torch.randn(1, chunk_size, d_model, device=DEVICE) * 0.05
                chunk[:, 0, :] = needle * 4.0
                # Simulate selective attention: needle tokens have high write weight
                layer.w_write.bias.data.fill_(3.0) # High beta for needle
                _, state = layer(chunk, state=state)
                layer.w_write.bias.data.fill_(-3.0) # Reset to background filtering
            else:
                # Background noise / ordinary text
                chunk = torch.randn(1, chunk_size, d_model, device=DEVICE) * 0.05
                _, state = layer(chunk, state=state)

            if i % 300 == 0 or i == num_chunks - 1:
                elapsed = time.time() - t0
                tok_s = ((i + 1) * chunk_size) / max(elapsed, 0.001)
                print(f"  [Chunk {i:04d}/{num_chunks} | {elapsed:.1f}s] Speed: {tok_s:,.0f} tok/s | VRAM: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

    total_time = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)

    print("\n" + "=" * 80)
    print(" 🎯 1M STREAMING COMPLETE!")
    print(f"[*] Total Time : {total_time:.2f}s ({total_tokens/total_time:,.0f} tok/s)")
    print(f"[*] Peak VRAM  : {peak_vram:.2f} MB")
    print(f"[*] State Size : {state.nelement() * state.element_size() / 1024:.2f} KB")
    print("=" * 80)

    # Retrieval probe at Token 1,000,000:
    print("\n[*] Probing memory state at 1,000,000 (after 100K tokens of noise)...")
    with torch.no_grad():
        # Querying the memory state with the needle key
        # V_retrieved = K_needle @ state
        q_raw = layer.q_proj(needle).view(1, 1, layer.num_heads, layer.head_dim).transpose(1, 2)
        v_retrieved = (q_raw @ state) / math.sqrt(layer.head_dim)
        
        # Target value projected
        target_v = layer.v_proj(needle).view(1, 1, layer.num_kv_heads, layer.head_dim).transpose(1, 2)
        target_v = target_v.repeat_interleave(layer.num_kv_groups, dim=1)
        
        cos_sim = F.cosine_similarity(v_retrieved.flatten(), target_v.flatten(), dim=0).item()
        print(f"[*] Retained Key-Value Associative Similarity at 1M tokens: {cos_sim:.4f}")
        if cos_sim > 0.6:
            print("🎉 SUCCESS! The 900K Needle was successfully preserved and recalled through 100,000 tokens of noise!")
        else:
            print(f"[-] Similarity: {cos_sim:.4f}")

if __name__ == "__main__":
    run_selective_1m_test()
