// bifrost_multiseq: (1) iki dizi aynı anda (B=2) ↔ tek tek (B=1); (2) seq_cp ile kopyalanan dizi farklı devam eder;
// (3) silinen dizinin hücresinde yeni dizi. Hepsinde logitler tek başına çalıştırmayla aynı olmalı.  NGL=99 bifrost_multiseq model.gguf
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
static llama_context * mk(llama_model * m, int nseq) {
    auto cp = llama_context_default_params(); cp.n_ctx = 1024; cp.n_seq_max = nseq; cp.n_batch = 512; cp.n_ubatch = 512;
    return llama_init_from_model(m, cp);
}
int main(int argc, char ** argv) {
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = getenv("NGL") ? atoi(getenv("NGL")) : 0;
    llama_model * m = llama_model_load_from_file(argv[1], mp);
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(m));
    const int P = 40, N = 120;
    std::vector<std::vector<int>> tok(2);
    for (int s = 0; s < 2; ++s) for (int i = 0; i < P + N; ++i) tok[s].push_back(100 + (i * (7 + 6 * s) + 13 * s) % 3000);
    std::vector<std::vector<float>> ref(2), par(2);
    for (int s = 0; s < 2; ++s) {   // alone
        llama_context * c = mk(m, 1);
        int pos = 0;
        while (pos < P + N) {
            int n = pos == 0 ? P : 1;
            llama_batch b = llama_batch_init(n, 0, 1);
            for (int i = 0; i < n; ++i) { b.token[i] = tok[s][pos + i]; b.pos[i] = pos + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1; }
            b.n_tokens = n; if (llama_decode(c, b)) { fprintf(stderr, "decode err\n"); return 1; }
            for (int i = 0; i < n; ++i) { float * l = llama_get_logits_ith(c, i); ref[s].insert(ref[s].end(), l, l + nv); }
            llama_batch_free(b); pos += n;
        }
        llama_free(c);
    }
    {   // together
        llama_context * c = mk(m, 2);
        int pos = 0;
        while (pos < P + N) {
            int n = pos == 0 ? P : 1;
            llama_batch b = llama_batch_init(2 * n, 0, 1);
            for (int s = 0; s < 2; ++s) for (int i = 0; i < n; ++i) { int j = s * n + i; b.token[j] = tok[s][pos + i]; b.pos[j] = pos + i; b.n_seq_id[j] = 1; b.seq_id[j][0] = s; b.logits[j] = 1; }
            b.n_tokens = 2 * n; if (llama_decode(c, b)) { fprintf(stderr, "decode err\n"); return 1; }
            for (int s = 0; s < 2; ++s) for (int i = 0; i < n; ++i) { float * l = llama_get_logits_ith(c, s * n + i); par[s].insert(par[s].end(), l, l + nv); }
            llama_batch_free(b); pos += n;
        }
        llama_free(c);
    }
    double mx = 0;
    for (int s = 0; s < 2; ++s) for (size_t i = 0; i < ref[s].size(); ++i) mx = fmax(mx, fabs(ref[s][i] - par[s][i]));
    printf("B=2 vs B=1 max logit diff: %.3e\n", mx);

    // tokens fed one by one (or as a prefill chunk) into seq `s` of context `c`; returns the logits of each token
    auto run = [&](llama_context * c, int s, const std::vector<int> & t, int pos0, int chunk) {
        std::vector<float> out;
        for (size_t a = 0; a < t.size();) {
            int n = std::min<int>(chunk, t.size() - a);
            llama_batch b = llama_batch_init(n, 0, 1);
            for (int i = 0; i < n; ++i) { b.token[i] = t[a + i]; b.pos[i] = pos0 + a + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = s; b.logits[i] = 1; }
            b.n_tokens = n; if (llama_decode(c, b)) { fprintf(stderr, "decode err\n"); exit(1); }
            for (int i = 0; i < n; ++i) { float * l = llama_get_logits_ith(c, i); out.insert(out.end(), l, l + nv); }
            llama_batch_free(b); a += n;
        }
        return out;
    };
    auto maxdiff = [](const std::vector<float> & a, const std::vector<float> & b) {
        double m = 0; for (size_t i = 0; i < a.size(); ++i) m = fmax(m, fabs(a[i] - b[i])); return m;
    };
    std::vector<int> pre(tok[0].begin(), tok[0].begin() + P + 30), cont(tok[1].begin(), tok[1].begin() + 90);
    std::vector<int> all = pre; all.insert(all.end(), cont.begin(), cont.end());
    std::vector<float> want;
    // reference: one fresh sequence, chunked the same way (prefill P, then single tokens): on GPUs prefill and decode
    // use different kernels, so a different chunking alone changes logits at the 1e-2 level
    auto fresh = [&](llama_context * c, int s) {
        std::vector<float> a = run(c, s, std::vector<int>(all.begin(), all.begin() + P), 0, P);
        std::vector<float> b = run(c, s, std::vector<int>(all.begin() + P, all.end()), P, 1);
        return std::vector<float>(b.end() - cont.size() * nv, b.end());
    };
    {
        llama_context * c = mk(m, 1);
        want = fresh(c, 0);
        llama_free(c);
    }
    double mcp, mrm;
    {   // seq 0 runs the prefix, is copied to seq 1, seq 0 continues differently, seq 1 continues like the reference
        llama_context * c = mk(m, 2);
        run(c, 0, std::vector<int>(pre.begin(), pre.begin() + P), 0, P);
        run(c, 0, std::vector<int>(pre.begin() + P, pre.end()), P, 1);
        llama_memory_seq_cp(llama_get_memory(c), 0, 1, -1, -1);
        run(c, 0, std::vector<int>(tok[0].begin(), tok[0].begin() + 20), pre.size(), 1);
        mcp = maxdiff(run(c, 1, cont, pre.size(), 1), want);
        // remove seq 1 and start a new sequence there: it must not see anything of the old one
        llama_memory_seq_rm(llama_get_memory(c), 1, -1, -1);
        mrm = maxdiff(fresh(c, 1), want);
        llama_free(c);
    }
    printf("seq_cp continuation max logit diff: %.3e\n", mcp);
    printf("reused cell after seq_rm max logit diff: %.3e\n", mrm);
    return mx < 1e-3 && mcp < 1e-3 && mrm < 1e-3 ? 0 : 1;
}
