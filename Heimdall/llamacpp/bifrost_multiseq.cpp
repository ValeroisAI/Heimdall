// bifrost_multiseq: iki dizi aynı anda (B=2) ↔ tek tek (B=1); logitler aynı olmalı.  NGL=99 bifrost_multiseq model.gguf
#include "llama.h"
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
    return mx < 1e-3 ? 0 : 1;
}
