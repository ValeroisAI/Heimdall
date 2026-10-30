// bifrost_logits: verilen tokenleri parça parça (prefill + tek token) işler, her konumun logitlerini ikili dosyaya yazar.
//   bifrost_logits model.gguf tokens.bin out.bin "5,1,13,1,1,9,30"   (parça boyutları; kalan tokenler tek tek)
// tokens.bin: int32 dizisi. out.bin: float32 [n_tokens, n_vocab].
// Ortam: NGL = GPU'ya konacak katman sayısı (varsayılan 0 = CPU), N_CTX = bağlam boyu (varsayılan token sayısı + 16).
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "kullanım: %s model.gguf tokens.bin out.bin [parçalar]\n", argv[0]); return 1; }
    std::vector<int> parts;
    if (argc > 4) { std::string s = argv[4]; size_t p = 0; while (p < s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); parts.push_back(std::stoi(s.substr(p, q - p))); p = q + 1; } }
    FILE * f = fopen(argv[2], "rb"); if (!f) return 1;
    std::vector<llama_token> toks; int32_t t; while (fread(&t, 4, 1, f) == 1) toks.push_back(t); fclose(f);

    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = getenv("NGL") ? atoi(getenv("NGL")) : 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "model yüklenemedi\n"); return 1; }
    auto cp = llama_context_default_params();
    cp.n_ctx = getenv("N_CTX") ? atoi(getenv("N_CTX")) : toks.size() + 16; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = 1;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "bağlam kurulamadı\n"); return 1; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    FILE * o = fopen(argv[3], "wb");
    size_t pos = 0, pi = 0;
    while (pos < toks.size()) {
        int n = pi < parts.size() ? parts[pi++] : 1;
        if (pos + n > toks.size()) n = toks.size() - pos;
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) { b.token[i] = toks[pos + i]; b.pos[i] = pos + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1; }
        b.n_tokens = n;
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode hatası (konum %zu)\n", pos); return 1; }
        for (int i = 0; i < n; ++i) fwrite(llama_get_logits_ith(ctx, i), 4, n_vocab, o);
        llama_batch_free(b);
        pos += n;
    }
    fclose(o);
    fprintf(stderr, "tamam: %zu token, vocab %d\n", toks.size(), n_vocab);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
