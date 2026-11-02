// bifrost_passkey: uzun dolgu metnin içine gömülü 5 haneli parolayı birebir geri isteme testi (greedy).
//   NGL=99 bifrost_passkey model.gguf 8192,32768,65536 [derinlikler 0.1,0.5,0.9] [deneme 2]
// Her bağlam × derinlik × deneme için bir satır, sonunda doğruluk (sıkı: ilk sayı parola; esnek: cevapta geçiyor).
// Model parola görünene, sırasını bitirene ya da MAX_NEW (512) token'a kadar üretir.
// Ortam: NGL, CTK/CTV (KV tipi: f16, q8_0, q4_0), MAX_NEW.
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<double> parse(const char * s) {
    std::vector<double> v; std::string t = s; size_t p = 0;
    while (p < t.size()) { size_t q = t.find(',', p); if (q == std::string::npos) q = t.size(); v.push_back(atof(t.substr(p, q - p).c_str())); p = q + 1; }
    return v;
}

static ggml_type kv_type(const char * e) {
    const char * s = getenv(e);
    if (!s || !strcmp(s, "f16")) return GGML_TYPE_F16;
    if (!strcmp(s, "q8_0")) return GGML_TYPE_Q8_0;
    if (!strcmp(s, "q4_0")) return GGML_TYPE_Q4_0;
    fprintf(stderr, "bilinmeyen KV tipi %s\n", s); exit(1);
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "kullanım: %s model.gguf 8192,32768 [0.1,0.5,0.9] [2]\n", argv[0]); return 1; }
    std::vector<double> lens = parse(argv[2]), depths = parse(argc > 3 ? argv[3] : "0.1,0.5,0.9");
    const int trials = argc > 4 ? atoi(argv[4]) : 2;
    int max_len = 0; for (double l : lens) max_len = std::max(max_len, (int) l);

    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = getenv("NGL") ? atoi(getenv("NGL")) : 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto tok = [&](const std::string & s) {
        std::vector<llama_token> t(s.size() + 16);
        int n = llama_tokenize(vocab, s.c_str(), s.size(), t.data(), t.size(), false, false);
        t.resize(n); return t;
    };
    const std::string head = "There is an important piece of information hidden inside a lot of irrelevant text. Find it and "
                             "memorize it. I will quiz you about it at the end.\n";
    const std::string junk = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ";
    const std::vector<llama_token> t_head = tok(head), t_junk = tok(junk);

    auto cp = llama_context_default_params();
    cp.n_ctx = max_len + 256; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = 1;
    cp.type_k = kv_type("CTK"); cp.type_v = kv_type("CTV");
    if (cp.type_k != GGML_TYPE_F16 || cp.type_v != GGML_TYPE_F16) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "bağlam kurulamadı (bellek?)\n"); return 1; }
    const int n_vocab = llama_vocab_n_tokens(vocab);

    srand(1234);
    int ok_all = 0, found_all = 0, n_all = 0;
    const int max_new = getenv("MAX_NEW") ? atoi(getenv("MAX_NEW")) : 512;
    for (double L : lens) for (double d : depths) for (int tr = 0; tr < trials; ++tr) {
        const int key = 10000 + rand() % 90000;
        const std::string needle = "The pass key is " + std::to_string(key) + ". Remember it. " + std::to_string(key) + " is the pass key.\n";
        const std::vector<llama_token> t_needle = tok(needle), t_q = tok("\nWhat is the pass key? The pass key is");
        const int n_fill = ((int) L - (int) (t_head.size() + t_needle.size() + t_q.size())) / (int) t_junk.size();
        const int at = (int) (n_fill * d);
        std::vector<llama_token> p = t_head;
        for (int i = 0; i < n_fill; ++i) {
            if (i == at) p.insert(p.end(), t_needle.begin(), t_needle.end());
            p.insert(p.end(), t_junk.begin(), t_junk.end());
        }
        p.insert(p.end(), t_q.begin(), t_q.end());

        llama_memory_clear(llama_get_memory(ctx), true);
        for (size_t a = 0; a < p.size(); a += 512) {
            const int n = std::min<size_t>(512, p.size() - a);
            llama_batch b = llama_batch_init(n, 0, 1);
            for (int i = 0; i < n; ++i) { b.token[i] = p[a + i]; b.pos[i] = a + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = a + i == p.size() - 1; }
            b.n_tokens = n;
            if (llama_decode(ctx, b)) { fprintf(stderr, "decode hatası\n"); return 1; }
            llama_batch_free(b);
        }
        std::string out;
        int pos = p.size();
        // greedy until the key appears, the model ends its turn, or max_new tokens (it may think first)
        const std::string ks = std::to_string(key);
        int n_gen = 0;
        for (; n_gen < max_new && out.find(ks) == std::string::npos; ++n_gen) {
            const float * l = llama_get_logits_ith(ctx, -1);
            int best = 0; for (int v = 1; v < n_vocab; ++v) if (l[v] > l[best]) best = v;
            if (llama_vocab_is_eog(vocab, best)) break;
            char buf[64]; int k = llama_token_to_piece(vocab, best, buf, sizeof(buf), 0, false);
            out.append(buf, std::max(k, 0));
            llama_batch b = llama_batch_init(1, 0, 1);
            b.token[0] = best; b.pos[0] = pos++; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1; b.n_tokens = 1;
            if (llama_decode(ctx, b)) return 1;
            llama_batch_free(b);
        }
        // first run of digits in the answer must be exactly the key
        std::string num; size_t i0 = out.find_first_of("0123456789");
        if (i0 != std::string::npos) { size_t i1 = out.find_first_not_of("0123456789", i0); num = out.substr(i0, i1 - i0); }
        const bool ok = num == ks;                                   // strict: the first number is the key
        const bool found = out.find(ks) != std::string::npos;         // lenient: the key appears somewhere
        ok_all += ok; found_all += found; ++n_all;
        std::string shown = out; for (char & c : shown) if (c == '\n') c = ' ';
        if (shown.size() > 160) shown = shown.substr(0, 70) + " … " + shown.substr(shown.size() - 70);
        printf("bağlam %6zu  derinlik %.2f  parola %d  %3d token  cevap \"%s\"  %s\n", p.size(), d, key, n_gen,
               shown.c_str(), ok ? "DOĞRU" : found ? "DOĞRU (düşündükten sonra)" : "yanlış");
        fflush(stdout);
    }
    printf("doğruluk: sıkı (ilk sayı) %d / %d | esnek (cevapta geçiyor) %d / %d\n", ok_all, n_all, found_all, n_all);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
