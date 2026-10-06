// pw-probe: greedy-decode a few tokens with llama.cpp's C API and print, as one JSON object, the exact
// token ids and summary statistics of the logits at every step. Used by the llamacpp-synth-* workloads
// because llama-completion prints text, not token ids. Built by tools/llamacpp/build.sh against the
// pinned llama.cpp (MIT) with plain `c++`; it uses only the public C API in llama.h and ggml-backend.h.
//
//   pw-probe -m model.gguf -p "prompt text" [-n 6] [-t 2] [-ngl 0] [-c 256] [--no-bos]
//            [-ub N] [--fa on|off|auto] [--kv f16|f32] [--no-repack]     (variants, to measure run-to-run spread)
//
// Greedy = the highest logit, the lowest token id on an exact tie. End-of-generation tokens do not stop
// the loop (random-weight models emit them freely): exactly -n tokens are generated.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"

static void quiet(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

int main(int argc, char ** argv) {
    std::string model_path, prompt = "The";
    int n_predict = 6, n_threads = 2, n_gpu_layers = 0, n_ctx = 256;
    bool add_bos = true, no_repack = false;
    int n_ubatch = 0, fa = -1;
    ggml_type kv = GGML_TYPE_F16;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "-m") model_path = next();
        else if (a == "-p") prompt = next();
        else if (a == "-n") n_predict = atoi(next());
        else if (a == "-t") n_threads = atoi(next());
        else if (a == "-ngl") n_gpu_layers = atoi(next());
        else if (a == "-c") n_ctx = atoi(next());
        else if (a == "--no-bos") add_bos = false;
        else if (a == "--no-repack") no_repack = true;
        else if (a == "-ub") n_ubatch = atoi(next());
        else if (a == "--fa") { std::string v = next(); fa = v == "on" ? 1 : v == "off" ? 0 : -1; }
        else if (a == "--kv") kv = std::string(next()) == "f32" ? GGML_TYPE_F32 : GGML_TYPE_F16;
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty()) { fprintf(stderr, "usage: pw-probe -m model.gguf -p text [-n N] [-t T] [-ngl L]\n"); return 2; }

    llama_log_set(quiet, nullptr);
    ggml_backend_load_all();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    mp.use_extra_bufts = !no_repack;
    mp.progress_callback = [](float, void *) { return true; };   // no progress dots
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "cannot load %s\n", model_path.c_str()); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ctx;
    cp.n_ubatch = n_ubatch > 0 ? n_ubatch : n_ctx;
    cp.flash_attn_type = (llama_flash_attn_type) fa;
    cp.type_k = kv;
    cp.type_v = kv;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "cannot create a context\n"); return 1; }

    std::vector<llama_token> toks(prompt.size() + 8);
    int n = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), toks.data(), (int) toks.size(), add_bos, false);
    if (n < 0) { toks.resize(-n); n = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), toks.data(), (int) toks.size(), add_bos, false); }
    if (n <= 0) { fprintf(stderr, "empty prompt\n"); return 1; }
    toks.resize(n);

    const int n_vocab = llama_vocab_n_tokens(vocab);
    printf("{\"n_vocab\":%d,\"prompt_tokens\":[", n_vocab);
    for (int i = 0; i < n; i++) printf(i ? ",%d" : "%d", toks[i]);
    printf("],\"steps\":[");

    std::vector<llama_token> gen;
    std::vector<llama_token> feed = toks;
    for (int step = 0; step < n_predict; step++) {
        llama_batch batch = llama_batch_get_one(feed.data(), (int) feed.size());
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "llama_decode failed at step %d\n", step); return 1; }
        const float * lg = llama_get_logits_ith(ctx, -1);
        int best = 0, second = -1;
        double sum = 0, sum2 = 0, mx = lg[0];
        for (int v = 0; v < n_vocab; v++) {
            if (lg[v] > lg[best]) best = v;
            sum += lg[v]; sum2 += (double) lg[v] * lg[v];
            if (lg[v] > mx) mx = lg[v];
        }
        for (int v = 0; v < n_vocab; v++) if (v != best && (second < 0 || lg[v] > lg[second])) second = v;
        double lse = 0;
        for (int v = 0; v < n_vocab; v++) lse += exp((double) lg[v] - mx);
        const double mean = sum / n_vocab, sd = sqrt(sum2 / n_vocab - mean * mean);
        printf("%s{\"top1\":%d,\"top2\":%d,\"top1_logit\":%.6g,\"margin\":%.6g,\"mean\":%.6g,\"std\":%.6g,\"logsumexp\":%.6g}",
               step ? "," : "", best, second, lg[best], lg[best] - lg[second], mean, sd, mx + log(lse));
        gen.push_back(best);
        feed.assign(1, best);
    }
    printf("],\"generated\":[");
    for (size_t i = 0; i < gen.size(); i++) printf(i ? ",%d" : "%d", gen[i]);
    printf("]}\n");
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
