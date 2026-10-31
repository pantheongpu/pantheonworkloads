// pw-margin: one model load, many prompts. For each prompt greedy-decode -n tokens (no chat template, BOS added as
// llama-completion does) and print one JSON line per prompt with, per step, the top-1 and top-2 token pieces and
// the logit gap top1-top2 (equal to the log-probability gap, softmax only shifts every logit by the same constant).
// Prompts come one per line from a file, "\n" written as the two characters \n. Built the same way as pw-probe.cpp:
//   c++ -std=c++17 -O2 pw-margin.cpp -I$src/include -I$src/ggml/include -L$build/bin -lllama -lggml -lggml-base -Wl,-rpath,$build/bin -o pw-margin
//   pw-margin -m model.gguf -f prompts.txt [-n 3] [-ngl 99] [-c 256] [-t 2]
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"

static void quiet(ggml_log_level level, const char * text, void *) { if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr); }

static std::string esc(const std::string & s) {
    std::string o;
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += (char) c; }
        else if (c == '\n') o += "\\n";
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += (char) c;
    }
    return o;
}

static std::string piece(const llama_vocab * v, llama_token t) {
    char buf[256];
    int n = llama_token_to_piece(v, t, buf, sizeof buf, 0, true);
    return n < 0 ? std::string("?") : std::string(buf, n);
}

int main(int argc, char ** argv) {
    std::string model_path, file;
    int n_predict = 3, n_threads = 2, ngl = 99, n_ctx = 256;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "-m") model_path = next(); else if (a == "-f") file = next(); else if (a == "-n") n_predict = atoi(next());
        else if (a == "-t") n_threads = atoi(next()); else if (a == "-ngl") ngl = atoi(next()); else if (a == "-c") n_ctx = atoi(next());
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || file.empty()) { fprintf(stderr, "usage: pw-margin -m model.gguf -f prompts.txt [-n 3] [-ngl 99]\n"); return 2; }
    llama_log_set(quiet, nullptr);
    ggml_backend_load_all();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    mp.progress_callback = [](float, void *) { return true; };
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "cannot load %s\n", model_path.c_str()); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = n_ctx; cp.n_ubatch = n_ctx; cp.n_threads = n_threads; cp.n_threads_batch = n_threads;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "cannot create a context\n"); return 1; }
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::ifstream in(file);
    std::string line;
    while (std::getline(in, line)) {
        std::string prompt;
        for (size_t i = 0; i < line.size(); i++) {
            if (line[i] == '\\' && i + 1 < line.size() && line[i + 1] == 'n') { prompt += '\n'; i++; } else prompt += line[i];
        }
        if (prompt.empty()) continue;
        llama_memory_clear(llama_get_memory(ctx), true);
        std::vector<llama_token> toks(prompt.size() + 8);
        int n = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), toks.data(), (int) toks.size(), true, true);
        if (n < 0) { toks.resize(-n); n = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), toks.data(), (int) toks.size(), true, true); }
        toks.resize(n);
        std::vector<llama_token> feed = toks;
        std::string out = "";
        printf("{\"prompt\":\"%s\",\"steps\":[", esc(prompt).c_str());
        for (int s = 0; s < n_predict; s++) {
            llama_batch b = llama_batch_get_one(feed.data(), (int) feed.size());
            if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
            const float * lg = llama_get_logits_ith(ctx, -1);
            int best = 0, second = -1;
            for (int v = 1; v < n_vocab; v++) if (lg[v] > lg[best]) best = v;
            for (int v = 0; v < n_vocab; v++) if (v != best && (second < 0 || lg[v] > lg[second])) second = v;
            printf("%s{\"top1\":\"%s\",\"top2\":\"%s\",\"gap\":%.4f}", s ? "," : "", esc(piece(vocab, best)).c_str(), esc(piece(vocab, second)).c_str(), lg[best] - lg[second]);
            out += piece(vocab, best);
            feed.assign(1, best);
        }
        printf("],\"output\":\"%s\"}\n", esc(out).c_str());
        fflush(stdout);
    }
    llama_free(ctx); llama_model_free(model);
    return 0;
}
