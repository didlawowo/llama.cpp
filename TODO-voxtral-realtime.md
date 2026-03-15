# Voxtral Realtime 4B — llama.cpp Integration

## Context

Clean implementation of Voxtral Realtime 4B support for llama.cpp,
based on PR #19698 by @Acceldium, refactored into minimal incremental changes.

Related:

- Issue: <https://github.com/ggml-org/llama.cpp/issues/19696>
- Original PR: <https://github.com/ggml-org/llama.cpp/pull/19698>
- Fork branch: didlawowo/llama.cpp@feat/voxtral-realtime-basic

## Done

### Phase 1 — Core model support (commit dcfaec8db)

- [x] GGUF conversion (`convert_hf_to_gguf.py`) — VoxtralRealtimeEncoderModel + ada_norm precompute
- [x] Tensor mapping (`gguf-py/`) — FFN_ADA_NORM_DOWN/UP, VOXTRAL_REALTIME projector
- [x] Decoder ada_rms_norm_t_cond (`src/models/llama.cpp`) — precomputed scale in forward pass
- [x] Causal audio encoder (`tools/mtmd/models/voxtral-realtime-enc.cpp`) — sliding window, RoPE, SwiGLU
- [x] Audio preprocessor (`tools/mtmd/mtmd-audio.cpp`) — Voxtral-specific mel params
- [x] mtmd integration (`clip.cpp`, `mtmd.cpp`) — 7 case blocks for VOXTRAL_REALTIME
- [x] Verified: GGUF conversion (text Q8_0 + mmproj F16) works
- [x] Verified: llama-server loads model, detects audio encoder correctly

### Phase 2 — Dual-stream prefill (commit 42e668bae)

- [x] `mtmd_helper_eval_voxtral_realtime()` in mtmd-helper.cpp
- [x] Token embedding table loader from GGUF
- [x] Audio + text embedding summation at each position
- [x] Prefill decode with combined embeddings
- [x] Compiles and links OK

## TODO

### Phase 2 — Dual-stream (remaining)

- [ ] Add caller code in `mtmd-cli.cpp` that uses `mtmd_helper_eval_voxtral_realtime()`
      then samples tokens autoregressively (needed to test end-to-end)
- [ ] Test with real audio file (need GPU free — scale down llama-cpp pod first)
- [ ] Fix: current API requires `model_path` string to reload token embeddings from GGUF.
      Ideally should access `tok_embd` from llama_model directly (needs llama API discussion)
- [ ] Handle autoregressive phase: after prefill, each new token needs
      `combined[pos] = audio_embd[pos] + tok_embd[new_token]` while pos < n_audio_tokens

### Phase 3 — Server integration

- [ ] Modify `llama-server` to detect `mtmd_support_voxtral_realtime()` and use dual-stream path
- [ ] Support `/v1/chat/completions` with `input_audio` content type
- [ ] Handle the audio marker in the chat template for Voxtral Realtime

### Phase 4 — Cleanup for upstream PR

- [ ] Wait for @ngxson response on issue #19696
- [ ] Run llama-perplexity and llama-bench
- [ ] Convert and upload GGUF to HuggingFace
- [ ] Remove any remaining debug code
- [ ] Ensure no PADDLEOCR or other master changes are missing
- [ ] Test CI locally
- [ ] Credit @Acceldium in PR description

## Architecture Notes

### Dual-stream protocol (Voxtral Realtime specific)

Unlike standard multimodal models that prepend media embeddings:

```
Standard:  [audio_embeddings] [text_tokens] → decoder
Voxtral RT: [audio_embd[i] + tok_embd[prompt[i]]] → decoder
```

Each position sums audio and text embeddings. This enables streaming
with <500ms latency.

### Prefix tokens

```
[BOS] + [PAD × 32] + [DELAY × 6] = 39 tokens
TOKEN_BOS = 1, TOKEN_STREAMING_PAD = 32, N_DELAY_TOKENS = 6
```

### Key files

| File | Role |
|------|------|
| `tools/mtmd/models/voxtral-realtime-enc.cpp` | Causal audio encoder graph |
| `tools/mtmd/mtmd-helper.cpp` | Dual-stream prefill logic |
| `tools/mtmd/mtmd-audio.cpp` | Mel spectrogram preprocessor |
| `tools/mtmd/clip.cpp` | Encoder integration (7 case blocks) |
| `convert_hf_to_gguf.py` | GGUF conversion + ada_norm precompute |
| `src/models/llama.cpp` | ada_rms_norm in forward pass |

### Model files (on NFS /mnt/models/)

- Source: `models--mistralai--Voxtral-Mini-4B-Realtime-2602/`
- Converted: `/tmp/voxtral-4b-rt.gguf` (3.5 GB, Q8_0)
- Converted: `/tmp/mmproj-voxtral-4b-rt.gguf` (1.9 GB, F16)

### Docker build

Custom image in `infra/docker-build/builds/llama-cpp-voxtral-rt/`
