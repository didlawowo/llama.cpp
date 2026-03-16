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

### Phase 2 — Dual-stream transcription (commits 42e668bae → 22f14007f)

- [x] `mtmd_helper_eval_voxtral_realtime()` in mtmd-helper.cpp
- [x] Token embedding table loader from GGUF
- [x] Audio + text embedding summation at each position
- [x] Prefill decode with combined embeddings
- [x] Full autoregressive decoding with dual-stream continuation
- [x] Caller code in `mtmd-cli.cpp` — detects Voxtral RT and uses dual-stream path
- [x] Tested with LibriSpeech audio — **perfect transcription output**
- [x] Performance: 169ms encode, 145 tok/s decode on RTX 3090

### Wrapper server (tools/voxtral-rt-server.py)

- [x] OpenAI-compatible `/v1/audio/transcriptions` endpoint
- [x] Multipart form-data parsing (file + language + model fields)
- [x] Forks llama-mtmd-cli per request, cleans up STREAMING_PAD/WORD tokens
- [x] Compatible with audioloadtest UI

## TODO

### Phase 3 — Server integration (blocked: needs @ngxson feedback)

The llama-server integration requires modifying the **slot processing** pipeline
in `server-context.cpp`. The current architecture processes chunks sequentially
(tokenize → [encode chunk → decode chunk] → sample), but Voxtral RT needs the
dual-stream path where audio and text embeddings are summed at each position.

Changes needed (~500 lines in server code owned by @ngxson):
- [ ] Modify slot processing to detect `mtmd_support_voxtral_realtime()`
- [ ] Route audio requests through `mtmd_helper_eval_voxtral_realtime()` instead of `process_chunk()`
- [ ] Continue dual-stream summation during autoregressive decode (not just prefill)
- [ ] Support `/v1/audio/transcriptions` endpoint (currently missing from llama-server)
- [ ] Support `/v1/chat/completions` with `input_audio` content type

**Decision**: Wait for @ngxson's response on issue #19696 before modifying server code.
He maintains this code and will want to validate the approach.

### Phase 4 — Cleanup for upstream PR

- [ ] Wait for @ngxson response on issue #19696
- [ ] Discuss `tok_embd` access — current API reloads from GGUF file (wasteful),
      ideally should access from llama_model directly (needs llama API change)
- [ ] Replace greedy sampling in helper with proper sampler passed by caller
- [ ] Run llama-perplexity and llama-bench
- [ ] Convert and upload GGUF to HuggingFace
- [ ] Remove any remaining debug code
- [ ] Ensure no PADDLEOCR or other master changes are missing after rebase
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
| `tools/mtmd/mtmd-helper.cpp` | Dual-stream prefill + autoregressive decode |
| `tools/mtmd/mtmd-helper.h` | Public API declaration |
| `tools/mtmd/mtmd-cli.cpp` | CLI with Voxtral RT single-turn mode |
| `tools/mtmd/mtmd-audio.cpp` | Mel spectrogram preprocessor |
| `tools/mtmd/clip.cpp` | Encoder integration (7 case blocks) |
| `convert_hf_to_gguf.py` | GGUF conversion + ada_norm precompute |
| `src/models/llama.cpp` | ada_rms_norm in forward pass |
| `tools/voxtral-rt-server.py` | OpenAI-compatible wrapper server |

### Model files (on NFS /mnt/models/)

- Source: `models--mistralai--Voxtral-Mini-4B-Realtime-2602/`
- Converted: `/tmp/voxtral-4b-rt.gguf` (3.5 GB, Q8_0)
- Converted: `/tmp/mmproj-voxtral-4b-rt.gguf` (1.9 GB, F16)

### Docker build

Custom image in `infra/docker-build/builds/llama-cpp-voxtral-rt/`

### Testing with audioloadtest

```bash
# 1. On nvidia — scale down llama-cpp pod first
kubectl scale deploy llama-cpp -n mlops --replicas=0

# 2. Start wrapper server
cd ~/llama.cpp
python3 tools/voxtral-rt-server.py \
    --model /tmp/voxtral-4b-rt.gguf \
    --mmproj /tmp/mmproj-voxtral-4b-rt.gguf \
    --port 8090

# 3. audioloadtest UI targets http://nvidia:8090/v1
# Note: wrapper forks llama-mtmd-cli per request (~2s cold start per req)
# For proper load testing, need Phase 3 (server integration)

# 4. Restore pod when done
kubectl scale deploy llama-cpp -n mlops --replicas=1
```

### Test result (2026-03-16)

```
Input:  LibriSpeech 7127-75946-0002.flac
Ref:    LET HIM COME IN THEN SAID THE KING AND AS IF COLBERT HAD BEEN
        LISTENING AT THE DOOR FOR THE PURPOSE OF KEEPING HIMSELF AU
        COURANT WITH THE CONVERSATION HE ENTERED AS SOON AS THE KING
        HAD PRONOUNCED HIS NAME TO THE TWO COURTIERS
Output: Let him come in then, said the king. And as if Colbert had been
        listening at the door for the purpose of keeping himself au
        courant with the conversation, he entered as soon as the king
        had pronounced his name to the two courtiers.
Encode: 169ms | Decode: 145 tok/s | Total: ~6.6s
```

## Code stats

| Metric | Value |
|--------|-------|
| Original PR #19698 | 2966 lines, 24 files |
| Our implementation | ~950 lines, 20 files |
| Reduction | 68% fewer lines |
| New files | 2 (voxtral-realtime-enc.cpp, voxtral-rt-server.py) |
