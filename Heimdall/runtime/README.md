# Valerois Runtime

Local OpenAI-compatible server for Bifrost and Heimdall GGUF models. It is llama.cpp's `llama-server` with the Bifrost
patch (`../llamacpp/bifrost.patch`), built with Vulkan, so one build runs on AMD, NVIDIA and Intel GPUs on Linux and
Windows.

## Start

| | |
|---|---|
| Linux | `tar -xzf valerois-runtime-*-linux-x64-vulkan.tar.gz && cd valerois-runtime-*/ && ./valerois-runtime model.gguf` |
| Windows | unzip, then `valerois-runtime.cmd model.gguf` (or `.\valerois-runtime.ps1 model.gguf`) |
| From a home build | `Heimdall/runtime/valerois-runtime model.gguf` (finds `~/llama.cpp-bifrost/build/bin`) |

The server listens on `http://127.0.0.1:1234/v1`: chat completions, completions, models, streaming. Check the download
against `SHA256SUMS` from the release.

## Nova

Settings → Models → **Local**. Port 1234 is the LM Studio port Nova already knows, so nothing else needs to be set. If LM
Studio is running, start the runtime with `--port 1235` and use **Your own server** with `http://127.0.0.1:1235/v1`.

## Defaults

| option | default | why |
|---|---|---|
| `--ctx` | 131072 for a file name with `bifrost`, else 32768 | Bifrost layers keep a constant state, so a long context costs no extra GPU memory. A normal GGUF grows a KV cache. |
| `-ngl 99` | all layers on the GPU | |
| `--checkpoints` | 4 | Prompt-cache checkpoints for recurrent state. One checkpoint of Qwen3.8-27B Bifrost is ~0.8 GB of system RAM. |
| `--jinja` | on | the model's own chat template (tools, thinking) |
| `--host` | 127.0.0.1 | only this machine; use `--api-key KEY` before opening it to a network |

Anything after `--` goes to `llama-server` unchanged:
* `-- --reasoning off`: no thinking block;
* `-- --temp 0.6 --top-p 0.95`: sampling.

With two GPUs (for example an iGPU next to a dedicated card), pick one with `GGML_VK_VISIBLE_DEVICES=0`.

## Measured (RX 9070 XT, Qwen3.8-27B Bifrost Q3_K_M, llama-bench)

Decode 33.7 t/s at an empty context, 33.7 at 32K and 32.9 at 131K. The original model decodes 37.1 t/s at an empty context
and 32.9 at 32K. Prefill at 32K is 832 t/s (original: 682).

## Build

`.github/workflows/valerois-runtime.yml` builds Linux x64 (Ubuntu 22.04 base, so older distributions work) and Windows x64
on every change to `Heimdall/llamacpp` or `Heimdall/runtime`. A tag `runtime-vX.Y.Z` publishes a release. Local build:
`GPU=vulkan bash Heimdall/llamacpp/kur.sh`.

## License

The binaries contain llama.cpp (MIT, `LICENSE-llama.cpp`). The Bifrost patch and these scripts are covered by this
repository's license unless stated otherwise.
