# llama.cpp-miniQuant: Sub-2.3 BPW Inference & Quantization Engine

A high-performance fork of `llama.cpp` engineered for **extreme low-bit quantization (sub-2.3 BPW)**, activation-aware **Hadamard rotation (QuaRot)**, and native execution of hybrid attention/SSM architectures (including **Qwen 3.5 4B** and **Qwen 3.8 27B**).

---

## Key Features

1. **Native Hybrid Attention & SSM Support**:
   - Implements the complete computation graph for `qwen35` and `spark2_5` hybrid sliding-window / SSM architectures.
   - Dynamic per-layer sliding window attention (`n_swa`) and RoPE sectioning.

2. **Hadamard Rotation & Incoherence Processing (QuaRot)**:
   - Full support for online and offline Walsh-Hadamard Transforms (FWHT) across CPU (AVX-512, AVX2) and CUDA.
   - GGUF metadata parsing for `prism.hadamard.*` and `quarot.hadamard.*` sign vectors and block sizes.
   - CPU butterfly butterfly kernels via `iqk_hadamard` and `ggml_hadamard`.

3. **Sub-2.3 BPW Quantization (SQMP + Imatrix)**:
   - Enhanced `llama-quantize` with **Structural Mixed-Precision (SQMP)** sparing flags:
     - `--token-embedding-type Q4_K`: Prevents vocabulary deadzone collapse.
     - `--output-tensor-type Q6_K`: Preserves LM head logit distribution.
     - `--attn-qkv-type IQ2_K`: Protects attention query-key projection ranks.
   - Activation Hessian Importance Matrix (`llama-imatrix`) calibration integration for `IQ2_XS`, `IQ2_KS`, and `IQ2_XXS` lattice codebooks.

4. **Optimized SIMD Vector Kernels**:
   - Highly parallel AVX-512 / AVX-VNNI matrix multiplication routines.
   - `IQ2_KS` 2.1875 BPW decoding reaching over 7.6 tokens/sec on modern x86 CPU architectures for 27B models.

---

## Build Instructions

### Linux (CPU with AVX-512 & OpenMP)
```bash
git clone https://github.com/fencerJP/llama.cpp-miniQuant.git
cd llama.cpp-miniQuant
cmake -B build \
  -DGGML_AVX512=ON \
  -DGGML_AVX512_VBMI=ON \
  -DGGML_AVX512_VNNI=ON \
  -DGGML_AVX512_BF16=ON
cmake --build build --config Release -j $(nproc)
```

### Linux (CUDA)
```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)
```

---

## Workflow: Sub-2.3 BPW Model Creation

### 1. Compute Importance Matrix
Run activation calibration against a multi-domain corpus:
```bash
./build/bin/llama-imatrix \
  -m /path/to/base-model-BF16.gguf \
  -f calibration_corpus.txt \
  -o imatrix.dat \
  --chunks 32 -t 16
```

### 2. Quantize with SQMP Sparing
Generate sub-2.3 BPW models with protected boundary tensors:
```bash
# Qwen 3.5 4B -> IQ2_XS (2.21 BPW)
./build/bin/llama-quantize \
  --imatrix imatrix.dat \
  --token-embedding-type Q4_K \
  --output-tensor-type Q6_K \
  --attn-qkv-type IQ2_K \
  Huihui-Qwen3.5-4B-ab-BF16.gguf \
  Huihui-Qwen3.5-4B-ab_miniQuant.gguf \
  IQ2_XS 16

# Qwen 3.8 27B -> IQ2_KS (2.18 BPW)
./build/bin/llama-quantize \
  --imatrix imatrix.dat \
  --token-embedding-type Q4_K \
  --output-tensor-type Q6_K \
  --attn-qkv-type IQ2_K \
  DavidAU-Qwen3.8-27B-Ultimate-BF16.gguf \
  DavidAU-Qwen3.8-27B-Ultimate-IQ2_KS_miniQuant.gguf \
  IQ2_KS 16
```

### 3. Inference
```bash
./build/bin/llama-cli \
  -m DavidAU-Qwen3.8-27B-Ultimate-IQ2_KS_miniQuant.gguf \
  -p "The boiling point of liquid nitrogen is" \
  -n 128 --temp 0
```

---

## Model Benchmark Reference

| Model | Quant Type | Size | BPW | Speed (CPU) | Physics Fact | Math Sequence ($2,6,12,20,30\dots$) | Python Code |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- | :--- |
| **Qwen 3.5 4B** | **IQ2_XS + SQMP** | 1,341 MB | **2.21** | 34.4 t/s | ✅ **77 K** | ✅ **42** | ✅ `word == word[::-1]` |
| **Qwen 3.8 27B** | **IQ2_KS + SQMP** | 7,649 MB | **2.18** | **7.68 t/s** | ✅ **77 K** | ✅ Exact ($42, 56, \dots, 342$) | ✅ `word == word[::-1]` |
| **Qwen 3.8 27B** | **IQ2_XXS + SQMP** | 7,395 MB | **2.06** | 5.90 t/s | ✅ **77 K** | ✅ Differences ($4, 6, 8\dots$) | ✅ `word == word[::-1]` |

---

## License
MIT License. Contains modules from GGML and llama.cpp contributors.
