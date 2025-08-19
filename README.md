# 30-Day, 100-Problem Triton Challenge 🚀

_Tackle one month of progressively harder GPU-kernel exercises written with [Triton](https://github.com/openai/triton). Check ✓ as you go!_

| Day    | Problems (✓ when done)                                                            | Core Topics                                 |
| ------ | --------------------------------------------------------------------------------- | ------------------------------------------- |
| **01** | [✓] Hello Triton print[ ] Vector addition[ ] Scalar × Vector[ ] Element-wise ReLU | Kernel anatomy - program_id - tl.load/store |
| **02** | [ ] Vector dot-product[ ] AXPY (a·x + y)[ ] Element-wise sigmoid                  | Pointer math - broadcasting - exp/log       |
| **03** | [ ] Hadamard matrix multiply[ ] Vector L2 norm[ ] Columnwise add bias             | Strides - reductions - in-kernel loops      |
| **04** | [ ] 1-D inclusive sum scan[ ] 1-D exclusive scan[ ] Prefix max                    | Hillis-Steele - warp sync - mask            |
| **05** | [ ] Histogram 256-bin[ ] Histogram privatized[ ] 1-bit population count           | Atomic add - privatization - bit ops        |
| **06** | [ ] 1-D convolution[ ] 3-pt stencil[ ] Moving average                             | Sliding windows - boundary guards           |
| **07** | [ ] 2-D grayscale to RGB[ ] Image negation[ ] 2-D mirror pad[ ] 2-D crop          | 2-D indexing - shape transforms             |
| **08** | [ ] 2-D matrix transpose[ ] Strided transpose (m×n→n×m)[ ] Tile-based transpose   | Memory coalescing - shared SRAM             |
| **09** | [ ] Naïve GEMM (C=AB)[ ] Block-tiled GEMM[ ] GEMM with bias                       | Block mapping - accumulators                |
| **10** | [ ] GEMV row-major[ ] GEMV col-major[ ] Outer product build-up                    | Thread coarsening - reuse                   |
| **11** | [ ] Softmax forward naïve[ ] Softmax forward fused[ ] Log-softmax                 | Exp/max reduction - numerical stability     |
| **12** | [ ] Softmax backward naïve[ ] Softmax backward optimized[ ] Cross-entropy loss    | Derivatives - in-kernel reuse               |
| **13** | [ ] LayerNorm forward[ ] LayerNorm backward[ ] RMSNorm forward                    | Row stats - variance trick                  |
| **14** | [ ] BatchNorm inference[ ] BatchNorm training[ ] Welford mean/var                 | Channel stats - running buffers             |
| **15** | [ ] Max pooling 2-D fwd[ ] Max pooling 2-D bwd[ ] Avg pooling fwd                 | Window overlaps - argmax cache              |
| **16** | [ ] Depthwise conv 3×3[ ] Pointwise conv 1×1[ ] Separable conv                    | Pad/stride dilation - groups                |
| **17** | [ ] Im2col transform[ ] Col2im reverse[ ] GEMM-based conv                         | 5-D flattening - layout choice              |
| **18** | [ ] Sparse SPMV COO[ ] SPMV CSR[ ] Sparse softmax                                 | Indirect loads - gather/scatter             |
| **19** | [ ] CSR2COO convert[ ] Sort rows by nnz[ ] Jagged-to-dense pad                    | Prefix sums - segmented ops                 |
| **20** | [ ] Inclusive scan segmented[ ] Exclusive scan segmented[ ] ZX-scan benchmark     | Flags array - multi-block sync              |
| **21** | [ ] K-way merge sort[ ] Bitonic sort 1024 keys[ ] Radix sort 32-bit               | Shared memory sorting - scan reuse          |
| **22** | [ ] Top-K logits (small K)[ ] Argmax along dim[ ] Partial sort Top-N              | Heap vs. scan - iterative max               |
| **23** | [ ] BFS frontier push[ ] BFS frontier pull[ ] Degree histogram                    | Graph CSR traversal - mask ops              |
| **24** | [ ] PageRank one iteration[ ] Connected-components label[ ] Triangle count        | Sparse-dense mix - atomic float             |
| **25** | [ ] ReLU fused GEMM[ ] GELU fused GEMM[ ] SiLU fused GEMM                         | Kernel fusion - activation math             |
| **26** | [ ] Multi-head QKV matmul[ ] Bias-Dropout-Add fusion[ ] Rotary PE apply           | MHA building blocks - tensor views          |
| **27** | [ ] Flash-Attention forward[ ] Flash-Attention backward                           | Block-scatter - shared softmax              |
| **28** | [ ] Transformer block end-to-end[ ] Residual + LayerNorm fusion                   | Pipe parallel - cuda_graph                  |
| **29** | [ ] GEMM autotune sweep[ ] Occupancy roofline plot[ ] NVLink scale test           | Triton autotuner - profiling APIs           |
| **30** | [ ] Benchmark vs. cuBLAS[ ] Nsight Compute metric dump[ ] Write blog recap        | Validation - reporting                      |

---

## How to Use

1. **Fork and clone** this repo, then create a branch `your-name/progress`.
2. Each problem lives in `day_##/problem_##.py` (or `.ipynb`).
3. Run, test, profile, and tick the box in the table above.
4. Daily commits keep you accountable—30 days → 100 kernels!

---

### Environment Quick-start

```bash
conda create -n triton30 python=3.10 -y
conda activate triton30
pip install triton torch numpy
```

---

### Tips

- **Docs**: `python -m triton --help` and examples in `python -m triton.examples`.
- **Debug**: Insert `tl.debug_barrier()`; sync with `torch.cuda.synchronize()`.
- **Profile**: `nsys profile -t cuda,osrt python day_09/gemm.py`.

---

Happy hacking—may your memory accesses always be coalesced!
