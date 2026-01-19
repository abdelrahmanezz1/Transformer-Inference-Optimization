\# ⚡ Transformer Inference Optimization Toolkit



\*\*Achieved 3.9x latency reduction, 50% memory footprint reduction, and ~9,000 req/s throughput on NVIDIA T4 GPU.\*\*



\## 🚀 Project Overview

This toolkit provides a high-performance inference pipeline for Transformer models (DistilBERT/BERT). It bypasses standard PyTorch limitations by implementing \*\*ONNX Runtime\*\*, \*\*Mixed Precision (FP16)\*\*, and \*\*Graph Optimization\*\*.



\*\*Key Engineering Wins:\*\*

\* \*\*Solved Opset 12 Conflicts:\*\* Implemented custom `SDPBackend.MATH` context managers to resolve PyTorch 2.x `scaled\_dot\_product\_attention` export failures.

\* \*\*Throughput Maximization:\*\* Achieved \*\*~9,000 samples/second\*\* at Batch 32 using FP16 memory optimization.

\* \*\*Memory Optimization:\*\* Reduced model footprint by \*\*50%\*\* (255MB → 127MB) via FP16 quantization.

\* \*\*Dependency Engineering:\*\* Resolved NumPy 2.0 ABI incompatibilities via strict version pinning.



\## 📊 Benchmark Results (NVIDIA T4)



| Batch Size | PyTorch Latency | ONNX FP16 Latency | Speedup | Throughput (ONNX) |

|:----------:|:---------------:|:-----------------:|:-------:|:-----------------:|

| 1 | 5.00 ms | 1.28 ms | \*\*3.9x\*\* | 781 req/s |

| 16 | 7.46 ms | 2.24 ms | \*\*3.3x\*\* | 7,142 req/s |

| 32 | 13.00 ms | 3.61 ms | \*\*3.6x\*\* | 8,864 req/s |



\## 📉 Visual Proof



\### 1. Speedup Factor (The "Money Shot")

!\[Speedup Curve](images/chart\_speedup.png)



\### 2. Throughput Scaling (Scalability)

!\[Throughput](images/chart\_throughput.png)



\### 3. Latency Reduction (Responsiveness)

!\[Latency](images/chart\_latency.png)



\### 4. Memory Footprint (Efficiency)

!\[Memory](images/chart\_memory.png)



\### 5. Optimization Frontier (Trade-off Analysis)

!\[Tradeoff](images/chart\_tradeoff.png)



\## 🛠️ Technical Stack

\* \*\*Core:\*\* PyTorch 2.x, Transformers (Hugging Face)

\* \*\*Compiler:\*\* ONNX Runtime (CUDA Execution Provider)

\* \*\*Optimization:\*\* Mixed Precision (FP16), Operator Fusion

\* \*\*Hardware:\*\* NVIDIA Tesla T4



\## 💻 Usage

```python

\# 1. Initialize Toolkit

toolkit = TransformerOptimizationToolkit("distilbert-base-uncased")



\# 2. Export \& Optimize

toolkit.export\_onnx(dummy\_input)



\# 3. Benchmark

df = toolkit.run\_full\_analysis(text)

