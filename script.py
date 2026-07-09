import re

with open('C:/Users/ishan/Documents/Projects/Awesome-Weight-Sharing/README.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Section 1
sec1 = '''*   **The Hardwired Spatial Receptive Field Era (Neocognitron / LeNet CNNs, 1980–2012)**
    *   *Concept:* The core structural genesis born from biological vision research. Kunihiko Fukushima's **Neocognitron (1980)** and Yann LeCun's **LeNet-5 (1989)** proved that visual features (like edges and contours) are spatially invariant—an edge in the top-left corner shares the exact same physical property as an edge in the bottom-right corner. They introduced a sliding window function (convolutional kernel) that reuses an identical, small matrix of parameter weights across the entire visual canvas.
    *   *Limitation:* Confined to local spatial neighborhoods inside convolutional windows, making it structurally incompatible with open-ended natural language sequence or abstract token tracking.
*   **The Semantic Vocabulary Embedding Tying Era (~2016–2021)**
    *   *Concept:* Expanded parameter sharing into natural language text sequences by linking vocabularies symmetrically. Formally popularized by Press & Wolf (2016) via **Weight Tying**, it forces the transformer decoder’s input token embedding matrix and its terminal output projection layer to share the exact same physical weight tensor ({\text{input}} = W_{\text{output}}$) [INDEX: 1].
    *   *Significance:* Slashed the parameter count of language networks by up to \\times$ to \\times$ over massive dictionaries, ensuring semantic vectors remain perfectly inverted and aligned between input comprehension and output token emission [INDEX: 1].
*   **The Distributed Runtime State Sharding Era (FSDP / ZeRO-3, ~2021–2024)**
    *   *Concept:* Ported parameter reuse out of static layer designs and straight into distributed cluster memory management [INDEX: 22]. Frameworks like Meta AI's **Fully Sharded Data Parallel (FSDP)** prove that a multi-node supercomputing group can treat a sharded parameter slice as a globally shared, reusable asset [INDEX: 22].
    *   *Significance:* GPUs execute an `All-Gather` step to temporarily share parameter blocks right before a specific layer's forward pass math and immediately evict them from local VRAM afterward [INDEX: 22], letting clusters train trillion-parameter configurations smoothly without memory crashes [INDEX: 15, 22].
*   **The Hardware-Fused Latent Parallelism Era (Present)**
    *   *Concept:* The current modern state-of-the-art foundation infrastructure standard. Instead of sharing raw, uncompressed parameter columns across separate server nodes via heavy network wires, modern architectures (such as DeepSeek-V3) enforce weight sharing directly inside fast GPU SRAM registers using **Multi-Head Latent Attention (MLA)** [INDEX: 18].
    *   *Significance:* Compresses the shared key-value context history down into a highly dense, low-rank latent vector on-the-fly, utilizing fused register kernels to decode and route shared attention maps across thousands of active user pipelines simultaneously with near-zero memory bus latency [INDEX: 18].'''

rep1 = '''| Era / Concept | Description | Year First Used | Paper Link |
| :--- | :--- | :--- | :--- |
| [**The Hardwired Spatial Receptive Field Era (Neocognitron / LeNet CNNs, 1980–2012)**](details/hardwired-spatial-receptive-field.md) | *Concept:* The core structural genesis born from biological vision research. Kunihiko Fukushima's **Neocognitron (1980)** and Yann LeCun's **LeNet-5 (1989)** proved that visual features (like edges and contours) are spatially invariant—an edge in the top-left corner shares the exact same physical property as an edge in the bottom-right corner. They introduced a sliding window function (convolutional kernel) that reuses an identical, small matrix of parameter weights across the entire visual canvas.<br><br>*Limitation:* Confined to local spatial neighborhoods inside convolutional windows, making it structurally incompatible with open-ended natural language sequence or abstract token tracking. | 1980 | [Neocognitron (1980)](https://www.rctn.org/bruno/public/papers/Fukushima1980.pdf) |
| [**The Semantic Vocabulary Embedding Tying Era (~2016–2021)**](details/semantic-vocabulary-embedding-tying.md) | *Concept:* Expanded parameter sharing into natural language text sequences by linking vocabularies symmetrically. Formally popularized by Press & Wolf (2016) via **Weight Tying**, it forces the transformer decoder’s input token embedding matrix and its terminal output projection layer to share the exact same physical weight tensor ({\text{input}} = W_{\text{output}}$) [INDEX: 1].<br><br>*Significance:* Slashed the parameter count of language networks by up to \\times$ to \\times$ over massive dictionaries, ensuring semantic vectors remain perfectly inverted and aligned between input comprehension and output token emission [INDEX: 1]. | 2016 | [Press & Wolf (2016)](https://arxiv.org/abs/1608.05859) |
| [**The Distributed Runtime State Sharding Era (FSDP / ZeRO-3, ~2021–2024)**](details/distributed-runtime-state-sharding.md) | *Concept:* Ported parameter reuse out of static layer designs and straight into distributed cluster memory management [INDEX: 22]. Frameworks like Meta AI's **Fully Sharded Data Parallel (FSDP)** prove that a multi-node supercomputing group can treat a sharded parameter slice as a globally shared, reusable asset [INDEX: 22].<br><br>*Significance:* GPUs execute an `All-Gather` step to temporarily share parameter blocks right before a specific layer's forward pass math and immediately evict them from local VRAM afterward [INDEX: 22], letting clusters train trillion-parameter configurations smoothly without memory crashes [INDEX: 15, 22]. | 2020 | [ZeRO (2020)](https://arxiv.org/abs/1910.02054) |
| [**The Hardware-Fused Latent Parallelism Era (Present)**](details/hardware-fused-latent-parallelism.md) | *Concept:* The current modern state-of-the-art foundation infrastructure standard. Instead of sharing raw, uncompressed parameter columns across separate server nodes via heavy network wires, modern architectures (such as DeepSeek-V3) enforce weight sharing directly inside fast GPU SRAM registers using **Multi-Head Latent Attention (MLA)** [INDEX: 18].<br><br>*Significance:* Compresses the shared key-value context history down into a highly dense, low-rank latent vector on-the-fly, utilizing fused register kernels to decode and route shared attention maps across thousands of active user pipelines simultaneously with near-zero memory bus latency [INDEX: 18]. | 2024 | [DeepSeek-V3 (2024)](https://arxiv.org/abs/2412.19437) |'''

content = content.replace(sec1, rep1)

# Section 2
sec2 = '''- ### A. Spatial Weight Sharing (Convolutional Kernels)
	*   **Mechanism:** Slides a localized parameter kernel array across a multidimensional tensor space recursively. The network updates a single, tiny weight matrix based on the accumulated error gradients calculated across all receptive fields concurrently.
	*   **Pros:** Installs absolute translation invariance, letting the model recognize objects perfectly regardless of their physical coordinate position on the canvas.

- ### B. Temporal Weight Sharing (Recurrent Networks)
	*   **Mechanism:** The core optimization link underpining Recurrent Neural Networks (RNNs) and LSTMs. It uses an identical, fixed transition weight matrix ({hh}$) across *every chronological time step* in a sequence.
	*   **Cons:** Highly parallelization-blind, introducing intense gradient vanishing/exploding loops over long text timelines.

- ### C. Input-Output Weight Tying (Token Space Inversion)
	*   **Mechanism:** Maps the final word token generation head directly back to the step-zero input lookup table [INDEX: 1]. The token logit calculation utilizes the exact same continuous embedding coordinates used during input parsing, forcing an absolute geometric duality:
	    P(y_t | x) = \text{Softmax}(H_t \cdot W_{\text{embed}}^T)

- ### D. Parameter-Efficient Low-Rank Sharing (Shared LoRA Tuning)
	*   **Mechanism:** Freezes foundation parameters completely, routing multiple disconnected multi-tenant user tasks concurrently through a single, shared server model backbone [INDEX: 11]. Downstream task variations are isolated into microscopic, parallel low-rank adapter layers, allowing nodes to share \\%$ of active parameter memory slots stably [INDEX: 11].'''

rep2 = '''| Variant | Description | Year First Used | Paper Link |
| :--- | :--- | :--- | :--- |
| [**A. Spatial Weight Sharing (Convolutional Kernels)**](details/spatial-weight-sharing.md) | **Mechanism:** Slides a localized parameter kernel array across a multidimensional tensor space recursively. The network updates a single, tiny weight matrix based on the accumulated error gradients calculated across all receptive fields concurrently.<br><br>**Pros:** Installs absolute translation invariance, letting the model recognize objects perfectly regardless of their physical coordinate position on the canvas. | 1980 | [Neocognitron (1980)](https://www.rctn.org/bruno/public/papers/Fukushima1980.pdf) |
| [**B. Temporal Weight Sharing (Recurrent Networks)**](details/temporal-weight-sharing.md) | **Mechanism:** The core optimization link underpining Recurrent Neural Networks (RNNs) and LSTMs. It uses an identical, fixed transition weight matrix ({hh}$) across *every chronological time step* in a sequence.<br><br>**Cons:** Highly parallelization-blind, introducing intense gradient vanishing/exploding loops over long text timelines. | 1986 | [Rumelhart et al. (1986)](https://www.nature.com/articles/323533a0) |
| [**C. Input-Output Weight Tying (Token Space Inversion)**](details/input-output-weight-tying.md) | **Mechanism:** Maps the final word token generation head directly back to the step-zero input lookup table [INDEX: 1]. The token logit calculation utilizes the exact same continuous embedding coordinates used during input parsing, forcing an absolute geometric duality: P(y_t \\| x) = \\text{Softmax}(H_t \cdot W_{\\text{embed}}^T) | 2016 | [Press & Wolf (2016)](https://arxiv.org/abs/1608.05859) |
| [**D. Parameter-Efficient Low-Rank Sharing (Shared LoRA Tuning)**](details/parameter-efficient-low-rank-sharing.md) | **Mechanism:** Freezes foundation parameters completely, routing multiple disconnected multi-tenant user tasks concurrently through a single, shared server model backbone [INDEX: 11]. Downstream task variations are isolated into microscopic, parallel low-rank adapter layers, allowing nodes to share \\%$ of active parameter memory slots stably [INDEX: 11]. | 2021 | [Hu et al. (2021)](https://arxiv.org/abs/2106.09685) |'''

content = content.replace(sec2, rep2)

# Section 3
sec3 = '''*   **Asynchronous Pre-fetch Kernels**
    *   *Profile:* Interleaved network execution. As layer $ is computing its forward pass tensors inside GPU SRAM, the DeepSpeed or FSDP communication scheduler executes an asynchronous All-Gather primitive in the background to fetch the shared parameters for layer +1$ over the network switches ahead of time, eliminating interconnect latency stalls [INDEX: 22].
*   **The Weight-Gradient Decoupling Block**
    *   *Profile:* Slashes pipeline bubble constraints. It refactors the backward optimization loops of massive clusters into two independent actions: backward for activations ($) and backward for weights ($), scheduling the weight-sharing gradient calculations directly inside empty hardware bubble intervals to hit peak compute velocity.'''

rep3 = '''| Component | Description | Year First Used | Paper Link |
| :--- | :--- | :--- | :--- |
| [**Asynchronous Pre-fetch Kernels**](details/asynchronous-pre-fetch-kernels.md) | *Profile:* Interleaved network execution. As layer $ is computing its forward pass tensors inside GPU SRAM, the DeepSpeed or FSDP communication scheduler executes an asynchronous All-Gather primitive in the background to fetch the shared parameters for layer +1$ over the network switches ahead of time, eliminating interconnect latency stalls [INDEX: 22]. | 2020 | [ZeRO (2020)](https://arxiv.org/abs/1910.02054) |
| [**The Weight-Gradient Decoupling Block**](details/weight-gradient-decoupling.md) | *Profile:* Slashes pipeline bubble constraints. It refactors the backward optimization loops of massive clusters into two independent actions: backward for activations ($) and backward for weights ($), scheduling the weight-sharing gradient calculations directly inside empty hardware bubble intervals to hit peak compute velocity. | 2023 | [PyTorch FSDP (2023)](https://arxiv.org/abs/2304.11277) |'''

content = content.replace(sec3, rep3)

# Section 4
sec4 = '''*   **The Memory-Bus Activation Bloat Wall**
    *   *The Problem:* While sharing or tying weights drastically slashes parameter size on disk, it does not reduce the **Activation Memory Footprint**. Forcing multiple parallel attention heads or data streams to reuse an identical parameter weight matrix concurrently generates massive, un-reduced intermediate activation tensors, saturating GPU memory bandwidth and triggering system-wide Out-of-Memory crashes [INDEX: 22].
    *   *Mitigation:* Implementing **Selective Activation Checkpointing (Rematerialization)**, which discards intermediate activation tensors immediately after forward execution, independently recalculating them on-the-fly inside fast GPU registers only when the backward loop returns.
*   **The Low-Precision Underflow Gradient Saturation Crisis**
    *   *The Problem:* When executing weight-tying alignment across massive vocabularies using low-precision 16-bit floats (FP16 or BF16) [INDEX: 11], sharing an identical weight matrix between input lookups and output logits can cause updates to conflict. Gradients from the final layer can scale down or overwrite early layer parameters, triggering numerical underflow errors that stall cross-entropy loss optimization [INDEX: 11, 16].
    *   *Mitigation:* Enforcing a strict **FP32 Master Weight Optimizer configuration (AdamW integration)** [INDEX: 11]. While model forward and backward passes execute in high-speed, low-bit 16-bit matrices, DeepSpeed maintains and updates a high-precision copy of the master weights and optimizer moments in full 32-bit floating-point registers to protect low-bit learning increments safely.'''

rep4 = '''| Challenge | Description | Year First Used | Paper Link |
| :--- | :--- | :--- | :--- |
| [**The Memory-Bus Activation Bloat Wall**](details/memory-bus-activation-bloat.md) | *The Problem:* While sharing or tying weights drastically slashes parameter size on disk, it does not reduce the **Activation Memory Footprint**. Forcing multiple parallel attention heads or data streams to reuse an identical parameter weight matrix concurrently generates massive, un-reduced intermediate activation tensors, saturating GPU memory bandwidth and triggering system-wide Out-of-Memory crashes [INDEX: 22].<br><br>*Mitigation:* Implementing **Selective Activation Checkpointing (Rematerialization)**, which discards intermediate activation tensors immediately after forward execution, independently recalculating them on-the-fly inside fast GPU registers only when the backward loop returns. | 2016 | [Chen et al. (2016)](https://arxiv.org/abs/1604.06174) |
| [**The Low-Precision Underflow Gradient Saturation Crisis**](details/low-precision-underflow.md) | *The Problem:* When executing weight-tying alignment across massive vocabularies using low-precision 16-bit floats (FP16 or BF16) [INDEX: 11], sharing an identical weight matrix between input lookups and output logits can cause updates to conflict. Gradients from the final layer can scale down or overwrite early layer parameters, triggering numerical underflow errors that stall cross-entropy loss optimization [INDEX: 11, 16].<br><br>*Mitigation:* Enforcing a strict **FP32 Master Weight Optimizer configuration (AdamW integration)** [INDEX: 11]. While model forward and backward passes execute in high-speed, low-bit 16-bit matrices, DeepSpeed maintains and updates a high-precision copy of the master weights and optimizer moments in full 32-bit floating-point registers to protect low-bit learning increments safely. | 2017 | [Micikevicius et al. (2017)](https://arxiv.org/abs/1710.03740) |'''

content = content.replace(sec4, rep4)

# Section 5
sec5 = '''*   **Pre-Training Web-Scale Foundational Transformers (Llama / DeepSeek)**
    *   *Application:* Serves as the crucial structural backbone used to scale up token ingestion throughput [INDEX: 15, 22]. Foundation clusters layer Input-Output weight tying and sharded FSDP parameters concurrently to simulate colossal global batch sizes over thousands of GPUs cleanly without experiencing VRAM exhaustion [INDEX: 15, 22].
*   **Spatio-Temporal Video Generative Flow-Matching Simulators (Sora Class)**
    *   *Application:* Drives large-scale physical simulation training workflows. Massive 3D spatio-temporal video token cubes are processed through spatial-temporal parameter kernels that reuse weight matrices concurrently across frame dimensions, allowing the model to internalize consistent fluid physics, lighting shifts, and collision boundaries seamlessly.
*   **Low-Latency Consumer-Device Edge Assistant Deployments (Mobile SLMs)**
    *   *Application:* Running localized assistants on mobile phones or consumer laptops [INDEX: 16]. Structural parameter reuse, layered alongside group-wise block quantization and **BitsAndBytes 4-bit templates**, compresses model footprints to fit inside restricted system memory lines, running text loops without draining device batteries [INDEX: 16].'''

rep5 = '''| Application | Description | Year First Used | Paper Link |
| :--- | :--- | :--- | :--- |
| [**Pre-Training Web-Scale Foundational Transformers (Llama / DeepSeek)**](details/pre-training-web-scale.md) | *Application:* Serves as the crucial structural backbone used to scale up token ingestion throughput [INDEX: 15, 22]. Foundation clusters layer Input-Output weight tying and sharded FSDP parameters concurrently to simulate colossal global batch sizes over thousands of GPUs cleanly without experiencing VRAM exhaustion [INDEX: 15, 22]. | 2023 | [LLaMA (2023)](https://arxiv.org/abs/2302.13971) |
| [**Spatio-Temporal Video Generative Flow-Matching Simulators (Sora Class)**](details/spatio-temporal-video.md) | *Application:* Drives large-scale physical simulation training workflows. Massive 3D spatio-temporal video token cubes are processed through spatial-temporal parameter kernels that reuse weight matrices concurrently across frame dimensions, allowing the model to internalize consistent fluid physics, lighting shifts, and collision boundaries seamlessly. | 2024 | [Sora (2024)](https://openai.com/research/video-generation-models-as-world-simulators) |
| [**Low-Latency Consumer-Device Edge Assistant Deployments (Mobile SLMs)**](details/low-latency-consumer-device.md) | *Application:* Running localized assistants on mobile phones or consumer laptops [INDEX: 16]. Structural parameter reuse, layered alongside group-wise block quantization and **BitsAndBytes 4-bit templates**, compresses model footprints to fit inside restricted system memory lines, running text loops without draining device batteries [INDEX: 16]. | 2023 | [QLoRA (2023)](https://arxiv.org/abs/2305.14314) |'''

content = content.replace(sec5, rep5)

with open('C:/Users/ishan/Documents/Projects/Awesome-Weight-Sharing/README.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done")
