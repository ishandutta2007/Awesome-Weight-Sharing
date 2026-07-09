import os
import markdown

details_dir = 'C:/Users/ishan/Documents/Projects/Awesome-Weight-Sharing/details'
os.makedirs(details_dir, exist_ok=True)

files = [
    ('hardwired-spatial-receptive-field.md', 'Hardwired Spatial Receptive Field Era', 'A structural genesis born from biological vision research. Introduced sliding window functions.'),
    ('semantic-vocabulary-embedding-tying.md', 'Semantic Vocabulary Embedding Tying Era', 'Expanded parameter sharing into natural language text sequences by linking vocabularies symmetrically.'),
    ('distributed-runtime-state-sharding.md', 'Distributed Runtime State Sharding Era', 'Ported parameter reuse out of static layer designs into distributed cluster memory management (FSDP).'),
    ('hardware-fused-latent-parallelism.md', 'Hardware-Fused Latent Parallelism Era', 'Compresses the shared key-value context history down into a highly dense, low-rank latent vector on-the-fly.'),
    ('spatial-weight-sharing.md', 'Spatial Weight Sharing (Convolutional Kernels)', 'Slides a localized parameter kernel array across a multidimensional tensor space recursively.'),
    ('temporal-weight-sharing.md', 'Temporal Weight Sharing (Recurrent Networks)', 'Uses an identical, fixed transition weight matrix across every chronological time step.'),
    ('input-output-weight-tying.md', 'Input-Output Weight Tying (Token Space Inversion)', 'Maps the final word token generation head directly back to the step-zero input lookup table.'),
    ('parameter-efficient-low-rank-sharing.md', 'Parameter-Efficient Low-Rank Sharing', 'Freezes foundation parameters, routing multiple tasks via microscopic, parallel low-rank adapter layers.'),
    ('asynchronous-pre-fetch-kernels.md', 'Asynchronous Pre-fetch Kernels', 'Executes asynchronous All-Gather primitive in the background to fetch shared parameters ahead of time.'),
    ('weight-gradient-decoupling.md', 'The Weight-Gradient Decoupling Block', 'Refactors the backward optimization loops of massive clusters into independent actions.'),
    ('memory-bus-activation-bloat.md', 'The Memory-Bus Activation Bloat Wall', 'Mitigated by implementing Selective Activation Checkpointing.'),
    ('low-precision-underflow.md', 'The Low-Precision Underflow Gradient Saturation Crisis', 'Mitigated by enforcing a strict FP32 Master Weight Optimizer configuration.'),
    ('pre-training-web-scale.md', 'Pre-Training Web-Scale Foundational Transformers', 'Serves as the crucial structural backbone used to scale up token ingestion throughput.'),
    ('spatio-temporal-video.md', 'Spatio-Temporal Video Generative Flow-Matching Simulators', 'Massive 3D spatio-temporal video token cubes processed through spatial-temporal parameter kernels.'),
    ('low-latency-consumer-device.md', 'Low-Latency Consumer-Device Edge Assistant Deployments', 'Compresses model footprints to fit inside restricted system memory lines.')
]

template = '''# {title}

{desc}

## Architecture Diagram
`mermaid
flowchart TD
    A[Input] --> B[{title}]
    B --> C[Output]
`

[Back to README](../README.md)
'''

for filename, title, desc in files:
    with open(os.path.join(details_dir, filename), 'w', encoding='utf-8') as f:
        f.write(template.format(title=title, desc=desc))

print("Created 15 files")
