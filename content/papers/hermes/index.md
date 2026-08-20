<figure id="fig:teaser_c" data-latex-placement="t">
<figure id="fig:teaser_a">
<embed src="figures/teaser.pdf" />
<figcaption><span style="color: HermesBlue"><em><strong>HERMES</strong></em></span> Framework</figcaption>
</figure>
<figure id="fig:teaser_b">
<embed src="figures/teaser.pdf" />
<figcaption>Attention Analysis</figcaption>
</figure>
<figure id="fig:teaser_c">
<embed src="figures/teaser.pdf" />
<figcaption>Efficiency Test</figcaption>
</figure>
<figcaption><strong>Left</strong>: <span style="color: HermesBlue"><em><strong>HERMES</strong></em></span> is a training-free approach for efficient streaming video understanding, enabling stable inference by reusing KV cache and performing hierarchical management of video tokens stored in KV cache. <strong>Middle</strong>: <span style="color: HermesBlue"><em><strong>HERMES</strong></em></span> is based on a mechanistic investigation of the layer-wise attention preferences over hierarchical video information. <strong>Right</strong>: We evaluate LLaVA-OV-7B on a single A800 GPU (80 GB). As input frames increase, <span style="color: HermesBlue"><em><strong>HERMES</strong></em></span> consistently maintains extremely low latency (TTFT &lt; 30 ms) and stable GPU memory consumption, exhibiting no risk of OOM errors and requiring no auxiliary external computational resources.</figcaption>
</figure>

# Introduction {#sec:introduction}

Recent years have witnessed remarkable evolution in the capabilities of Multimodal Large Language Models (MLLMs) in video understanding tasks [@gemini25; @li2024llavaonevisioneasyvisualtask; @bai2025qwen3vltechnicalreport]. Despite the progress, the rapid emergence of real-time applications demands stable long video understanding, low-latency response, and memory-efficient deployment. However, existing MLLMs struggle to simultaneously satisfy these requirements on streaming videos. Notably, TimeChat-Online [@timechatonline] observes that a large number of streaming video tokens are redundant, motivating compression methods to address these challenges. While numerous compression techniques have been proposed for offline videos [@wang2025videotreeadaptivetreebasedvideo; @yang2024visionziplongerbetternecessary; @tao2025dycokedynamiccompressiontokens], most are ill-suited for memory management in streaming scenarios, as streaming inputs are unpredictable in future frames and queries.

To adapt to streaming inputs, recent research introduces specialized memory management techniques, which generally fall into two paradigms: external memory and internal memory. External memory methods store video content as captions or raw vision patches in databases, and perform ad-hoc retrieval and multimodal prefilling at query time [@xiong2025streamingvideounderstandingmultiround; @yang2025streamagentanticipatoryagentsstreaming], suffering from high latency and a lack of end-to-end cohesion. Additionally, many of these methods necessitate costly model-specific training [@wang2025streambridgeturningofflinevideo; @xu2025streamingvlmrealtimeunderstandinginfinite; @zeng2025streamforestefficientonlinevideo]. In contrast, internalizing memory directly into the key-value cache (KV cache) remains underexplored, yet is crucial for low-latency responses and seamless end-to-end reasoning over stored video contexts. Moreover, KV cache naturally acts as a latent, model-intrinsic memory [@hu2025memoryageaiagents] that frequently interacts with the video stream, making it particularly suitable for training-free memory management. ReKV [@di2025streamingvideoquestionansweringincontext] and LiveVLM [@ning2025livevlmefficientonlinevideo] are representative training-free, cache-based methods for streaming memory management. They store previous video segments in external CPU or disk and need to perform an additional retrieval when a user query arrives, which still rely on external computational resources and leads to significant latency. StreamMem [@yang2025streammemqueryagnostickvcache] leverages chat template tokens to guide compression but lacks fine-grained KV management and mechanistic interpretability.

To overcome the aforementioned limitations of existing streaming video methods, we propose [***HERMES***]{style="color: HermesBlue"}(KV Cache as [**H**]{.underline}i[**ER**]{.underline}archical [**M**]{.underline}emory for [**E**]{.underline}fficient [**S**]{.underline}treaming Video Understanding), a training-free and plug-and-play approach that can be seamlessly integrated into existing MLLMs. Grounded in a mechanistic investigation of layer-wise attention shown in [2](#fig:teaser_b){reference-type="ref+label" reference="fig:teaser_b"}, we conceptualize KV cache as a hierarchical memory framework that stores video information across multiple levels of granularity: shallow layers function as sensory memory, exhibiting a strong recency bias toward newly arriving frames; deep layers act as long-term memory, focusing on frame-level rhythmic anchor tokens; and middle layers serve as transitional working memory that balances recency information with frame-level semantic representations. Our method [***HERMES***]{style="color: HermesBlue"} comprises three components: *hierarchical KV cache management*, *cross-layer memory smoothing*, and *position re-indexing*. During inference, [***HERMES***]{style="color: HermesBlue"} reuses the compact KV cache and requires no auxiliary computations or external devices upon the arrival of user queries, thereby guaranteeing real-time responses. Experiments show that [***HERMES***]{style="color: HermesBlue"} maintains stable and accurate performance with up to 68% fewer video tokens, while maintaining consistently low response latency and a constant GPU memory footprint.

To summarize, our main contributions are as follows:

1.  Grounded in a mechanistic analysis on attention visualization, we pioneer the conceptualization of KV cache as a hierarchical video memory framework across multiple granularities.

2.  We propose [***HERMES***]{style="color: HermesBlue"}, a training-free method for streaming video understanding by reusing hierarchically managed KV cache. Despite reducing video tokens by up to 68%, [***HERMES***]{style="color: HermesBlue"} achieves competitive accuracy, with gains of up to 11.4% on streaming benchmarks.

3.  [***HERMES***]{style="color: HermesBlue"} exhibits outstanding efficiency in streaming scenarios. Compared to the prior training-free SOTA method, it achieves up to a 10$\times$ speedup in latency. With a constant, compact GPU memory footprint and no auxiliary computation at query time, [***HERMES***]{style="color: HermesBlue"} ensures consistently low-latency responses.

<figure id="fig:layer_vis" data-latex-placement="!t">
<figure id="fig:shallow_vis">
<embed src="figures/vis/layer_00_attention.pdf" />
<figcaption>Shallow layer attention.</figcaption>
</figure>
<figure id="fig:deep_vis">
<embed src="figures/vis/deep.pdf" />
<figcaption>Deep layer attention.</figcaption>
</figure>
<figure id="fig:mid_vis">
<embed src="figures/vis/layer_08_attention.pdf" />
<figcaption>Middle layer attention.</figcaption>
</figure>
<figcaption>Visualization of the average attention weights (log scale) for user queries over video tokens in LLaVA-OV-7B with a FIFO KV cache budget of 6K video tokens per layer, averaged across 300 user video questions.</figcaption>
</figure>

# Layer-wise Preference for Hierarchical Streaming Video Information {#sec: investigation}

Sliding Window is a standard paradigm for streaming video processing by incrementally encoding the continuous video stream chunk by chunk. When KV cache reaches the pre-defined memory budget, token eviction is triggered, and deciding which tokens to keep is crucial for stable understanding. Existing methods [@di2025streamingvideoquestionansweringincontext; @yang2025streammemqueryagnostickvcache; @xu2025streamingvlmrealtimeunderstandinginfinite] rely on coarse-grained eviction strategies such as FIFO uniformly across all layers, overlooking layer-wise attention preferences.

To fill this gap, we conduct a mechanistic investigation of attention preferences in MLLM decoder layers, revealing how layers specialize in storing multiple-granularity video memory. To derive generalized insights, we randomly sample 100 video-question pairs from each of the short (62s[^1] - 141s), medium (251s - 1,092s) and long (1,795s - 3,579s) duration subsets of the VideoMME benchmark [@fu2025videommefirstevercomprehensiveevaluation] to cover diverse video durations and user queries. The video samples are uniformly sampled at 0.5 fps and subsequently fed into LLaVA-OV-7B in a streaming chunk-wise manner, with each chunk containing 8 frames. LLaVA-OV-7B consists of 28 decoder layers, and each video frame is uniformly encoded into 196 visual tokens. During the prefilling stage for video tokens, we maintain a constant budget $|M|$ of 6K video tokens per KV cache layer. After each eviction step, the positional indices of tokens per KV cache layer are re-indexing to contiguous \[0, $|M|$).

Layer-wise attention visualizations over video tokens maintained in a FIFO KV cache in [8](#fig:layer_vis){reference-type="ref+label" reference="fig:layer_vis"} reveal three general stages of attention preference, along with more visualization results presented in [\[app:attn_vis\]](#app:attn_vis){reference-type="ref+label" reference="app:attn_vis"}:

- **Shallow Layers as Sensory Memory**: As shown in [5](#fig:shallow_vis){reference-type="ref+label" reference="fig:shallow_vis"}, the shallow layers (e.g., layer 0) exhibit an intense recency bias, with attention sharply concentrated on the most recent visual tokens and rapidly decaying over earlier ones. This behavior aligns with the concept of *Sensory Memory* [@ATKINSON196889; @shan2025cognitivememorylargelanguage]: shallow layers function as a short-lived buffer for the most recent visual inputs, enabling the model to quickly perceive incoming frames.

- **Deep Layers as Long-term Memory**: In deep layers (e.g., layer 26 in [6](#fig:deep_vis){reference-type="ref+label" reference="fig:deep_vis"}), recency bias largely disappears. Instead, the attention pattern becomes highly sparse and rhythmic, with local extrema appearing at regular intervals. These extrema are exactly N = 196 tokens apart, matching to the number of tokens encoding a single frame in LLaVA-OV-7B. These local maxima can be regarded as frame-level \"anchor tokens\", summarizing the visual information of each frame. This pattern reflects *Long-term Memory* [@ATKINSON196889; @shan2025cognitivememorylargelanguage]: deep layers store critical frame-level semantic representations for long-horizon understanding.

- **Middle Layers as Working Memory**: Middle layers (e.g., layer 8 in [7](#fig:mid_vis){reference-type="ref+label" reference="fig:mid_vis"}) exhibit a gradual reduction in recency bias, with attention more evenly distributed across recent and earlier tokens. Simultaneously, the attention begins to transition toward the rhythmic patterns in the deep layers. This behavior corresponds to *Working Memory* [@BADDELEY197447; @hu2025memoryageaiagents]: middle layers integrate recent and earlier visual information, bridging short-term sensory traces with frame-level semantic summaries.

<figure id="fig:hermes" data-latex-placement="t">
<embed src="figures/HERMES.pdf" />
<figcaption>Overview of the <span style="color: HermesBlue"><em><strong>HERMES</strong></em></span> architecture for streaming video QA. By implementing a hierarchical KV cache and specialized management strategies, <span style="color: HermesBlue"><em><strong>HERMES</strong></em></span> enables real-time and accurate responses through direct cache reuse, eliminating the need for additional retrieval operations or external memory whenever users pose questions.</figcaption>
</figure>

# HERMES {#sec:method}

We propose [***HERMES***]{style="color: HermesBlue"}, a training-free framework that can be seamlessly integrated with MLLMs. As shown in [9](#fig:hermes){reference-type="ref+label" reference="fig:hermes"}, [***HERMES***]{style="color: HermesBlue"} has three components: hierarchical KV cache management, cross-layer memory smoothing, and position re-indexing.

## Hierarchical KV Cache Management {#sec:hierarchical}

Motivated by the layer-wise attention patterns identified in [2](#sec: investigation){reference-type="ref+label" reference="sec: investigation"}, we design a hierarchical KV cache strategy. For each video token with KV cache index $i$ at layer $l$, where $i$ denotes its physical position in KV cache, we compute an importance score $S_i^l$ to decide its retention:

- **Shallow Layers**: They act as sensory memory with strong recency bias. Inspired by Ebbinghaus' memory decay theory [@ebbinghaus2013memory], we model token importance using an exponential forgetting curve based on temporal distance: $$\begin{equation}
  \label{eq:shallow}
  S_i^l = \alpha_i^l \cdot e^{-k\Delta t_i}, \Delta t_i = T - 1 - i,
  \end{equation}$$ where $T$ is the total number of video tokens in the cache, $k > 0$ is the forgetting rate, $\alpha_i^l$ denotes the normalization factor.

- **Deep Layers**: Deep layers function as frame-level long-term memory with stable anchor tokens. Their attention distributions are sparse, and these anchor tokens consistently receive high attention across frames, making attention magnitude a reliable indicator of long-term importance. We therefore compute token importance directly from attention weights with respect to the user query. To handle unpredictable queries in streaming scenarios, we use a generic guidance prompt (see [8](#app:prompt){reference-type="ref+label" reference="app:prompt"}) as a pseudo query. Token importance is computed as: $$\begin{equation}
  \label{eq:deep}
  S_i^l = \alpha_i^l \cdot W_i^l,
  \end{equation}$$ where $W_i^l$ denotes the attention weight of the $i$-th token at the layer $l$.

- **Middle Layers**: Middle layers serve as working memory, transitioning from recency-dominated shallow layers to attention-driven deep layers. We compute importance by interpolating recency and attention with a layer-dependent weight: $$\begin{equation}
  \omega^l = \omega_0 - \gamma \cdot \frac{l - l_{\text{short}}}{l_{\text{long}} - l_{\text{short}}},
  \end{equation}$$ where $l_{\text{short}}$ and $l_{\text{long}}$ denote the layer indices, with $\omega_0 = 0.75$ and $\gamma = 0.6$. The importance score of token $i$ at layer $l$ is then computed as $$\begin{equation}
  S_i^l = (1 - \omega^l)\,A_i^l + \omega^l\, R_i^l,
  \end{equation}$$ where $A_i^l$ and $R_i^l$ denote the normalized attention weight and recency score, respectively, computed as in [\[eq:deep,eq:shallow\]](#eq:deep,eq:shallow){reference-type="ref+label" reference="eq:deep,eq:shallow"}.

## Cross-Layer Memory Smoothing {#sec: smoothing}

Hierarchical KV cache management may introduce cross-layer inconsistency, as tokens at the same cache index can be evicted independently across layers, leading to misaligned visual memory. Since effective LLM memory relies on cross-layer interaction [@packer2024memgptllmsoperatingsystems; @behrouz2024titanslearningmemorizetest; @sun2025hierarchicalmemoryhighefficiencylongterm; @hu2025memoryageaiagents], we address this issue with *Cross-Layer Memory Smoothing*.

Instead of treating video tokens at the same KV cache index as independent across layers, we propagate and smooth importance signals from deeper to shallower layers. Given raw importance scores $S_i^l$, the smoothed score is computed as: $$\begin{equation}
\tilde{S_i^l} = (1 - {\lambda}_l) \cdot S_i^l + \lambda_l \cdot S_i^{l+1},
\end{equation}$$ $\lambda \in [0,1]$ is the smoothing hyperparameter that controls the strength of cross-layer smoothing.

We then apply Top-K selection based on $\tilde{S}_i^l$ to maintain a fixed memory budget $|M|$ per layer: $$\begin{equation}
\begin{aligned}
\mathcal{I}_l &= \mathrm{TopK}(\tilde{S}_l, |M|), \\
K_l &= K_l[\mathcal{I}_l], \quad
V_l = V_l[\mathcal{I}_l].
\end{aligned}
\end{equation}$$

To preserve long-term information, evicted tokens are aggregated into a **summary token** per layer, which compactly encodes long-term memory and is retained in the KV cache (see [\[alg:summary\]](#alg:summary){reference-type="ref+label" reference="alg:summary"}).

## Position Re-Indexing {#sec:pos}

Continuous accumulation of streaming inputs causes positional indices to exceed the model's maximum supported range, severely degrading text generation quality. To stabilize inference, we apply position re-indexing, which remaps positional indices to a contiguous range $[0, |M|)$ within the memory budget $|M|$. We design two strategies:

#### Lazy Re-Indexing

Re-indexing is triggered only when positional indices approach the model limit, resulting in lower computational overhead. By preserving the original positional indices of recent tokens, it prevents positional drift compared to eager re-indexing, making it well suited for streaming video understanding.

#### Eager Re-Indexing

Re-indexing is performed at each compression step, maintaining strictly contiguous RoPE indices in KV cache. While this strategy stabilizes long-range visual semantics [@kim2024infinipotinfinitecontextprocessing; @kim2025infinipotvmemoryconstrainedkvcache; @xu2025streamingvlmrealtimeunderstandinginfinite], it leads to higher computational cost due to frequent re-indexing, making it more suitable for offline videos.

The details of re-indexing implementation for 1D RoPE (LLaVA-OV) and 3D M-RoPE (Qwen2.5-VL) are illustrated in [11.1](#app:1d_pos){reference-type="ref+label" reference="app:1d_pos"} and  [11.2](#app:3d_pos){reference-type="ref+label" reference="app:3d_pos"}, respectively.

# Experiments {#sec:experiments}

## Experimental Setup

#### Benchmarks.

We evaluate [***HERMES***]{style="color: HermesBlue"} on diverse streaming and offline benchmarks. For streaming understanding, we use StreamingBench [@lin2024streamingbenchassessinggapmllms], OVO-Bench [@li2025ovobenchfarvideollmsrealworld] and RVS (including RVS-Ego and EVS-Movie) [@zhang2024flashvstreammemorybasedrealtimeunderstanding]. For offline video evaluation, we adopt one short video dataset MVBench [@li2024mvbenchcomprehensivemultimodalvideo], along with two long video datasets, VideoMME [@fu2025videommefirstevercomprehensiveevaluation] and Egoschema [@mangalam2023egoschemadiagnosticbenchmarklongform]. We conduct evaluation on the official dev split of Egoschema and report VideoMME results without subtitles. Our benchmark selection covers both multiple-choice and open-ended questions as QA form. The details of utilized benchmarks are demonstrated in [10](#app:details_of_benchmarks){reference-type="ref+label" reference="app:details_of_benchmarks"}.

#### Models.

To further verify the broad applicability of our method, we select two popular open-source MLLM series, LLaVA-OneVision (LLaVA-OV) [@li2024llavaonevisioneasyvisualtask] and Qwen2.5-VL [@bai2025qwen25vltechnicalreport]. Each is tested across two different parameter scales, covering a large range from 0.5B to 32B. For Qwen2.5-VL, we maintain its native dynamic resolution on video input, ensuring a fair comparison with the base model.

#### Implementation Details.

For evaluating [***HERMES***]{style="color: HermesBlue"} across all benchmarks, each video is encoded and processed chunk by chunk, with 16 frames per chunk, and sequentially prefilling the backbone LLM. Then, token compression is triggered once the predefined memory budget is exceeded.

For the layer partition, we follow the mechanistic investigations presented in  [2](#sec: investigation){reference-type="ref+label" reference="sec: investigation"}: 10% shallow, 60% middle and 30% deep layers. A more comprehensive analysis of attention behaviors as supportive evidence can be found in [14](#fig:more_vis){reference-type="ref+label" reference="fig:more_vis"}. The cross-layer memory smoothing hyperparameter $\lambda$ proposed in [3.2](#sec: smoothing){reference-type="ref+label" reference="sec: smoothing"} is layer-dependent, with detailed configurations reported in [9](#app:smooth_config){reference-type="ref+label" reference="app:smooth_config"}.

All evaluations are conducted using FP16 mixed precision and efficiency tests are conducted on a single A800 GPU, consistent with prior works [@di2025streamingvideoquestionansweringincontext; @chen2025streamingtomstreamingtokencompression]. Greedy decoding is used to generate deterministic outputs. Accuracy evaluations can be completed on one H200 GPU.

## Main Results

#### Streaming Video Understanding

::: table*
+:---------------------------------------------------------------+:-----------:+:------------------:+:---------:+:---------:+:---------:+
| **Model**                                                      | **#Frames** | **StreamingBench** | **OVO-Bench**                     |
|                                                                |             +--------------------+-----------+-----------+-----------+
|                                                                |             | Real-Time          | Real-Time | Backward  | **Avg.**  |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| Human                                                          | \-          | 91.46              | 93.20     | 92.33     | 92.83     |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| **Proprietary MLLMs**                                                                                                                 |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| Gemini 1.5 pro [@gemini25]                                     | 1 fps       | 75.69              | 69.32     | 62.54     | 66.41     |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| GPT-4o [@openai2024gpt4ocard]                                  | 64          | 73.28              | 64.46     | 60.75     | 62.87     |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| Claude 3.5 Sonnet [@claude3_5]                                 | 20          | 72.44              | \-        | \-        | \-        |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| **Open-source Offline MLLMs**                                                                                                         |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| Video-LLaMA2-7B [@videollama2]                                 | 32          | 49.52              | \-        | \-        | \-        |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| VILA-1.5-8B [@vila]                                            | 14          | 52.32              | \-        | \-        | \-        |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| Video-CCAM-14B [@videoccam]                                    | 96          | 53.96              | \-        | \-        | \-        |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| LongVA-7B [@longva]                                            | 128         | 59.96              | \-        | \-        | \-        |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| Qwen2-VL-7B [@wang2024qwen2vlenhancingvisionlanguagemodels]    | 64          | 69.04              | 60.65     | 48.58     | 54.62     |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| InternVL-V2-8B [@internvl2]                                    | 16          | 63.72              | 60.73     | 44.00     | 52.37     |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| LLaVA-NeXT-Video-32B [@llava-next]                             | 64          | 66.96              | \-        | \-        | \-        |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| MiniCPM-V-2.6-8B [@minicpm]                                    | 32          | 67.44              | \-        | \-        | \-        |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
|                                                                |             |                    |           |           |           |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| Flash-VStream-7B [@flashvstream]                               | \-          | 23.23              | 29.86     | 25.35     | 27.61     |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| VideoLLM-online-8B [@videollmonline]                           | 2 fps       | 35.99              | 20.79     | 17.73     | 19.26     |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| Dispider-7B [@dispider]                                        | 1 fps       | 67.63              | 54.55     | 36.06     | 45.31     |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| TimeChat-Online-7B [@timechatonline]                           | 1 fps       | 75.36              | 61.90     | 41.70     | 51.80     |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| StreamForest-7B [@zeng2025streamforestefficientonlinevideo]    | 1 fps       | 77.26              | 61.20     | 52.02     | 56.61     |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| **Training-free Offline-to-Online Methods**                                                                                           |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| LLaVA-OV-7B [@li2024llavaonevisioneasyvisualtask]              | 64          | 71.34              | 63.06     | 43.64     | 53.35     |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| \+ ReKV [@di2025streamingvideoquestionansweringincontext]      | 0.5 fps     | 69.22              | 57.33     | 44.16     | 50.75     |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| \+ LiveVLM [@ning2025livevlmefficientonlinevideo]              | 0.5 fps     | 72.92              | \-        | \-        | \-        |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| \+ StreamKV [@chen2025streamkvstreamingvideoquestionanswering] | 0.5 fps     | 68.80              | \-        | \-        | \-        |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| \+ HERMES (6K tokens)                                          | 0.5 fps     | 72.63              | 65.07     | 48.80     | 56.94     |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| \+ HERMES (4K tokens)                                          | 0.5 fps     | **73.23**          | **66.34** | **50.20** | **58.27** |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| LLaVA-OV-0.5B [@li2024llavaonevisioneasyvisualtask]            | 64          | 59.64              | 49.70     | 34.59     | 42.15     |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| \+ ReKV [@di2025streamingvideoquestionansweringincontext]      | 0.5 fps     | 57.39              | 43.77     | 33.06     | 38.42     |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| \+ HERMES (6K tokens)                                          | 0.5 fps     | 61.04              | 50.34     | 34.75     | 42.55     |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| \+ HERMES (4K tokens)                                          | 0.5 fps     | **62.04**          | **50.72** | **34.80** | **42.76** |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| Qwen2.5-VL-7B [@bai2025qwen25vltechnicalreport]                | 1 fps       | 73.31              | 59.90     | 44.65     | 52.28     |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| \+ HERMES (6K tokens)                                          | 1 fps       | 78.72              | 68.42     | 48.10     | 58.26     |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| \+ HERMES (4K tokens)                                          | 1 fps       | **79.44**          | **68.98** | **49.43** | **59.21** |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| Qwen2.5-VL-32B [@bai2025qwen25vltechnicalreport]               | 1 fps       | 74.27              | 64.40     | 50.33     | 57.37     |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| \+ HERMES (6K tokens)                                          | 1 fps       | **80.20**          | 71.93     | **57.71** | **64.82** |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
| \+ HERMES (4K tokens)                                          | 1 fps       | 80.08              | **72.37** | 55.42     | 63.90     |
+----------------------------------------------------------------+-------------+--------------------+-----------+-----------+-----------+
:::

::: minipage
:::

::: minipage
:::

Extensive experiments on streaming benchmarks reveal the key findings:

\(1\) *[***HERMES***]{style="color: HermesBlue"} outperforms on multiple-choice streaming datasets, showing exceptional real-time understanding and backward tracing capabilities*. As shown in [\[tab:streaming_main\]](#tab:streaming_main){reference-type="ref+label" reference="tab:streaming_main"}, it achieves state-of-the-art performance on StreamingBench and OVO-Bench, significantly surpassing base models and training-free baselines. Built on Qwen2.5-VL-7B, [***HERMES***]{style="color: HermesBlue"} reaches 79.44% and 59.21% accuracy using only 4K video tokens, improving over Qwen2.5-VL-7B by 6.13% and 6.93%, while outperforming all 7B-scale open-source online and offline models. Full results on StreamingBench and OVO-Bench are shown in [\[tab:streamingbench_full\]](#tab:streamingbench_full){reference-type="ref+label" reference="tab:streamingbench_full"} and [\[tab:ovobench_full\]](#tab:ovobench_full){reference-type="ref+label" reference="tab:ovobench_full"} respectively.

\(2\) *[***HERMES***]{style="color: HermesBlue"} excels on open-ended streaming tasks, showing fine-grained temporal and spatial comprehension*. On RVS-Ego and RVS-Movie ([\[tab:rvs\]](#tab:rvs){reference-type="ref+label" reference="tab:rvs"}), we evaluate the model answer by GPT-3.5-turbo-0125 on accuracy and score (1--5 scale), consistent with compared baselines. [***HERMES***]{style="color: HermesBlue"} consistently surpasses all prior training-free methods and improves accuracy by up to 11.4% over the base model with uniformly sampled 64 frames. These extensive experiments demonstrate [***HERMES***]{style="color: HermesBlue"}'s strong abilities in various streaming tasks, as well as its general applicability across foundation models. Moreover, we provide case studies from RVS benchmark, showing finer-grained temporal (shown in [15](#fig:case_temporal){reference-type="ref+label" reference="fig:case_temporal"}) and spatial understanding (shown in  [16](#fig:case_spatial){reference-type="ref+label" reference="fig:case_spatial"}) abilities of [***HERMES***]{style="color: HermesBlue"} than its base model.

#### Offline Video Understanding

::: {#tab:offline_main}
+:------------------------------------------------------------+:-----------:+:-----------:+:-------------:+:---------:+:---------:+
| **Model**                                                   | **#Frames** | **MVBench** | **Egoschema** | **VideoMME**          |
|                                                             |             +-------------+---------------+-----------+-----------+
|                                                             |             |             |               | Long      | **Avg.**  |
+-------------------------------------------------------------+-------------+-------------+---------------+-----------+-----------+
| **Proprietary MLLMs**                                                                                                           |
+-------------------------------------------------------------+-------------+-------------+---------------+-----------+-----------+
| Gemini 1.5 pro [@gemini25]                                  | 1 fps       | 75.69       | 69.32         | 62.54     | 66.41     |
+-------------------------------------------------------------+-------------+-------------+---------------+-----------+-----------+
| GPT-4o [@openai2024gpt4ocard]                               | 64          | 73.28       | 64.46         | 60.75     | 62.87     |
+-------------------------------------------------------------+-------------+-------------+---------------+-----------+-----------+
| Claude 3.5 Sonnet [@claude3_5]                              | 20          | 72.44       | \-            | \-        | \-        |
+-------------------------------------------------------------+-------------+-------------+---------------+-----------+-----------+
| **Open-source Offline MLLMs**                                                                                                   |
+-------------------------------------------------------------+-------------+-------------+---------------+-----------+-----------+
| Video-LLaMA2-7B [@videollama2]                              | 32          | 49.52       | \-            | \-        | \-        |
+-------------------------------------------------------------+-------------+-------------+---------------+-----------+-----------+
| VILA-1.5-8B [@vila]                                         | 14          | 52.32       | \-            | \-        | \-        |
+-------------------------------------------------------------+-------------+-------------+---------------+-----------+-----------+
| Video-CCAM-14B [@videoccam]                                 | 96          | 53.96       | \-            | \-        | \-        |
+-------------------------------------------------------------+-------------+-------------+---------------+-----------+-----------+
| LongVA-7B [@longva]                                         | 128         | 59.96       | \-            | \-        | \-        |
+-------------------------------------------------------------+-------------+-------------+---------------+-----------+-----------+
| LLaVA-Video-7B [@zhang2025llavavideovideoinstructiontuning] | 32          | 58.60       | 57.3          | \-        | 63.30     |
+-------------------------------------------------------------+-------------+-------------+---------------+-----------+-----------+
| Qwen2-VL-7B [@wang2024qwen2vlenhancingvisionlanguagemodels] | 64          | 67.00       | 66.70         | \-        | 63.30     |
+-------------------------------------------------------------+-------------+-------------+---------------+-----------+-----------+
| InternVL-V2-8B [@internvl2]                                 | 16          | 65.80       | \-            | \-        | 56.30     |
+-------------------------------------------------------------+-------------+-------------+---------------+-----------+-----------+
| Kangaroo-7B [@kangaroo]                                     | 64          | 64.60       | \-            | \-        | \-        |
+-------------------------------------------------------------+-------------+-------------+---------------+-----------+-----------+
| LLaVA-NeXT-Video-32B [@llava-next]                          | 64          | 66.96       | \-            | \-        | \-        |
+-------------------------------------------------------------+-------------+-------------+---------------+-----------+-----------+
| MiniCPM-V-2.6-8B [@minicpm]                                 | 32          | 67.44       | \-            | \-        | \-        |
+-------------------------------------------------------------+-------------+-------------+---------------+-----------+-----------+
|                                                             |             |             |               |           |           |
+-------------------------------------------------------------+-------------+-------------+---------------+-----------+-----------+
| Dispider-7B [@dispider]                                     | 1 fps       | \-          | 55.60         | \-        | 57.20     |
+-------------------------------------------------------------+-------------+-------------+---------------+-----------+-----------+
| TimeChat-Online-7B [@timechatonline]                        | 1 fps       | 75.36       | 61.90         | 41.70     | 53.22     |
+-------------------------------------------------------------+-------------+-------------+---------------+-----------+-----------+
| StreamForest-7B [@zeng2025streamforestefficientonlinevideo] | 1 fps       | 70.20       | \-            | \-        | 61.40     |
+-------------------------------------------------------------+-------------+-------------+---------------+-----------+-----------+
| **Training-free Offline-to-Online Methods**                                                                                     |
+-------------------------------------------------------------+-------------+-------------+---------------+-----------+-----------+
| LLaVA-OV-7B [@li2024llavaonevisioneasyvisualtask]           | 64          | **57.02**   | 59.93         | 48.00     | 57.67     |
+-------------------------------------------------------------+-------------+-------------+---------------+-----------+-----------+
| \+ ReKV [@di2025streamingvideoquestionansweringincontext]   | 0.5 fps     | 56.83       | **60.70**     | 46.89     | 57.74     |
+-------------------------------------------------------------+-------------+-------------+---------------+-----------+-----------+
| \+ HERMES (6K tokens)                                       | 0.5 fps     | 56.95       | 60.23         | 49.11     | 58.44     |
+-------------------------------------------------------------+-------------+-------------+---------------+-----------+-----------+
| \+ HERMES (4K tokens)                                       | 0.5 fps     | 56.92       | 60.29         | **49.22** | **58.85** |
+-------------------------------------------------------------+-------------+-------------+---------------+-----------+-----------+
| Qwen2.5-VL-7B [@bai2025qwen25vltechnicalreport]             | 1 fps       | 65.00       | 58.47         | 53.89     | **64.52** |
+-------------------------------------------------------------+-------------+-------------+---------------+-----------+-----------+
| \+ HERMES (6K tokens)                                       | 1 fps       | 65.40       | 59.47         | **54.44** | 62.00     |
+-------------------------------------------------------------+-------------+-------------+---------------+-----------+-----------+
| \+ HERMES (4K tokens)                                       | 1 fps       | **65.53**   | **59.97**     | 53.44     | 60.63     |
+-------------------------------------------------------------+-------------+-------------+---------------+-----------+-----------+

: Performance comparison (%) on offline benchmarks.
:::

The results presented in [1](#tab:offline_main){reference-type="ref+label" reference="tab:offline_main"} demonstrate the *competitive performance of [***HERMES***]{style="color: HermesBlue"} across multiple temporal scales on offline benchmarks*, compared to the base model and other training-free methods. Under a limited budget of video tokens, [***HERMES***]{style="color: HermesBlue"} achieves performance that is better than or comparable to the corresponding base models. [***HERMES***]{style="color: HermesBlue"} based on LLaVA-OV-7B surpasses the base model on long video datasets Egoschema and VideoMME, achieving 60.29% and 58.85%, respectively, and attains 56.92% accuracy on the short video dataset MVBench, which is comparable to the base model's 57.02%.

## Efficiency Analysis

<figure id="fig:efficency_compare" data-latex-placement="ht">
<embed src="figures/efficiency_compare_new.pdf" />
<figcaption>GPU memory and TTFT latency comparison across input frame numbers. <span style="color: HermesBlue"><em><strong>HERMES</strong></em></span> achieves 10 <span class="math inline">×</span> faster in TTFT compared to prior SOTA.</figcaption>
</figure>

To evaluate the efficiency of [***HERMES***]{style="color: HermesBlue"}, we utilize three metrics: peak GPU memory usage, Time to First Token (TTFT), defined as the latency measured from the moment a user inputs a query to the decoding of the first output token, and Time Per Output Token (TPOT) across varying numbers of input frames. All experiments are conducted using LLaVA-OV-7B as the base model with a 4K-token memory budget.  [10](#fig:efficency_compare){reference-type="ref+label" reference="fig:efficency_compare"} shows the comparison of memory usage and TTFT among [***HERMES***]{style="color: HermesBlue"} and representative streaming methods. Unlike Dispider and LiveVLM, [***HERMES***]{style="color: HermesBlue"} consistently maintains stable memory usage and TTFT as frames increase. Notably, under the 256-frame setting, [***HERMES***]{style="color: HermesBlue"} achieves 1.04$\times$ reduction in peak memory compared to the prior SOTA LiveVLM, while achieving an impressive 10$\times$ speedup in TTFT over the prior SOTA StreamingTOM.

We further examine the efficiency of [***HERMES***]{style="color: HermesBlue"} under varying encoded video chunk sizes, with the results shown in [\[tab:efficiency_chunk\]](#tab:efficiency_chunk){reference-type="ref+label" reference="tab:efficiency_chunk"}. GPU memory usage does not increase with longer video lengths due to the fixed memory budget. TTFT and TPOT remain consistently low across varying video lengths and encoding chunk sizes, confirming real-time responsiveness in practical streaming scenarios.

## Ablation Study

We conduct ablation studies to evaluate the contributions of [***HERMES***]{style="color: HermesBlue"}'s components and hyperparameter choices, covering: (1) KV cache memory budget, (2) cross-layer memory smoothing and its hyperparameters, (3) position re-indexing strategies for streaming and offline datasets, and (4) summary tokens for long-term memory retention.

#### Memory Budget {#sec:memory_ablation}

<figure id="fig:memory_budget_qwen" data-latex-placement="ht">
<figure id="fig:memory_budget_llava">
<embed src="figures/llava_budget_ablation.pdf" />
<figcaption>Performance comparison of LLaVA-OV-7B across different memory budgets.</figcaption>
</figure>
<figure id="fig:memory_budget_qwen">
<embed src="figures/qwen_budget_ablation.pdf" />
<figcaption>Performance Comparison of Qwen2.5-VL-7B across Different Memory Budgets.</figcaption>
</figure>
<figcaption>Performance Comparison of Qwen2.5-VL-7B across Different Memory Budgets.</figcaption>
</figure>

To investigate the impact of memory budget on understanding performance, we conduct ablations by varying the memory budget $|M|$ from 1K to 10K. As shown in [11](#fig:memory_budget_llava){reference-type="ref+label" reference="fig:memory_budget_llava"}, for [***HERMES***]{style="color: HermesBlue"} built upon LLaVA-OV-7B, the performance on both streaming and offline datasets stabilizes once memory budget reaches 4K. Notably, streaming datasets can tolerate a smaller memory budget. In contrast, the performance on long offline datasets degrades significantly when the memory budget is below 4K. The ablation on Qwen2.5-VL-7B is provided in [13](#fig:memory_budget_qwen){reference-type="ref+label" reference="fig:memory_budget_qwen"}, yielding conclusions consistent with those on LLaVA-OV-7B.

::: minipage
:::

::: minipage
:::

#### Cross-Layer Memory Smoothing {#cross-layer-memory-smoothing}

In [\[tab:lambda_ablation\]](#tab:lambda_ablation){reference-type="ref+label" reference="tab:lambda_ablation"}, we evaluate variants without the proposed cross-layer memory smoothing mechanism, as well as alternative hyperparameter configurations. All these variants exhibit degraded performance on the VideoMME benchmark, demonstrating both the critical role of memory smoothing and the effectiveness of our chosen hyperparameter settings.

::: minipage
:::

::: minipage
:::

#### Position Re-Indexing Strategies

For all streaming evaluations, we adopt the lazy position re-indexing strategy, while we use the eager re-indexing strategy for offline evaluations. Ablation studies in [\[tab:pos_ablation_streaming\]](#tab:pos_ablation_streaming){reference-type="ref+label" reference="tab:pos_ablation_streaming"} and [\[tab:pos_ablation_offline\]](#tab:pos_ablation_offline){reference-type="ref+label" reference="tab:pos_ablation_offline"} show the effectiveness of these strategies in their respective scenarios.

#### Summary Tokens in Deep Layers

In [3.2](#sec: smoothing){reference-type="ref+label" reference="sec: smoothing"}, we aggregate the evicted tokens in each deep layer into one summary token at each compression step. The results in [\[tab:summary_token_ablation\]](#tab:summary_token_ablation){reference-type="ref+label" reference="tab:summary_token_ablation"} indicate that these summary tokens effectively preserve long-term memory, leading to improved performance on VideoMME.

# Related Work {#sec:related}

#### Streaming Video Understanding

Existing MLLMs [@gemini25; @li2024llavaonevisioneasyvisualtask; @bai2025qwen25vltechnicalreport; @bai2025qwen3vltechnicalreport] are primarily designed for pre-defined offline videos and struggle with continuous streaming videos. While some prior works have adapted existing offline MLLMs to online settings [@timechatonline; @zeng2025streamforestefficientonlinevideo; @xu2025streamingvlmrealtimeunderstandinginfinite], they rely on costly model-specific training. Training-free streaming methods, such as ReKV [@di2025streamingvideoquestionansweringincontext] and LiveVLM [@ning2025livevlmefficientonlinevideo], prefill offload KV cache to external devices. At user query time, they retrieve the full KV cache and reconstruct it on the GPU, incurring high latency and overall memory usage. In contrast, StreamMem [@yang2025streammemqueryagnostickvcache] heuristically reuses KV cache, but lacks fine-grained KV cache management and interpretability. Unlike prior training-free methods, [***HERMES***]{style="color: HermesBlue"} is grounded in a systematic attention analysis with improved interpretability and reliability.

#### KV Cache Compression for Video Input

Numerous KV cache compression techniques have been proposed for offline video understanding [@yang2024visionziplongerbetternecessary; @wang2024dynamicvlmsimpledynamicvisual; @wang2025videotreeadaptivetreebasedvideo; @tao2025dycokedynamiccompressiontokens], but most of these methods are poorly suited for streaming scenarios due to the unpredictable future frames and user queries [@chen2025streamingtomstreamingtokencompression]. Existing online KV cache compression paradigms [@di2025streamingvideoquestionansweringincontext; @ning2025livevlmefficientonlinevideo; @yang2025streammemqueryagnostickvcache; @chen2025streamingtomstreamingtokencompression] largely overlook the inherently hierarchical storage structure of the KV cache. [***HERMES***]{style="color: HermesBlue"} addresses this gap by introducing a hierarchical KV cache management strategy, which enables fine-grained memory utilization and low-latency responses.

# Conclusion {#sec:conclusion}

This paper proposes [***HERMES***]{style="color: HermesBlue"}, a training-free framework for efficient streaming video understanding. Guided by mechanistic attention analysis, we conceptualizes KV cache as a hierarchical video memory system across multiple granularities. By introducing a cross-layer memory smoothing and position re-indexing, [***HERMES***]{style="color: HermesBlue"} further enhances the understanding performance for long streaming input. Extensive experiments demonstrate that [***HERMES***]{style="color: HermesBlue"} delivers accurate performance under continuously growing video streams, while consistently maintaining extremely low response latency and compact GPU memory usage, making it well suited for real-world streaming deployment.

# Appendix Contents {#appendix-contents .unnumbered}

# More Attention Visualization

We provide more detailed attention visualization in [14](#fig:more_vis){reference-type="ref+label" reference="fig:more_vis"} under different sliding window sizes, showing that the observed attention patterns consistently hold across varying window lengths, thus confirming the generality of the findings in [2](#sec: investigation){reference-type="ref+label" reference="sec: investigation"}.

[]{#app:attn_vis label="app:attn_vis"}

<figure id="fig:more_vis" data-latex-placement="htbp">
<figure>
<embed src="figures/vis/sampled_layers_summary_4k.pdf" />
<figcaption>Sliding window of 4,000 video tokens</figcaption>
</figure>
<figure>
<embed src="figures/vis/sampled_layers_summary_6k.pdf" />
<figcaption>Sliding window of 6,000 video tokens</figcaption>
</figure>
<figure>
<embed src="figures/vis/sampled_layers_summary_1w.pdf" />
<figcaption>Sliding window of 10,000 video tokens</figcaption>
</figure>
<figcaption> Visualization of the average attention weights of video tokens in LLaVA-OV-7B under different sliding window sizes. </figcaption>
</figure>

# Guidance Prompt {#app:prompt}

The following two figures show the local and global guidance prompt with and without conversation history to guide the token compression, respectively. For the deep layers, since they primarily focus on frame-level global semantic information, we employ a global guidance prompt as a pseudo-query to extract attention weights of video tokens. In contrast, the middle layers lie in a transition between recency-biased attention and global semantic focus. Therefore, we adopt a hybrid guidance strategy, in which the local guidance prompt and the global guidance prompt are concatenated into a single prompt string to jointly guide the token compression.

<figure data-latex-placement="ht">
<div class="minipage">
<div class="tcolorbox">
<p>Find recent details related to: {last_conv}. Describe the current scene in detail, focusing on specific objects, fine-grained actions, and spatial relationships.</p>
</div>
</div>
<div class="minipage">
<div class="tcolorbox">
<p>Describe the current scene in detail, focusing on specific objects, fine-grained actions, and spatial relationships.</p>
</div>
</div>
<figcaption>Local guidance prompt to guide the token compression if there is no conversation history.</figcaption>
</figure>

<figure data-latex-placement="ht">
<div class="minipage">
<div class="tcolorbox">
<p>Context summary: {last_conv}. Summarize the video narrative, identifying main characters, key events, timeline changes, and the overall theme.</p>
</div>
</div>
<div class="minipage">
<div class="tcolorbox">
<p>Summarize the video narrative, identifying main characters, key events, timeline changes, and the overall theme.</p>
</div>
</div>
<figcaption>Global guidance prompt to guide the token compression if there is no conversation history.</figcaption>
</figure>

# Configuration of Cross-Layer Memory Smoothing {#app:smooth_config}

Given that long-term memory tends to remain relatively stable, while short-term memory focuses on diverse perception, we set different $\lambda$ for different layer stages: $$\begin{equation}
\lambda_l =
\begin{cases}
0.1,   & \text{if } l \in \mathcal{L}_{shallow} \\
0.3,   & \text{if } l \in \mathcal{L}_{middle} \\
0.4, & \text{if } l \in \mathcal{L}_{deep}
\end{cases}
\end{equation}$$ The ablation study [\[tab:lambda_ablation\]](#tab:lambda_ablation){reference-type="ref+label" reference="tab:lambda_ablation"} shows the effectiveness of this hyperparameter choice.

# Details of evaluated benchmarks {#app:details_of_benchmarks}

::: minipage
:::

::: minipage
:::

## Streaming Benchmarks

- **StreamingBench** [@lin2024streamingbenchassessinggapmllms] assesses the streaming video understanding capabilities of MLLMs. It evaluates three core aspects: real-time visual understanding, omni-source understanding, and contextual understanding. The Real-Time Visual Understanding subset is the most extensive component, featuring 2,500 questions across 500 videos. It covers 10 tasks, such as object perception and causal reasoning. In this paper, we focus on the Real-Time Visual Understanding subset for evaluation.

- **OVO-Bench** [@li2025ovobenchfarvideollmsrealworld] evaluates the online reasoning and temporal awareness of MLLMs, featuring 644 videos with approximately 2,800 fine-grained multiple-choice QA pairs. It organizes 12 tasks into three distinct categories, which are real-time visual perception, backward tracing, and forward active responding. Given that we do not focus on the proactive responding ability of MLLMs in this paper, we exclusively utilize the real-time perception and the backward tracing subsets.

- **RVS-Ego** and **RVS-Movie** [@zhang2024flashvstreammemorybasedrealtimeunderstanding] are designed to evaluate the real-time understanding capabilities of models in online streaming scenarios. The datasets consist of 10 long ego-centric videos from the Ego4D dataset [@grauman2022ego4dworld3000hours] and 22 long movie clips from the MovieNet dataset [@huang2020movienetholisticdatasetmovie] dataset, totaling over 21 hours of video content.

## Offline Benchmarks

- **MVBench** [@li2024mvbenchcomprehensivemultimodalvideo] systematically evaluates the temporal understanding capabilities of MLLMs. It utilizes a novel static-to-dynamic method to define 20 distinct temporal tasks, such as action sequence and moving direction, which cannot be effectively solved with a single frame. The videos are collected from a wide range of datasets, including NTU RGB+D [@shahroudy2016nturgbdlargescale], Perception [@pătrăucean2023perceptiontestdiagnosticbenchmark], etc.

- **Egoschema** [@mangalam2023egoschemadiagnosticbenchmarklongform] is a diagnostic benchmark designed to assess long-form video understanding abilities. Derived from Ego4D [@grauman2022ego4dworld3000hours], it consists of over 5,000 human-curated multiple-choice QA pairs associated with egocentric video clips.

- **VideoMME** [@fu2025videommefirstevercomprehensiveevaluation] is a full-spectrum, multimodal benchmark designed for the comprehensive evaluation of MLLMs in video analysis. It comprises 900 manually curated videos spanning six primary domains and diverse durations to assess temporal adaptability. The dataset features 2,700 high-quality QA pairs that necessitate processing multimodal inputs, including video frames, subtitles, and audio.

# Details of Position Re-Indexing

Inspired by StreamingVLM's strategy of managing positional stability in streaming scenarios [@xu2025streamingvlmrealtimeunderstandinginfinite], we adopt a unified left-compaction re-indexing scheme to eliminate positional gaps introduced by KV-cache pruning while preserving the semantic anchoring of the system prompt. Concretely, system text tokens are kept fixed to provide a stable textual anchor, whereas retained video tokens are re-indexed in a left-compact manner and placed contiguously after the static prefix. To reuse cached key states without re-computation, we further apply a delta-based rotary correction that compensates for the positional displacement.

## Re-indexing for LLaVA-OV (1D RoPE) {#app:1d_pos}

LLaVA-OV employs standard 1D RoPE, where each token is associated with a scalar positional index $p$. Therefore, we perform left-compaction of the 1D indices: the system prefix positions remain unchanged, while the retained positions of video tokens are reassigned to form a dense contiguous segment immediately following the fixed prefix.

Let `offset` denote the length of the system prompt prefix tokens, and let $$\mathcal{P} = \{ p_0 < p_1 < \cdots < p_{N-1} \}$$ be the sorted set of retained video token positions (excluding the fixed prefix). For a retained video token originally at position $p_{\mathrm{old}} \in \mathcal{P}$, its compacted 1D position is defined as $$\begin{equation}
p_{\mathrm{new}}
=
\texttt{offset}
+
\operatorname{rank}_{\mathcal{P}}\!\left(p_{\mathrm{old}}\right).
\end{equation}$$ This mapping removes gaps while preserving the original temporal ordering along the stream, and ensures that the video region occupies a dense range directly after the static text region.

To align cached key states with the updated positions, we avoid re-generating keys and instead apply a rotary delta correction induced by the positional shift. For a cached key vector $\mathbf{k}_{\mathrm{old}}$ associated with position $p_{\mathrm{old}}$ and remapped to $p_{\mathrm{new}}$, we compute $$\begin{equation}
\mathbf{k}_{\mathrm{new}}
=
\mathbf{k}_{\mathrm{old}}
\odot
\mathrm{RotaryDelta}\!\left(p_{\mathrm{old}}, p_{\mathrm{new}}\right),
\end{equation}$$ where the relative phase shift is $$\begin{equation}
\mathrm{RotaryDelta}\!\left(p_{\mathrm{old}}, p_{\mathrm{new}}\right)
=
e^{i(p_{\mathrm{new}} - p_{\mathrm{old}})\boldsymbol{\theta}},
\end{equation}$$ and $\boldsymbol{\theta}$ denotes the RoPE frequency vector. This update preserves the correctness of attention under the new indexing while enabling direct reuse of the cached KV states.

## Re-indexing for Qwen2.5-VL (3D M-RoPE) {#app:3d_pos}

For Qwen2.5-VL, video tokens are indexed by a 3D M-RoPE coordinate $\mathbf{p} = (p^{(t)}, p^{(h)}, p^{(w)})$, covering temporal and spatial dimensions. After pruning, the retained video tokens typically occupy sparse coordinates along each dimension $d \in \{t, h, w\}$. To eliminate the gaps without disturbing the monotonic ordering, we apply dimension-wise left-compaction independently along each axis, while keeping the system token prefix fixed.

Let $$\mathcal{P}^{(d)} = \{ p^{(d)}_0 < p^{(d)}_1 < \cdots < p^{(d)}_{N_{d}-1} \}$$ denote the sorted set of retained coordinates along dimension $d$. For a token originally located at $p^{(d)}_{\mathrm{old}} \in \mathcal{P}^{(d)}$, its compacted coordinate is defined by its rank within $\mathcal{P}^{(d)}$, shifted by the fixed prefix `offset`: $$\begin{equation}
p^{(d)}_{\mathrm{new}}
=
\texttt{offset}
+
\operatorname{rank}_{\mathcal{P}^{(d)}}\!\left(p^{(d)}_{\mathrm{old}}\right),
\qquad
d \in \{t, h, w\}.
\end{equation}$$ This procedure yields a dense and contiguous $(t,h,w)$ grid for the video tokens placed immediately after the static text region, thereby ensuring positional continuity while preserving the distinct semantic roles of temporal and spatial indices.

As in the 1D case, we reuse cached keys by applying a M-RoPE correction. Given a key $\mathbf{k}_{\mathrm{old}}$ associated with $$\mathbf{p}_{\mathrm{old}} = (p^{(t)}_{\mathrm{old}}, p^{(h)}_{\mathrm{old}}, p^{(w)}_{\mathrm{old}})$$ and remapped to $$\mathbf{p}_{\mathrm{new}} = (p^{(t)}_{\mathrm{new}}, p^{(h)}_{\mathrm{new}}, p^{(w)}_{\mathrm{new}}),$$ the corrected key is obtained as $$\begin{equation}
\mathbf{k}_{\mathrm{new}}
=
\mathbf{k}_{\mathrm{old}}
\odot
\mathrm{RotaryDelta}\!\left(\mathbf{p}_{\mathrm{old}}, \mathbf{p}_{\mathrm{new}}\right),
\end{equation}$$ with the relative phase shift: $$\begin{equation}
\mathrm{RotaryDelta}\!\left(\mathbf{p}_{\mathrm{old}}, \mathbf{p}_{\mathrm{new}}\right)
=
\operatorname*{Concat}_{d \in \{t, h, w\}}
\left(
e^{i
(
p^{(d)}_{\mathrm{new}} - p^{(d)}_{\mathrm{old}}
)
\boldsymbol{\theta}^{(d)}
}\right),
\end{equation}$$ where $\operatorname*{Concat}$ denotes the concatenation operation along the channel dimension, and $\boldsymbol{\theta}^{(d)}$ represents the rotary frequency vector corresponding to the channel section allocated for dimension $d$.

# Algorithm of Summary Tokens

:::: algorithm
[]{#alg:summary label="alg:summary"}

::: algorithmic
$K_{p}, V_{p}$: Pruned KV tensors from visual tokens; $P_{p}$: Original position indices of pruned tokens; $t$: Target position index for the summary token.\
$k_{sum}, v_{sum}$: Single aggregated summary token cache.\

**Step 1: Aggregate Value**\
\# Simple spatial mean\
$v_{sum} \leftarrow \text{Mean}(V_{p})$

**Step 2: Aggregate Key**\
\# Phase alignment before pooling\
$\Delta\theta \leftarrow \text{RotaryDelta}(P_{p} \to t)$\
\# Calculate rotation shift from $P_p$ to $t$\
$K_{aligned} \leftarrow \text{ApplyDelta}(K_{p}, \Delta\theta)$\
\# Align all keys to the same phase\
$k_{sum} \leftarrow \text{Mean}(K_{aligned})$

**Step 3: Update KV Cache**\
$K_{new} \leftarrow \text{Concat}([K_{kept}, k_{sum}])$ $V_{new} \leftarrow \text{Concat}([V_{kept}, v_{sum}])$

$K_{new}, V_{new}$
:::
::::

# Full Performances {#sec:full_performance}

## StreamingBench

:::: table*
::: adjustbox
max width=

+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **Model**                                                      | **#Frames** | **OP**    | **CR**    | **CS**    | **ATP**   | **EU**    | **TR**    | **PR**    | **SU**    | **ACP**   | **CT**    | **Avg.**  |
+:===============================================================+:===========:+:=========:+:=========:+:=========:+:=========:+:=========:+:=========:+:=========:+:=========:+:=========:+:=========:+:=========:+
| Human                                                          | \-          | 89.47     | 92.00     | 93.60     | 91.47     | 95.65     | 92.52     | 88.00     | 88.75     | 89.74     | 91.30     | 91.46     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **Proprietary MLLMs**                                                                                                                                                                                            |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| Gemini 1.5 pro [@gemini25]                                     | 1 fps       | 79.02     | 80.47     | 83.54     | 79.67     | 80.00     | 84.74     | 77.78     | 64.23     | 71.95     | 48.70     | 75.69     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| GPT-4o [@openai2024gpt4ocard]                                  | 64          | 77.11     | 80.47     | 83.91     | 76.47     | 70.19     | 83.80     | 66.67     | 62.19     | 69.12     | 49.22     | 73.28     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| Claude 3.5 Sonnet [@claude3_5]                                 | 20          | 73.33     | 80.47     | 84.09     | 82.02     | 75.39     | 79.53     | 61.11     | 61.79     | 69.32     | 43.09     | 72.44     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **Open-source Offline MLLMs**                                                                                                                                                                                    |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| Video-LLaMA2-7B [@videollama2]                                 | 32          | 55.86     | 55.47     | 57.41     | 58.17     | 52.80     | 43.61     | 39.81     | 42.68     | 45.61     | 35.23     | 49.52     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| VILA-1.5-8B [@vila]                                            | 14          | 53.68     | 49.22     | 70.98     | 56.86     | 53.42     | 53.89     | 54.63     | 48.78     | 50.14     | 17.62     | 52.32     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| Video-CCAM-14B [@videoccam]                                    | 96          | 56.40     | 57.81     | 65.30     | 62.75     | 64.60     | 51.40     | 42.59     | 47.97     | 49.58     | 31.61     | 53.96     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| LongVA-7B [@longva]                                            | 128         | 70.03     | 63.28     | 61.20     | 70.92     | 62.73     | 59.50     | 61.11     | 53.66     | 54.67     | 34.72     | 59.96     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| InternVL-V2-8B [@internvl2]                                    | 16          | 68.12     | 60.94     | 69.40     | 77.12     | 67.70     | 62.93     | 59.26     | 53.25     | 54.96     | 56.48     | 63.72     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| Kangaroo-7B [@kangaroo]                                        | 64          | 71.12     | 84.38     | 70.66     | 73.20     | 67.08     | 61.68     | 56.48     | 55.69     | 62.04     | 38.86     | 64.60     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| LLaVA-NeXT-Video-32B [@llava-next]                             | 64          | 78.20     | 70.31     | 73.82     | 76.80     | 63.35     | 69.78     | 57.41     | 56.10     | 64.31     | 38.86     | 66.96     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| MiniCPM-V-2.6-8B [@minicpm]                                    | 32          | 71.93     | 71.09     | 77.92     | 75.82     | 64.60     | 65.73     | 70.37     | 56.10     | 62.32     | 53.37     | 67.44     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
|                                                                |             |           |           |           |           |           |           |           |           |           |           |           |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| Flash-VStream-7B [@flashvstream]                               | \-          | 25.89     | 43.57     | 24.91     | 23.87     | 27.33     | 13.08     | 18.52     | 25.20     | 23.87     | 48.70     | 23.23     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| VideoLLM-online-8B [@videollmonline]                           | 2 fps       | 39.07     | 40.06     | 34.49     | 31.05     | 45.96     | 32.40     | 31.48     | 34.16     | 42.49     | 27.89     | 35.99     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| Dispider-7B [@dispider]                                        | 1 fps       | 74.92     | 75.53     | 74.10     | 73.08     | 74.44     | 59.92     | 76.14     | 62.91     | 62.16     | 45.80     | 67.63     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| TimeChat-Online-7B [@timechatonline]                           | 1 fps       | 80.22     | 82.03     | 79.50     | 83.33     | 76.10     | 78.50     | 78.70     | 64.63     | 69.60     | 57.98     | 75.36     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| StreamForest-7B [@zeng2025streamforestefficientonlinevideo]    | 1 fps       | 83.11     | 82.81     | 82.65     | 84.26     | 77.50     | 78.19     | 76.85     | 69.11     | 75.64     | 54.40     | 77.26     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **Training-free Offline-to-Online Methods**                                                                                                                                                                      |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| LLaVA-OV-7B [@li2024llavaonevisioneasyvisualtask]              | 32          | 78.75     | 78.12     | 80.76     | **81.19** | 71.70     | 72.59     | 72.22     | 63.82     | 66.01     | 38.34     | 71.34     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| \+ ReKV [@di2025streamingvideoquestionansweringincontext]      | 0.5 fps     | 76.02     | 81.25     | 77.92     | 76.90     | 66.04     | 66.04     | 69.44     | 60.98     | 64.31     | 49.22     | 69.22     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| \+ LiveVLM [@ning2025livevlmefficientonlinevideo]              | 0.5 fps     | **81.47** | 78.13     | 83.28     | 79.08     | 69.57     | **74.14** | **75.00** | **69.11** | 67.71     | 40.41     | 72.92     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| \+ StreamKV [@chen2025streamkvstreamingvideoquestionanswering] | 0.5 fps     | 73.80     | 77.30     | 85.90     | 77.50     | **73.30** | 63.90     | 69.40     | 61.40     | 63.20     | 35.80     | 68.80     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| \+ HERMES (6K tokens)                                          | 0.5 fps     | 77.93     | **82.03** | 86.12     | **81.19** | 66.04     | 73.52     | 74.07     | 63.01     | 67.71     | **45.08** | 72.63     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| \+ HERMES (4K tokens)                                          | 0.5 fps     | 79.02     | 81.25     | **87.70** | 80.20     | 69.18     | 71.96     | 73.15     | 66.26     | **69.41** | 43.52     | **73.23** |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| LLaVA-OV-0.5B [@li2024llavaonevisioneasyvisualtask]            | 32          | 71.39     | 57.81     | 65.93     | 69.64     | 69.18     | 55.76     | 57.41     | **52.85** | 62.04     | 16.58     | 59.64     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| \+ ReKV [@di2025streamingvideoquestionansweringincontext]      | 0.5 fps     | 65.12     | 60.16     | 66.56     | 66.01     | 66.67     | 52.96     | 57.41     | 48.37     | 60.34     | 18.13     | 57.39     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| \+ HERMES (6K tokens)                                          | 0.5 fps     | 71.93     | 60.16     | 69.09     | 71.29     | 68.55     | 57.32     | **60.19** | 51.22     | **63.74** | **19,69** | 61.04     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| \+ HERMES (4K tokens)                                          | 0.5 fps     | **72.21** | **61.72** | **70.98** | **72.94** | **72.33** | **57.94** | **60.19** | **52.85** | **63.74** | 19.17     | **62.04** |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| Qwen2.5-VL-7B [@bai2025qwen25vltechnicalreport]                | 1 fps       | 77.93     | 76.56     | 78.55     | 80.86     | 76.73     | 76.95     | 80.56     | 65.45     | 65.72     | **52.85** | 73.31     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| \+ HERMES (6K tokens)                                          | 0.5 fps     | 83.38     | 78.91     | 86.12     | 87.13     | **78.62** | **86.60** | **84.26** | 74.80     | 71.39     | 46.63     | 78.72     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| \+ HERMES (4K tokens)                                          | 0.5 fps     | **83.65** | **81.25** | **88.01** | **87.46** | 76.73     | **86.60** | 82.41     | **76.02** | **73.94** | 46.63     | **79.44** |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| Qwen2.5-VL-32B [@bai2025qwen25vltechnicalreport]               | 1 fps       | 76.29     | 79.69     | 78.55     | **83.50** | 76.10     | 79.44     | 80.56     | 61.38     | 68.27     | **59.07** | 74.27     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| \+ HERMES (6K tokens)                                          | 0.5 fps     | **84.47** | 79.69     | **87.70** | 83.17     | **81.76** | **88.16** | 86.11     | 74.80     | **77.62** | 49.22     | **80.20** |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| \+ HERMES (4K tokens)                                          | 0.5 fps     | 83.92     | **80.47** | **87.70** | **83.50** | 80.50     | **88.16** | **87.04** | **75.20** | 77.34     | 48.19     | 80.08     |
+----------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
:::
::::

## OVO-Bench

:::: table*
::: adjustbox
max width=

+:-------------------------------------------------------------+:-----------:+:---------:+:---------:+:---------:+:---------:+:---------:+:---------:+:---------:+:---------:+:---------:+:---------:+:---------:+:-----------:+
| **Model**                                                    | **#Frames** | **Real-Time Visual Perception**                                                   | **Backward Tracing**                          | **Overall** |
|                                                              |             +-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+             |
|                                                              |             | OCR       | ACR       | ATR       | STU       | FPD       | OJR       | Avg.      | EPM       | ASI       | HLD       | Avg.      |             |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| Human                                                        | \-          | 93.96     | 92.57     | 94.83     | 92.70     | 91.09     | 94.02     | 93.20     | 92.59     | 93.02     | 91.37     | 92.33     | 92.77       |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| **Proprietary MLLMs**                                                                                                                                                                                                        |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| Gemini 1.5 Pro [@gemini25]                                   | 1fps        | 85.91     | 66.97     | 79.31     | 58.43     | 63.37     | 61.96     | 69.32     | 58.59     | 76.35     | 52.64     | 62.54     | 65.93       |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| GPT-4o [@openai2024gpt4ocard]                                | 64          | 69.80     | 64.22     | 71.55     | 51.12     | 70.30     | 59.78     | 64.46     | 57.91     | 75.68     | 48.66     | 60.75     | 62.61       |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| **Open-source Offline MLLMs**                                                                                                                                                                                                |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| LLaVA-Video-7B [@zhang2025llavavideovideoinstructiontuning]  | 64          | 69.80     | 59.63     | 66.38     | 50.56     | 72.28     | 61.41     | 63.34     | 51.18     | 64.19     | 9.68      | 41.68     | 52.51       |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| Qwen2-VL-7B [@wang2024qwen2vlenhancingvisionlanguagemodels]  | 64          | 69.13     | 53.21     | 63.79     | 50.56     | 66.34     | 60.87     | 60.65     | 44.44     | 66.89     | 34.41     | 48.58     | 54.62       |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| InternVL2-8B [@internvl2]                                    | 64          | 68.46     | 58.72     | 68.97     | 44.94     | 67.33     | 55.98     | 60.73     | 43.10     | 61.49     | 27.41     | 44.00     | 52.37       |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| LongVU-7B [@shen2024longvuspatiotemporaladaptivecompression] | 1fps        | 55.70     | 49.54     | 59.48     | 48.31     | 68.32     | 63.04     | 57.40     | 43.10     | 66.22     | 9.14      | 39.49     | 48.45       |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| **Open-source Online MLLMs**                                                                                                                                                                                                 |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| VideoLLM-online-8B [@videollmonline]                         | 2fps        | 8.05      | 23.85     | 12.07     | 14.04     | 45.54     | 21.20     | 20.79     | 22.22     | 18.80     | 12.18     | 17.73     | 19.26       |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| Flash-VStream-7B [@flashvstream]                             | 1fps        | 25.50     | 32.11     | 29.31     | 33.71     | 29.70     | 28.80     | 29.86     | 36.36     | 33.78     | 5.91      | 25.35     | 27.61       |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| Dispider-7B [@dispider]                                      | 1fps        | 57.72     | 49.54     | 62.07     | 44.94     | 61.39     | 51.63     | 54.55     | 48.48     | 55.41     | 4.30      | 36.06     | 45.31       |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| TimeChat-Online-7B [@timechatonline]                         | 1fps        | 75.20     | 46.80     | 70.70     | 47.80     | 69.30     | 61.40     | 61.90     | 55.90     | 59.50     | 9.70      | 41.70     | 51.80       |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| StreamForest-7B [@zeng2025streamforestefficientonlinevideo]  | 1fps        | 68.46     | 53.21     | 71.55     | 47.75     | 65.35     | 60.87     | 61.20     | 58.92     | 64.86     | 32.26     | 52.02     | 56.61       |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| **Training-free Offline-to-Online Methods**                                                                                                                                                                                  |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| LLaVA-OV-7B [@li2024llavaonevisioneasyvisualtask]            | 32          | 67.79     | 55.05     | 72.41     | 48.31     | 72.28     | 62.50     | 63.06     | 57.24     | 55.41     | 18.28     | 43.64     | 53.35       |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| \+ ReKV [@di2025streamingvideoquestionansweringincontext]    | 0.5 fps     | 52.35     | 54.13     | 69.83     | 43.26     | 67.33     | 57.07     | 57.33     | 57.58     | 56.08     | 18.82     | 44.16     | 50.75       |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| \+ HERMES (6K tokens)                                        | 0.5 fps     | **72.48** | **62.39** | 69.83     | 47.75     | **73.27** | 64.67     | 65.07     | **61.28** | 58.78     | 26.34     | 48.80     | 56.94       |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| \+ HERMES (4K tokens)                                        | 0.5 fps     | **72.48** | **62.39** | **74.14** | **50.56** | **73.27** | **65.22** | **66.34** | 60.61     | **61.49** | **28.49** | **50.20** | **58.27**   |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| LLaVA-OV-0.5B [@li2024llavaonevisioneasyvisualtask]          | 32          | 53.69     | 53.21     | 48.28     | **33.71** | **60.40** | **48.91** | 49.70     | 46.13     | 45.27     | **12.37** | 34.59     | 42.15       |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| \+ ReKV [@di2025streamingvideoquestionansweringincontext]    | 0.5 fps     | 41.61     | 44.95     | 50.00     | 29.78     | **60.40** | 35.87     | 43.77     | 46.13     | 43.92     | 9.14      | 33.06     | 38.42       |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| \+ HERMES (6K tokens)                                        | 0.5 fps     | **57.05** | **49.54** | 55.17     | 32.58     | **60.40** | 47.28     | 50.34     | **47.81** | 47.30     | 9.14      | 34.75     | 42.55       |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| \+ HERMES (4K tokens)                                        | 0.5 fps     | 56.38     | 47.71     | **56.90** | 32.02     | 62.38     | **48.91** | **50.72** | **47.81** | **47.97** | 8.60      | **34.80** | **42.76**   |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| Qwen2.5-VL-7B [@bai2025qwen25vltechnicalreport]              | 1fps        | 67.79     | 55.05     | 67.24     | 42.13     | 66.34     | 60.87     | 59.90     | **51.52** | 58.78     | 23.66     | 44.65     | 52.28       |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| \+ HERMES (6K tokens)                                        | 0.5 fps     | **85.91** | 60.55     | **74.14** | 52.81     | 70.30     | **66.85** | 68.42     | 49.49     | 61.49     | 33.33     | 48.10     | 58.26       |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| \+ HERMES (4K tokens)                                        | 0.5 fps     | 85.23     | **64.22** | 71.55     | **53.37** | **74.26** | 65.22     | **68.98** | 48.48     | **62.16** | **37.63** | **49.43** | **59.21**   |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| Qwen2.5-VL-32B [@bai2025qwen25vltechnicalreport]             | 1fps        | 77.18     | 58.72     | 68.10     | 50.56     | **74.26** | 57.61     | 64.40     | **58.59** | 62.84     | 29.57     | 50.33     | 57.37       |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| \+ HERMES (6K tokens)                                        | 0.5 fps     | 87.25     | **66.06** | **74.14** | 57.30     | 71.29     | 75.54     | 71.93     | 55.56     | **70.27** | 47.31     | **57.71** | **64.82**   |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
| \+ HERMES (4K tokens)                                        | 0.5 fps     | **88.59** | 65.14     | **74.14** | **58.99** | 71.29     | **76.09** | **72.37** | 52.19     | 66.22     | **47.85** | 55.42     | 63.90       |
+--------------------------------------------------------------+-------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------------+
:::
::::

# Case Study {#app:more_case}

We provide six representative case study examples from RVS-Ego and RVS-Movie to demonstrate the advantages of [***HERMES***]{style="color: HermesBlue"} compared to the foundation model LLaVA-OV-7B. During the understanding of streaming long videos, [***HERMES***]{style="color: HermesBlue"} exhibits significantly finer-grained temporal (shown in [15](#fig:case_temporal){reference-type="ref+label" reference="fig:case_temporal"}) and spatial understanding [16](#fig:case_spatial){reference-type="ref+label" reference="fig:case_spatial"} capabilities than its corresponding foundation model.

<figure id="fig:case_temporal" data-latex-placement="p">
<embed src="figures/case_study_temporal.pdf" />
<figcaption>Cases demonstrating the superior fine-grained temporal understanding capability of <span style="color: HermesBlue"><em><strong>HERMES</strong></em></span> relative to the LLaVA-OV-7B base model.</figcaption>
</figure>

<figure id="fig:case_spatial" data-latex-placement="p">
<embed src="figures/case_study_spatial.pdf" />
<figcaption>Cases demonstrating the superior fine-grained spatial understanding capability of <span style="color: HermesBlue"><em><strong>HERMES</strong></em></span> relative to the LLaVA-OV-7B base model.</figcaption>
</figure>

[^1]: To ensure the sliding window contains 6,000 tokens, a video at 0.5 fps for LLaVA-OV must have a duration of at least $6,000 / 196 / 0.5\approx 62s$.
