maketitle thanks aketitle

# Introduction {#sec:introduction}

In the literature, video understanding tasks, such as action recognition [@activitynet; @something; @kinetics], visual object tracking [@got_10k; @trackingnet], and video question-answering [@msrvtt_qa; @jang2017tgifqa; @next_qa; @li2024mvbench], have primarily focused on short clips lasting from a few seconds to minutes. However, as vision models increasingly find applications in real-world scenarios like robotics, surveillance, and live broadcasts, the research in the vision community has gradually shifted towards understanding continuous video streams, where long-term contexts and real-time interaction are crucial.

In this paper, we consider the problem of **streaming video question-answering (StreamingVQA)**. As shown in Figure [1](#fig:teaser){reference-type="ref" reference="fig:teaser"}(a), it involves continuously processing long video streams and promptly responding to queries about the visual content at any moment. It can be treated as a generalization of the standard offline VideoQA, where the model processes the entire video and all questions simultaneously. By definition, such task of StreamingVQA presents three core challenges: (i) **Efficient Video Encoding:** Unlike traditional offline VideoQA, where models have access to the entire video clip, StreamingVQA demands real-time analysis of continuous streams. Models must efficiently process incoming frames without access to future frames or frequent revisiting of distant past frames. (ii) **Video Context Preservation:** To accurately answer questions posed later in the stream, models must preserve relevant information from earlier frames, making long-term context retention a key challenge. (iii) **Real-Time Response:** The model must provide accurate answers with minimal delay, requiring efficient retrieval of video context and rapid question-answering.

Current Video-LLMs often struggle to encode long video streams due to the large volume of video tokens, forcing most models to process only a sparse subset of frames [@video_chatgpt; @llava_next_video; @llava_onevision]. This results in limited video lengths or a significant loss of fine-grained visual information. While techniques like average pooling [@llamavid] and memory compression [@wu2022memvit; @wang2023memory; @he2024ma; @flashvstream; @videostreaming] reduce token volume, they come at the cost of losing details, particularly in temporal and lower-level visual features that are essential for complex question answering.

<figure id="fig:teaser" data-latex-placement="t">
<embed src="figures/src/teaser.pdf" />
<figcaption> <strong>Overview of the StreamingVQA task and our proposed ReKV.</strong> (a) StreamingVQA requires a model to continuously process video streams and answer questions about previously viewed content at any moment. (b) We propose ReKV to enhance efficiency and accuracy in StreamingVQA. Tested with <code>LLaVA-OV-7B</code> on an H800 (80GB) GPU, ReKV maintains stable latency and GPU memory usage, preventing out-of-memory (OOM) errors as frames increase. It also improves the accuracy on seven long-form VideoQA benchmarks compared to the uniform sampling baseline. Further details are provided in Section <a href="#sec:experiments" data-reference-type="ref" data-reference="sec:experiments">4</a>. </figcaption>
</figure>

To address the challenges, we propose **ReKV** (**Re**trieve In-context Video **KV**-Cache), a framework that seamlessly integrates with existing Video-LLMs [@video_chatgpt; @llava_next_video; @llava_onevision] without additional training. Our method employs two strategies for aggregating both short- and long-term temporal information. For **short-term temporal context**, the model adopts causal attention with a sliding-window mechanism [@lm_infinite], where tokens attend only to a limited set of preceding tokens during encoding. For **recalling long-term information**, we enable dynamic access to any point within the video sequence via retrieval. Specifically, our method retains and reuses past computations (KV-Cache) to avoid redundant processing while enhancing long-term reasoning without sacrificing detail. For extremely long videos, KV-Caches can be offloaded to RAM or disk to prevent memory overflow.

To ensure real-time and accurate responses, we retrieve a fixed number of KV-Caches relevant to the current question. This design strikes a balance between efficiency and accuracy by avoiding the need to process all past frames, while still accessing the most critical information. We experimented with two retrieval methods: one using external CLIP-like models [@clip; @siglip] for semantic matching, and another leveraging internal attention weights for faster, more integrated, and potentially stronger retrieval [@infllm; @snapkv].

In summary, ReKV efficiently encodes long video streams, preserves and retrieves in-context KV-Caches to address complex video question-answering. In addition, ReKV separates video encoding from question-answering into distinct processes, further enhancing efficiency. As shown in Figure [1](#fig:teaser){reference-type="ref" reference="fig:teaser"}(b), ReKV improves VideoQA accuracy while maintaining stable inference latency and memory usage as frames increase. The remainder of the paper is organized as follows: Section [5](#sec:related_work){reference-type="ref" reference="sec:related_work"} provides an overview of the relevant literature. Section [3](#sec:method){reference-type="ref" reference="sec:method"} formulates the StreamingVQA task and describes our proposed method in detail. In Section [4](#sec:experiments){reference-type="ref" reference="sec:experiments"}, we present ablation studies and comparisons to validate our approach. Consequently, our approach not only enhances accuracy on long VideoQA benchmarks, including MLVU [@mlvu], [QAEgo4D$_\texttt{MC}$]{.smallcaps} [@di2023groundvqa], EgoSchema [@egoschema], and ActivityNet-QA [@activitynet_qa], as well as StreamingVQA benchmarks [@flashvstream], but also reduces inference latency and memory usage.

# StreamingVQA: Task Definition and Discussion {#sec:task}

This paper considers the problem of streaming video question-answering (**StreamingVQA**), where a model continuously processes a video stream and can respond to questions about past visual content at any moment. In this section, we formally define the task and outline the design principles for our proposed solution.

Given a video stream $\mathcal{V}^T := [v_1, v_2, ..., v_T]$ consisting of $T$ frames and a set of $N$ questions $\mathcal{Q} := \{q_1, q_2, \dots, q_N\}$, StreamingVQA aims to answer a question $q_i$ at any time step $t~(1 \le t \le T)$, using only the frames seen up to that point, $\mathcal{V}^t := [v_1, v_2, ..., v_t]$.

**Discussion-I: StreamingVQA *vs.* OfflineVQA.** StreamingVQA involves continuously analyzing an incoming video stream and answering questions based on the observed visual content at any moment. In contrast, conventional video question-answering models [@frozen_bilm; @video_chatgpt; @llava_next_video; @llava_onevision] operate in an offline mode, referred to as OfflineVQA. The two paradigms differ in that: 1) StreamingVQA processes a continuous video stream, while OfflineVQA handles a predefined video input, and 2) StreamingVQA allows questions to be asked at any point during the stream, whereas OfflineVQA processes questions only after the entire video has been viewed. Notably, OfflineVQA can be considered a special case of StreamingVQA, where all questions are posed after the video is fully processed.

Conventional approaches typically employ a visual encoder [@clip; @siglip; @fang2023eva] and a projection module [@llava_next_video; @blip2] to process video frames ($\mathcal{V}^t$). The output is concatenated with the tokenized question to form a sequence $[\mathcal{V}_t, q_i]$ [^1], which is then passed to an LLMs to predict an answer. However, this approach is impractical due to the high computational cost associated with processing a large number of frames ($T$).

A common workaround is sparse frame sampling [@video_chatgpt; @llava_next_video; @llava_onevision], but this introduces new problems: (i) loss of critical visual information, leading to incomplete or inaccurate responses, and (ii) the need to reprocess frames for different questions, since questions asked at different time points require distinct frame samples. This becomes increasingly inefficient as $T$ and $N$ grow.

Given these challenges, current OfflineVQA methods fall short when applied to StreamingVQA scenarios. Therefore, designing a new approach optimized for StreamingVQA is crucial to handling video streams more efficiently, enabling real-time question answering and unlocking more interactive video analysis applications.

**Discussion-II: Design Principles for Efficient StreamingVQA.** To tackle the aforementioned challenges, we can exploit the causal nature of the LLM decoder to avoid redundant computations and strike a balance between accuracy and speed. During attention calculations, causal masking prevents the model from accessing future tokens, ensuring that video tokens are encoded independently of the questions. This allows us to *decouple video encoding from question-answering*.

For video encoding, we leverage the KV-Cache optimization to accelerate inference. However, as number of frames grows large, handling the massive number of video tokens becomes increasingly inefficient and may exceed the model's capacity [@lm_infinite; @streamingllm]. To address this, we adopt a sliding-window attention mechanism [@lm_infinite], which limits the attention scope to only the most recent frames.

Regarding question-answering, Video KV-Caches are stored and can be reused as context to answer different questions. However, long video sequences produce a substantial amount of KV-Caches, leading to excessive GPU memory consumption, computational overhead, and unnecessary distractions if all are used. To address this, we introduce an efficient retrieval method that selects the most relevant video key-value vectors from the video KV-Caches. These selected vectors then serve as context, enabling efficient and scalable StreamingVQA.

#  ReKV: [Re]{.underline}trieve In-context Video [KV]{.underline}-Cache  {#sec:method}

This section introduces **ReKV**, an approach that integrates seamlessly with a Video-LLM to enable efficient StreamingVQA without requiring additional training. Overall, ReKV efficiently encodes the video stream, maintains its KV-Caches, retrieves relevant caches based on the given question, and uses them for accurate question-answering.

<figure id="fig:framework" data-latex-placement="t">
<embed src="figures/src/framework.pdf" />
<figcaption> <strong>Overview of ReKV.</strong> We modify the attention mechanism in Decoder-based Video-LLMs: (a) The video stream is encoded with sliding-window attention (Equation <a href="#equ:video_encoding" data-reference-type="ref" data-reference="equ:video_encoding">[equ:video_encoding]</a>), with out-of-window Video KV-Caches offloaded to RAM or disk. (b) Upon receiving a question, relevant key-value vectors are retrieved based on cosine similarity, with compressed vectors to accelerate retrieval (Equation <a href="#equ:retrieval" data-reference-type="ref" data-reference="equ:retrieval">[equ:retrieval]</a>). (c) The retrieved key-value vectors are reloaded onto the GPU and utilized for autoregressive answer generation (Equation <a href="#equ:answer_generation" data-reference-type="ref" data-reference="equ:answer_generation">[equ:answer_generation]</a>). </figcaption>
</figure>

We aim to enable Video-LLMs, trained on limited frames, to perform StreamingVQA **without additional training**. As shown in Figure [2](#fig:framework){reference-type="ref" reference="fig:framework"}, the proposed ReKV has three components: video stream encoding, video KV-Cache retrieval, and question-answering using the retrieved key-value vectors.

**Video Stream Encoding with Sliding-window Attention.** We encode the video stream $\mathcal{V}^T$ incrementally, processing it chunk by chunk. At each step, the inputs include past key-value vectors $\mathbf{P} = \{(\mathbf{k}_j, \mathbf{v}_j)\}_{j=1}^{l_P}$ and the current tokens $\mathbf{X}=\{\mathbf{t}_{i+l_P}\}_{i=1}^{l_X}$, where $l_P$ denotes the lengths of past key-values, and $l_X$ refers to the chunk size. The local key-value vectors within a window $l_L$ can thus be derived as $\mathbf{L} = \mathbf{P}_{[l_P - l_L + 1 : l_P]}$. The attention calculation is then formulated as: $$\begin{equation}
    \mathbf{O} = \text{Attn}\left(\mathbf{W_Q}\mathbf{X}, [\mathbf{L}_k, \mathbf{W_K}\mathbf{X}], [\mathbf{L}_v, \mathbf{W_V}\mathbf{X}]\right),
    \label{equ:video_encoding}
\end{equation}$$ where $\mathbf{W_Q}$, $\mathbf{W_K}$, and $\mathbf{W_V}$ are the attention layer parameters, $\mathbf{L}_k$ and $\mathbf{L}_v$ correspond to the key and value vectors in $\mathbf{L}$. All video KV-Caches are stored for future retrieval. For extremely long videos, we manage memory constraints by offloading KV-Caches to RAM or disk, as in [@infllm].

**External Video KV-Cache Retrieval.** Here, we utilize an external CLIP-like model [@clip; @siglip] to retrieve question-relevant video KV-Cache, primarily as a baseline to assess whether retrieval can enhance VideoQA performance, as demonstrated in Section [4](#sec:experiments){reference-type="ref" reference="sec:experiments"}. Specifically, a CLIP-like model transformers each video frame into a vector $\mathbf{v} = f_v(v) \in \mathrm{R}^D$, where $f_v$ represents the visual encoder, $D$ denotes the vector dimension. Similarly, the question is encoded as $\mathbf{q} = f_t(q) \in \mathrm{R}^D$, where $f_t$ is the text encoder. We then compute the cosine similarity between the embeddings of frame and question: $$\begin{equation}
    \text{Sim}(\mathbf{v}, \mathbf{q}) = \frac{\mathbf{v} \cdot \mathbf{q}}{\tau~||\mathbf{v}||~||\mathbf{q}||}
    \label{equ:retrieval}
\end{equation}$$ where $\tau$ is a learnable temperature parameter. This similarity is calculated at the frame level, rather than at the token level. Alternatively, we can group $b$ consecutive frames into blocks by averaging their frame vectors and then compute block-level similarity scores. Finally, the $r$ most relevant video frames or $\lceil r/b \rceil$ video blocks are retrieved. The corresponding video KV-Cache, denoted as $\mathbf{R}$, is subsequently loaded onto the GPU for question-answering.

**Internal Video KV-Cache Retrieval.** Building on recent advancements in handling long sequences with LLMs [@infllm; @quickllama; @em_llm], we further explore using self-attention layers within Video-LLMs for retrieval. Similar to external retrieval, internal retrieval is still performed at the level of video frames or blocks.

During video modeling, the average of the key vectors of a frame is computed as its representative frame vector: $\mathbf{v} = \frac{1}{N_f} \sum_{j=1}^{N_f} \mathbf{k}_j \in \mathrm{R}^{D'}$, where $N_f$ is the number of tokens per frame, and $\mathbf{k}_j$ is the $j$-th key vector. To reduce computational overhead, we do not differentiate between attention heads and instead concatenate them into a single vector, with $D'$ as the resultant dimension. Similarly, the question vector is computed as $\mathbf{q} = \frac{1}{N_q} \sum_{k=1}^{N_q} \mathbf{q}_{k} \in \mathrm{R}^{D'}$, where $N_q$ is the number of tokens in the question, and $\mathbf{q}_{k}$ is its $k$-th query vector. The similarity computation and video KV-Cache retrieval are identical to that of external retrieval, except that $\tau$ is set to 1.

Note that, internal retrieval offers several advantages over external retrieval. First, it operates independently within each self-attention layer, allowing different layers to retrieve different video blocks.[^2] This allows for a broader capture of video context. Additionally, internal retrieval reuses already computed hidden representations and does not introduce extra parameters, which reduces the computational overhead compared to external retrieval.

**Question-Answering Using Retrieved KV.** The retrieved Video KV-Caches serve as the context for video question-answering. Formally, the attention calculation is formulated as: $$\begin{equation}
    \mathbf{O} = \text{Attn}\left(\mathbf{W_Q}\mathbf{X}, [\mathbf{R}_k, \mathbf{W_K}\mathbf{X}], [\mathbf{R}_v, \mathbf{W_V}\mathbf{X}]\right),
    \label{equ:answer_generation}
\end{equation}$$ where $\mathbf{X}$ represents either the question tokens or the current token being decoded, and $\mathbf{R}_k$ and $\mathbf{R}_v$ are the key and value vectors from the context, which includes the retrieved video, question, and previously generated tokens.

**Positional Encoding.** Our baseline Video-LLMs employ Rotary Position Embeddings (RoPE) [@su2024roformer], a commonly used relative positional encoding method. Our video streaming encoding process follows LM-Infinite [@lm_infinite], where RoPE operates normally within the local window but is constrained by a "distance ceiling" for more distant tokens. For question-answering, we do not account for the original positions of the retrieved KV-Caches, as handling unseen distances among tokens presents significant challenges [@lm_infinite]. Instead, we treat these retrieved tokens as regular consecutive tokens. We also experimented with a static variation from Inf-LLM [@infllm], where all retrieved tokens are assigned the same position. Our results show that applying standard RoPE to retrieved video tokens leads to better performance, likely due to the importance of capturing temporal information in video comprehension.

# Experiments {#sec:experiments}

## Benchmark and Metrics

**MLVU**$_\texttt{dev-mc}$ [@mlvu] is the multiple-choice subset of the MLVU-dev benchmark. It focuses on evaluating the long-form video understanding of MLLMs. The question-answer pairs are manually labeled and can be divided into 3 groups: single-detail, multi-detail, and holistic. The evaluation metric is Accuracy.

**[QaEgo4D]{.smallcaps}**$_\texttt{test-mc}$ [@di2023groundvqa] is the multiple-choice subset of the [QaEgo4D]{.smallcaps}-test benchmark, focusing on question-answering in long egocentric videos. It includes annotations marking video segments relevant to each question. The evaluation metric is Accuracy.

**EgoSchema** [@egoschema] is a diagnostic benchmark for long VideoQA, featuring over 5000 multiple-choice questions and long temporal certificate length. It challenges AI models with long-term understanding, as current state-of-the-art models achieve significantly lower accuracy compared to human performance.

**ActivityNet-QA** [@activitynet_qa] encompasses human-annotated QA pairs on 5,800 videos derived from the ActivityNet [@activitynet] dataset. This benchmark is designed to assess the capabilities of VideoQA models in long-term spatiotemporal reasoning. Our evaluation methodology aligns with that of Video-ChatGPT [@video_chatgpt], employing `GPT-3.5-turbo-0613` to judge the accuracy of the open-ended VideoQA responses.

::: wraptable
r0.54

[]{#tab:benchmark_comparison label="tab:benchmark_comparison"}
:::

**RVS-Ego** and **RVS-Movie** [@flashvstream] are Streaming VideoQA benchmarks, constructed using 10 long videos from the Ego4D dataset [@ego4d] and 22 long videos from the MovieNet dataset [@movienet], respectively. These benchmarks feature open-ended questions paired with timestamps, which are initially generated by GPT-4V [@gpt4v] and GPT-4 [@gpt4], and subsequently refined through manual filtering.

**CGBench**$_\texttt{mc}$ [@cgbench], the multiple-choice subset of CGBench, is designed for clue-grounded question answering in long videos. It focuses on the ability to retrieve relevant clues for questions, making it an ideal testbed for ReKV.

## Implementation Details

We primarily evaluate our approach by integrating it into `LLaVA-OV-0.5B` and `LLaVA-OV-7B` [@llava_onevision], chosen for their simplicity and strong performance. In the Appendix, we conduct experiments with several other Video-LLMs as further validations.

All experiments are conducted on NVIDIA A100 (80GB) GPUs with FP16 precision. For video modeling, we process the video stream at 0.5 FPS, in line with GPT-4o's testing on MLVU [@mlvu]. The local window size is set to 15K. For external video KV-Cache retrieval, we use `SigLIP-SO400M` [@siglip] as the retriever. For internal KV-Cache retrieval, we set the block size ($b$) to 1 and the number of retrieved frames ($r$) to 64 by default, with further hyper-parameter variations explored in Section [4.3](#sec:exp_ablations){reference-type="ref" reference="sec:exp_ablations"}.

Unless otherwise specified, **ReKV** refers to the use of internal video KV-Cache retrieval.

## Ablations {#sec:exp_ablations}

In this section, we conduct ablation studies on the effectiveness of in-context retrieval, number of retrieved frames, and the block size.

::: wraptable
r0.48
:::

**Effectiveness of In-context Retrieval.** The experiments on [QaEgo4D]{.smallcaps}$_{\texttt{test-mc}}$, as presented in Table [\[tab:ablation_qaego4d\]](#tab:ablation_qaego4d){reference-type="ref" reference="tab:ablation_qaego4d"}, demonstrate the effects of various retrieval methods on VideoQA accuracy and recall. The recall metric, defined as the percentage of question-relevant video frames retrieved, exhibits a strong positive correlation with VideoQA performance: higher recall consistently leads to better accuracy. Uniform Sampling, which sparsely selects frames, achieves the lowest recall and, consequently, the poorest VideoQA accuracy. In contrast, Oracle Retrieval, with perfect recall, delivers the highest VideoQA accuracy, significantly outperforming Uniform Sampling. While External and Internal Retrieval fall short of Oracle-level precision, both surpass Uniform Sampling, with Internal Retrieval excelling due to its higher recall.

The MLVU benchmark [@mlvu] encompasses three types of VideoQA tasks: *Single Detail* requires identifying a single critical plot within a long video, *Multi Detail* necessitates the integration of multiple plots, and *Holistic* demands a comprehensive understanding of the entire video. This makes MLVU an ideal platform for evaluating our in-context retrieval method. As shown in Table [\[tab:ablation_mlvu\]](#tab:ablation_mlvu){reference-type="ref" reference="tab:ablation_mlvu"}, both External and Internal Retrieval enhance the overall VideoQA accuracy over the Uniform Sampling baseline. The enhancements are most pronounced in Single Detail tasks, demonstrating that ReKV effectively retrieves question-relevant video context. Furthermore, Internal Retrieval significantly outperforms External Retrieval in Holistic tasks, likely due to its ability to capture broader context and leverage the Video-LLM's video modeling capabilities, as discussed in Section [3](#sec:method){reference-type="ref" reference="sec:method"}.

**Number of Retrieved Frames.** We fix the block size ($b = 1$) and evaluate the impact of varying the numbers of retrieved frames ($r \in \{8, 16, 32, 48, 64, 80\}$) on the [QaEgo4D]{.smallcaps} and MLVU benchmarks. As illustrated in Figure [3](#fig:ablation_retrieval){reference-type="ref" reference="fig:ablation_retrieval"}(a), increasing the number of retrieved frames generally improves VideoQA accuracy, as it implies capturing more relevant visual context. However, on MLVU, this improvement plateaus as more frames are retrieved since the additional irrelevant information hinders the subsequent question-answering process. Additionally, retrieving more frames increases the computational overhead of the question-answering stage, further slowing down inference.

**Retrieval Block Size.** When processing video streams, we group $b$ consecutive frames into blocks for block-level retrieval. For this experiment, we fix the number of retrieved frames at $r = 64$ and evaluate different block sizes ($b \in {1, 2, 4, 8, 16}$). With a fixed $r$, larger block sizes result in fewer, more concentrated retrieved blocks. Figure [3](#fig:ablation_retrieval){reference-type="ref" reference="fig:ablation_retrieval"}(b) shows that increasing block size negatively affects accuracy on MLVU, while performance on [QaEgo4D]{.smallcaps} remains relatively stable. This suggests that MLVU tasks benefit from retrieving more dispersed visual cues, aligning with its design of multi-detail and holistic tasks [@mlvu]. In contrast, [QaEgo4D]{.smallcaps} primarily relies on a single relevant clip per question [@di2023groundvqa].

<figure id="fig:ablation_retrieval" data-latex-placement="t">
<figure>
<embed src="figures/src/retrieve_size.pdf" />
<figcaption aria-hidden="true"></figcaption>
</figure>
<figure>
<embed src="figures/src/chunk_size.pdf" />
<figcaption aria-hidden="true"></figcaption>
</figure>
<figcaption> <strong>Ablation study of retrieval hyperparameters:</strong> (a) number of retrieved frames and (b) number of frames per retrieval block. Experiments are conducted with <code>LLaVA-OV-7B</code>. </figcaption>
</figure>

## Offline Video Question-answering

Streaming video understanding is a relatively under-explored area, with limited StreamingVQA benchmarks available [@flashvstream]. As discussed in Section [2](#sec:task){reference-type="ref" reference="sec:task"}, OfflineVQA can be considered as a special case of StreamingVQA. Thus, we first evaluate our method in the offline setting using four widely adopted long-form VideoQA benchmarks, comparing our results against state-of-the-art VideoQA methods. A summary of these benchmarks can be found in Table [\[tab:benchmark_comparison\]](#tab:benchmark_comparison){reference-type="ref" reference="tab:benchmark_comparison"}.

As shown in Table [\[tab:sota\]](#tab:sota){reference-type="ref" reference="tab:sota"}, our proposed ReKV always enhances the performance of `LLaVA-OV-0.5B` and `LLaVA-OV-7B` without additional training. Notably, `LLaVA-OV-7B`+ReKV outperforms two memory-based StreamingVQA models (VideoStreaming [@videostreaming] and Flash-VStream [@flashvstream]) by a large margin. While the base model already demonstrates strong performance, and we **do not** claim credit for this achievement, our method can integrate seamlessly with Video-LLMs, benefiting from their ongoing advancements.

## Streaming Video Question-answering {#sec:streaming}

We then evaluate our method on the streaming setting using the RVS-Ego and RVS-Movie benchmarks. During video stream modeling, questions are input immediately after their annotated end timestamps and answered based on the preceding video content.

**Question-answering Performance.** Table [\[tab:streamingvqa\]](#tab:streamingvqa){reference-type="ref" reference="tab:streamingvqa"} presents the StreamingVQA performance. Both external and internal retrieval methods significantly outperform the uniform sampling baseline. Additionally, our approach enables `LLaVA-OV-7B` to surpass Flash-VStream [@flashvstream], demonstrating ReKV's effectiveness for the StreamingVQA.

**Running Speed and Memory Usage.** We also examine the running speed and memory usage under controlled conditions. Specifically, a 1-hour, 1080P video from RVS-Ego with 100 scattered questions is used. Each question is padded to 64 tokens, and the generated answers are fixed at 128 tokens in length. The video frames are pre-extracted at 0.5 FPS (1,800 frames in total) and streamed to the Video-LLM frame by frame.

As illustrated in Table [\[tab:streamingvqa\]](#tab:streamingvqa){reference-type="ref" reference="tab:streamingvqa"}, both retrieval methods maintain high video encoding speeds, with `LLaVA-OV-7B` achieving 11 FPS and `LLaVA-OV-0.5B` achieving 17 FPS. Moreover, KV-Cache offloading remains manageable, with `LLaVA-OV-7B` at 18.8GB/h and `LLaVA-OV-0.5B` at 4.0GB/h (see appendix for more details). External retrieval, however, introduces higher latency and GPU memory usage due to additional computations in the external retriever, whereas internal retrieval significantly reduces both. Figure [1](#fig:teaser){reference-type="ref" reference="fig:teaser"} has also demonstrated that latency and GPU memory usage remain stable as more frames are processed. Flash-VStream also shows good efficiency. However, it only maintains a relatively small memory footprint (681 tokens) [@flashvstream], leading to potential information loss when dealing with extremely long videos.

**Qualitative Examples.** Figure [4](#fig:visualization){reference-type="ref" reference="fig:visualization"} presents an example of streaming video question-answering. Our approach continuously processes video streams while responding to questions posed at different timestamps. To improve efficiency, it stores and retrieves relevant video KV-Caches as contextual information for answering these questions.

We provide additional implementation details and experimental results in the Appendix.

<figure id="fig:visualization" data-latex-placement="t">
<embed src="figures/src/visualization.pdf" />
<figcaption> <strong>StreamingVQA qualitative examples.</strong> The example is drawn from the <span class="smallcaps">QaEgo4D</span> benchmark. The video stream is processed frame by frame. <span style="color: 35,166,8"><span class="math inline">●</span></span> and <span style="color: blue"><span class="math inline">●</span></span> mark the timestamps at which questions are posed. <span style="color: 35,166,8"><span class="math inline">▫</span></span> and <span style="color: blue"><span class="math inline">▫</span></span> indicate the relevant video contexts that support answering these questions. </figcaption>
</figure>

# Related Work {#sec:related_work}

**LLMs for Video Understanding.** In recent years, there has been a surge of interest in leveraging Large Language Models (LLMs) for video understanding, leading to the development of several innovative approaches [@video_chatgpt; @llava_next_video; @llava_onevision]. These models typically use a Vision Encoder to extract video features, followed by a mapping step with Linear Projection, MLP, or Q-Former [@blip2]. The mapped features are combined with textual data and fed into large language models (LLMs) to generate a text output. These models have relatively simple architectures, requiring less training data and computational resources, yet they achieve strong performance on short video understanding benchmarks [@msrvtt_qa; @next_qa; @li2024mvbench]. However, they employ sparse sampling or token compression techniques to reduce the number of tokens, which can result in significant information loss when dealing with longer or more content-rich videos. As a result, they are not well-suited for long video understanding or streaming video understanding.

**Long Video Understanding.** A central challenge in long video understanding is effectively compressing the information from lengthy videos. Many approaches use language as a bridge, condensing videos into dense captions [@llovi; @video_recap; @streaming_dvc]. While this achieves good results in some cases, compressing video content into text often leads to the loss of crucial visual details. Besides, as a pioneering approach to streaming video understanding, VideoLLM-Online [@videollm_online] employs a data-centric methodology by interleaving video and text during training. In contrast, our approach is training-free, allowing seamless integration with various existing Video-LLMs to extend their StreamingVQA capabilities. Additionally, VideoLLM-Online retains only a single token per frame to handle long videos, which may result in visual information loss. Our method preserves complete visual information and leverages In-Context KV-Cache Retrieval to enhance efficiency.

Another line of research focuses on compressing long videos into a memory bank [@wu2019long; @wu2022memvit; @wang2023memory]. MC-ViT [@mc_vit] adapts existing pretrained video transformers by fine-tuning them to attend to condensed visual memories. It relates closely to the token-pruning, merging, and memory-based video understanding methods. In comparison, we propose a training-free method specifically tailored to the StreamingVQA task. Incorporating MC-ViT into the StreamingVQA task could be an interesting avenue for future research, and we acknowledge its potential in this domain. This approach has been integrated into Video-LLMs for streaming video understanding, as shown in works like VideoStreaming [@videostreaming] and Flash-VStream [@flashvstream]. These methods dynamically update the memory during video processing and utilize it for downstream tasks. Despite their innovation, a major limitation of these methods is their failure to account for video length and information density, especially when using a fixed memory size. For example, Flash-VStream compresses both 10-second clips and hour-long videos into the same 681 tokens. Furthermore, these methods lack interpretability, making it difficult to determine how much information is being compressed into the memory or whether relevant video information is being accurately retrieved during downstream tasks.

In pursuit of greater interpretability in long video understanding, methods such as GroundVQA [@di2023groundvqa] and GeLM [@chen2024groundedmultihopvideoqalongform] advocate for localizing relevant video clips while responding to user queries. Drawing inspiration from these, this work refrains from excessively condensing video information. By harnessing the causal capabilities of Video-LLMs, it preserves the entire Video KV-Cache, allowing for the retrieval of relevant information when required. This strategy effectively mitigates the substantial loss of video content while improving interpretability.

**Long Context Handling for LLMs.** Handling long text sequences in LLMs has been a major challenge due to high computational and memory costs, leading to training constraints on shorter sequences. Techniques like StreamingLLM [@streamingllm] and LM-Infinite [@lm_infinite] use sliding window attention to process long sequences incrementally, but discard distant tokens, limiting the model's ability to capture long-range dependencies. Recent approaches [@infllm; @quickllama; @em_llm] address this by storing and retrieving previously computed KV-Caches, enabling better recall of distant contexts.

**Retrieval-Augmented Generation.** Retrieval-augmented generation (RAG) combines retrieval mechanisms with generative models to enhance performance across various NLP tasks by incorporating external knowledge [@guu2020retrieval; @lewis2020retrieval; @borgeaud2022improving] and improving performance in vision-language tasks [@xu2024retrieval]. In-context retrieval, recently proposed for handling long inputs [@ram2023context], retrieves information from the input document itself rather than an external knowledge base. In-context KV-Cache retrieval further improves efficiency by pre-encoding long documents, avoiding redundant encodings, and leveraging the LLM's own retrieval capabilities for faster, more effective performance.

# Conclusion {#sec:conclusions}

In conclusion, this paper introduces a training-free approach, ReKV, designed to enhance the efficiency of Video Large Language Models (Video-LLMs) for streaming video question-answering (StreamingVQA). Unlike conventional video question-answering (VideoQA) systems that must process entire videos before answering, ReKV enables rapid, real-time responses. By employing a sliding-window attention mechanism, it ensures that the model only considers a subset of previous frames while encoding the video stream, significantly cutting down on computational demands. To retain key video context, we developed an in-context KV-Cache retrieval method that efficiently stores and reloads key-value vectors that relevant for each query. This targeted retrieval strategy, combined with the ability to perform video modeling and question-answering on separate processes and GPUs, results in a highly efficient streaming VideoQA system. Extensive experiments show that ReKV not only surpasses existing VideoQA models in performance but also enhances their practicality for real-world streaming applications.

**Acknowledgements.** This work is supported by National Key R&D Program of China (No. 2022ZD0161400). We thank Yikun Liu for discussions and conducting experiments on CGBench.

In the appendix, we provide additional implementation details, experiments, and discussions of limitations and future work.

# Additional Implementation Details

## Multi-processing Serving

As discussed in Section [2](#sec:task){reference-type="ref" reference="sec:task"}, our approach enables the separation of video modeling and question-answering across different processes and GPUs, significantly enhancing efficiency in real-world applications. Specifically, we dedicate a primary process for video stream encoding, utilizing sliding-window attention to analyze the video and store the computed cache in RAM. If RAM capacity is exceeded, the data can be offloaded to disk. Additionally, a process pool is maintained, with the number of processes determined by the frequency of queries and available resources. Each process loads the same Video-LLM parameters but operates independently. The video processing continues uninterrupted, without waiting for question-answering tasks to complete. When a query is posed, we log its timestamp to ensure that video information after this point is excluded from the answer. An available process from the pool is then activated to retrieve relevant video key-value vectors using our method, loading them onto its GPU for question-answering. This approach enables efficient StreamingVQA applications, with significant potential in areas such as robotics, surveillance, augmented reality, and live broadcasting.

## Prompt Templates for VideoQA

We use the same prompt template for all multiple-choice VideoQA benchmarks. Text in [`red`]{style="color: red"} indicates variable inputs.

    System:
    You are a helpful assistant.
    User: 
    @<video>@
    Question: @<question>@
    Options:
    (A) @<Option_A>@
    (B) @<Option_B>@
    (C) @<Option_C>@
    (D) @<Option_D>@
    (E) @<Option_E>@
    Answer with the option's letter from the given choices directly.
    Assistant:

The prompt template for open-ended VideoQA is rather simpler:

    System:
    You are a helpful assistant.
    User: 
    @<video>@
    @<question>@
    Assistant:

## KV-Cache Size Calculation

The size of the KV-Cache can be calculated using the following formula, assuming FP16 precision: $$\begin{equation*}
    2 \times L~\text{layers} \times T~\text{frames} \times M~\text{tokens/frame} \times H~\text{heads} \times D~\text{dimension} \times 2~\text{bytes}.
\end{equation*}$$

For `LLaVA-OV-7B` [@llava_onevision], with $L=28$, $M=196$, $H=4$, and $D=128$, processing a 1-hour video at 0.5 FPS ($T=1800$) results in a total KV-Cache size of 18.8 GB.

Similarly, for `LLaVA-OV-0.5B` [@llava_onevision], with $L=24$, $M=196$, $H=2$, and $D=64$, processing a 1-hour video at 0.5 FPS results in a total KV-Cache size of 4.0 GB.

These theoretical calculations are consistent with the experimental results shown in Table [\[tab:streamingvqa\]](#tab:streamingvqa){reference-type="ref" reference="tab:streamingvqa"}.

# Additional Experiments

## Experiments with more Video-LLMs and Benchmark

To further assess the generalizability of our approach, we tested it on additional Video-LLMs: `Video-LLaVA-7B` [@video_llava], `LongVA-7B` [@longva], and `LLaVA-OV-72B` [@llava_onevision]. We used model sharding for `LLaVA-OV-72B`, significantly slowing inference. To mitigate this, we reduced the FPS to 0.1 and the number of retrieved frames to 32, ensuring efficient evaluation. As shown in [\[tab:additional\]](#tab:additional){reference-type="ref+Label" reference="tab:additional"}, ReKV consistently improved performance across various models and benchmarks, highlighting its robustness and adaptability.

## Fair comparisons with Flash-VStream {#sec:fair_comparison}

[\[tab:sota,tab:streamingvqa\]](#tab:sota,tab:streamingvqa){reference-type="ref+Label" reference="tab:sota,tab:streamingvqa"} compared `LLaVA-OneVision+ReKV` with `Flash-VStream`. However, these comparisons may be unfair due to different architecture and training data. Thus, here we conduct **fair** comparisons using the same Video-LLM backbone, including the identical visual encoder (`CLIP-ViT-L/14`), projector (2-layer MLP), LLM (`Vicuna-7B-v1.5`), training data, and train/eval pipelines.

Due to the inaccessibility of WebVid videos[^3] used in Flash-VStream's original training, we use 232K randomly sampled InternVid videos[^4] as a substitute. This ensures comparable experimental settings. We train a baseline Video-LLM model (`Base`) and a Flash-VStream-enhanced version (`Base+Flash`). Similarly, we integrate ReKV into the same baseline (`Base+ReKV`) for fair comparisons. To maintain parity, the baseline uniformly samples 16 frames per video, resized to $224\times224$. Visual features of shape $(T, 16, 16, D)$ are average-pooled to $(T, 8, 8, D)$ before being passed through the MLP projector and into the LLM. Both Flash-VStream and ReKV process video at 0.5 FPS, with ReKV retrieving 16 frames.

As shown in Table [\[tab:flash\]](#tab:flash){reference-type="ref" reference="tab:flash"}, `Base+ReKV` consistently outperforms the base Video-LLM `Base` and surpasses `Base+Flash` in most cases, highlighting its superiority under fair comparative conditions. Additionally, ReKV offers enhanced usability, seamlessly integrating with existing Video-LLMs without requiring extensive retraining.

On the contrary, the reproduced `Base+Flash` does not consistently outperform `Base`. It excels on StreamingVQA (RVS-Movie and RVS-Ego) and MLVU but underperforms on [QAEgo4D]{.smallcaps} and EgoSchema. This discrepancy is likely due to significant visual information loss: the `Base` model processes 1024 visual tokens ($16\times64$), while `Base+Flash` uses only 681 memory tokens.

For additional context, we include results from the original Flash-VStream (`Original Flash`) using checkpoints from its official repository[^5]. Our reproduced `Base+Flash` shows performance deviations, likely due to differences in training data and potential environmental factors.

## Computational Complexity

We ensure **fair** comparisons by using the identical Video-LLM backbone (Sec. [8.2](#sec:fair_comparison){reference-type="ref" reference="sec:fair_comparison"}) under controlled streaming conditions (Sec. [4.5](#sec:streaming){reference-type="ref" reference="sec:streaming"}). Specifically, we measured the FLOPs and MACs of the base Video-LLM, Flash-VStream, and our external and internal retrieval methods. We analyzed **average TFLOPs and TMACs per QA over various question frequencies** in a 1-hour video, leveraging the `calflops` library [@ye2023calflops].

As shown in [\[tab:tflops,tab:tmacs\]](#tab:tflops,tab:tmacs){reference-type="ref+Label" reference="tab:tflops,tab:tmacs"}, ReKV's efficiency improves significantly with increasing QA frequency. The video stream is encoded only once, and computed results are reused across QAs, leading to reduced per-query complexity as QA frequency rises. Flash-VStream outperforms ReKV at low QA frequencies (*e.g.*, 100 QAs). However, ReKV's complexity decreases more rapidly with increased QA frequency, primarily due to Flash-VStream's high memory update overhead. ReKV is thus better suited for high-concurrency scenarios such as live streaming and requires no additional training.

Furthermore, Internal retrieval consistently outperforms external retrieval, reducing average FLOPs by 15.5% and MACs by 15.2%. These results underscore ReKV's ability to balance computational efficiency and effectiveness, particularly in dynamic, high-query environments. This positions ReKV as a practical and scalable solution for streaming video understanding.

# Limitations and Future Work {#sec:limitation}

While ReKV improves the accuracy and efficiency of Video-LLMs in the StreamingVQA task, it still has several limitations that deserves future investigation: *First*, although the KV-Cache offloading to RAM or disk is manageable, as shown in Table [\[tab:streamingvqa\]](#tab:streamingvqa){reference-type="ref" reference="tab:streamingvqa"}, handling extremely long video streams, such as those in surveillance, may lead to an unsustainable increase in cache size. This issue can be mitigated by integrating techniques such as quantization, token pruning, and compression. *Second*, the use of a constant block size for grouping consecutive frames during retrieval can disrupt video continuity. A more refined solution would involve segmenting videos into semantically coherent blocks. *Third*, our method retrieves a fixed number of frames. Future work could explore dynamic retrieval strategies that adjust the number of frames based on video context and query requirements. *Finally*, StreamingVQA remains an under-explored task with few available benchmarks. Developing high-quality benchmarks with precise temporal annotations is crucial for advancing future research.

[^1]: We maintain the original notation for simplicity.

[^2]: For simplicity, we omit the layer index in the above explanation.

[^3]: https://github.com/m-bain/webvid

[^4]: https://huggingface.co/datasets/OpenGVLab/InternVid

[^5]: https://github.com/IVGSZ/Flash-VStream
