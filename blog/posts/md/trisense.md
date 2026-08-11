---
title: "TriSense: 보고, 듣고, 말을 이해하는 비디오 LLM"
dek: "시각·오디오·음성 세 모달리티를 쿼리에 맞게 동적으로 조합하는 모멘트 검색·캡셔닝 모델"
tags: ["Multimodal", "Video", "LLM"]
date: "2026-04-08"
readtime: "20"
slug: "trisense"
katex: false
---

## 논문 한눈에 보기

**TriSense** (NeurIPS 2025)는 비디오 시간 이해의 핵심 공백을 파고든다. 기존 모델들은 시각만 보거나, 세 모달리티를 무조건 합산하거나, 일부 모달리티가 빠지면 무너졌다. TriSense는 **쿼리가 무엇을 묻느냐에 따라** 시각·오디오·음성의 기여도를 동적으로 조절하는 Query-Based Connector를 핵심으로 삼는다. 함께 공개된 **TriSense-2M** 데이터셋은 200만 샘플, 평균 905초 롱 비디오로 이 분야 최대 규모다.

<figure>
<img src="img/trisense/teaser.jpg" alt="Figure 1">
<figcaption><strong>Figure 1</strong> — TriSense는 세 모달리티의 임의 조합(AVS, AV, VS, V 등 8가지)에서 Segment Captioning과 Moment Retrieval을 동시에 수행한다.</figcaption>
</figure>

## Abstract

<p><strong>흐름:</strong> 인간의 멀티모달 이해 능력 → 기존 모델의 오디오 통합 실패 → TriSense 제안(Query-Based Connector) → TriSense-2M 소개 → 실험 결과 예고</p>

### 문제: 오디오를 제대로 못 쓰는 기존 모델
<mark>*"A scientist passionately speaks on wildlife conservation as dramatic orchestral music plays, with the audience nodding and applauding"* — 이 장면을 찾으려면 시각·오디오·음성을 동시에 처리해야 한다. 기존 모델은 이를 못 한다.</mark>

*"Existing models often struggle to effectively fuse and interpret audio information, limiting their capacity for comprehensive video temporal understanding."*

### TriSense + TriSense-2M
<mark>Query-Based Connector로 모달리티 기여도를 쿼리에 맞게 조절. 200만 샘플의 고품질 데이터셋 TriSense-2M으로 학습.</mark>

*"Central to TriSense is a Query-Based Connector that adaptively reweights modality contributions based on the input query, enabling robust performance under modality dropout and allowing flexible combinations of available inputs."*

## Introduction

<p><strong>흐름:</strong> 인간 지각의 멀티모달 통합 → 기존 MLLM의 두 가지 핵심 한계 → TriSense의 세 가지 기여</p>

### 기존 MLLM의 두 가지 핵심 한계
<mark>① 데이터 부족: 세 모달리티를 동시에 annotate한 롱 비디오 데이터가 없다. ② 모달리티 적응 부재: 쿼리에 따라 어떤 모달리티가 중요한지 판단하는 메커니즘이 없다.</mark>

LongVALE는 모든 모달리티 토큰을 하나로 압축해 세부 정보를 잃고, 모달리티 dropout에 취약하다. Qwen2.5-Omni는 TMRoPE로 시간 정렬을 시도하지만 긴 영상의 fine-grained temporal grounding에서 여전히 부족하다.

*"Current MLLMs are generally not equipped to assess the relative importance of each modality based on task or query context."*

### 세 가지 기여
<mark>① TriSense-2M: 200만 샘플, 평균 905초 롱 비디오, 모달리티 조합 다양성 지원. ② TriSense: Query-Based Connector로 동적 모달리티 융합. ③ Segment Captioning + Moment Retrieval 8가지 모달리티 조합에서 SOTA.</mark>

## Related Work

<p><strong>흐름:</strong> 비디오 시간 이해 MLLM 리뷰 → 시간 이해 벤치마크 리뷰 → 공통 한계 정리</p>

### 비디오 시간 이해 MLLM
<mark>TimeChat, VTimeLLM, Momentor, VTG-LLM, TRACE 등이 시간 추론을 개선했으나 모두 시각 전용. LongVALE, Qwen2.5-Omni가 멀티모달을 시도하지만 적응성이 부족하다.</mark>

*"Despite these advancements, many existing models are either limited to visual modalities or lack support for flexible combinations of different modalities."*

### 벤치마크의 한계
<mark>VAST-27M, VALOR, LongVALE는 모달리티를 단순 연결(concatenation)만 하고, 모달리티가 빠졌을 때의 robustness를 고려하지 않는다.</mark>

*"In real-world videos, audio, visual, and speech inputs are not always available simultaneously, raising important questions about model robustness in the face of missing modalities."*

## Data Construction — TriSense-2M

<p><strong>흐름:</strong> 왜 새 데이터셋이 필요한가 → 파이프라인 구조(Generator + Judger) → 최종 데이터셋 통계</p>

### 왜 새 데이터셋인가
<mark>기존 데이터셋은 모든 모달리티가 항상 동시에 존재한다고 가정한다. 실제 비디오에서는 오디오가 없거나 음성이 없는 경우가 흔하다. 이 가정이 모달리티 dropout 상황에서의 실패를 낳는다.</mark>

*"This assumption limits the development of models that can handle missing or partial inputs effectively."*

<figure>
<img src="img/trisense/data_collection.jpg" alt="Figure 2">
<figcaption><strong>Figure 2</strong> — 자동화 데이터 구축 파이프라인. Generator가 세 모달리티 캡션을 AVS/AV/VS 조합으로 융합하고, Judger가 품질 점수를 부여해 3점 미만을 제거한다.</figcaption>
</figure>

### Generator + Judger 파이프라인
<mark>Qwen2.5-72B 기반 Generator가 시각·오디오·음성 캡션을 AVS/AV/VS 세 조합으로 융합. Judger가 0~5점으로 평가해 3점 미만 제거. GPT-o1으로 학습 데이터를 먼저 만들고 수동 검수.</mark>

Generator 학습 데이터 10,000개, Judger 학습 데이터 3,000개를 GPT-o1으로 생성 후 수동 필터링. InternVid와 VAST에서 원본 영상을 가져오고, 모달리티별 캡션은 기존 파이프라인으로 생성한다.

### 최종 데이터셋 통계
<mark>5백만 초기 샘플 → 200만 고품질 샘플. 약 38,000개 롱 비디오, 평균 905초 — 기존 최대인 LongVALE(235초)의 약 4배.</mark>

<figure>
<img src="img/trisense/duration.jpg" alt="Figure 3">
<figcaption><strong>Figure 3</strong> — 영상 길이 분포. 83.5%가 10~20분 롱 비디오로, 현실적인 장기 시간 이해를 지원한다.</figcaption>
</figure>

## TriSense Architecture

<p><strong>흐름:</strong> 전체 아키텍처 개요 → 멀티모달 정보 추출 → Query-Based Connector → Causal Event Prediction</p>

<figure>
<img src="img/trisense/method.jpg" alt="Figure 4">
<figcaption><strong>Figure 4</strong> — TriSense 전체 아키텍처. CLIP(시각), BEATs(오디오), Whisper(음성) 세 인코더 → Slot-Based Compression → Query-Based Connector → Time Encoder + LLM 백본.</figcaption>
</figure>

### 멀티모달 정보 추출
<mark>64프레임 균일 샘플링. 각 프레임마다 ±1초 오디오 세그먼트 추출. CLIP·BEATs·Whisper로 모달리티별 토큰 생성. Slot-Based Compression으로 각 16토큰으로 압축. Time Encoder로 타임스탬프를 6자리 문자 시퀀스로 인코딩.</mark>

타임스탬프 예시: `[123.4]` → `⟨0⟩⟨1⟩⟨2⟩⟨3⟩⟨.⟩⟨4⟩`. 세 모달리티 × 16토큰 + 시간 임베딩이 LLM에 입력된다.

### Query-Based Connector
<mark>압축된 모달리티 피처가 Cross-Attention으로 쿼리와 상호작용 → Global Average Pooling으로 각 모달리티의 전역 표현 추출 → 단층 MLP + Softmax로 가중치(wv, wa, ws) 계산 → 가중 합산 + 재압축 → 2층 MLP로 LLM 입력 차원 정렬.</mark>

$$w_m = \frac{\exp(\tilde{w}_m)}{\sum_{m'\in\{v,a,s\}} \exp(\tilde{w}_{m'})}$$

이 메커니즘 덕분에 특정 모달리티가 없으면(dropout) 해당 가중치가 자동으로 낮아지고 나머지가 보상한다.

### Causal Event Prediction
<mark>영상을 이벤트 시퀀스 `{e1, e2, ..., eK}`로 분해. 이전 이벤트들과 쿼리를 조건으로 다음 이벤트(타임스탬프 + 캡션)를 예측하는 인과 모델링 방식.</mark>

`⟨sync⟩` 특수 토큰이 Time Head(타임스탬프 예측)와 LM Head(텍스트 생성) 사이의 전환 신호 역할을 한다.

*"When the ⟨sync⟩ token is encountered, the LLM transitions between decoding modalities to generate either timestamp-aligned predictions or free-form textual outputs, depending on the task."*

## Experiments

<p><strong>흐름:</strong> 평가 태스크·메트릭·베이스라인 → TriSense-2M 결과 → LongVALE 제로샷 → 공개 벤치마크(Charades-STA, ActivityNet) → Ablation</p>

### 평가 세팅
<mark>두 태스크: Segment Captioning(BLEU-4, CIDEr, ROUGE_L, METEOR)과 Moment Retrieval(R@IoU=0.5, R@IoU=0.7, mIoU). 8가지 모달리티 조합: AVS·VS·AV·V 각각에 SC/MR 적용.</mark>

베이스라인: VTimeLLM, TimeChat, VTG-LLM, TRACE(시각 전용 VTG), LongVALE, Qwen2.5-Omni(옴니모달).

### TriSense-2M 메인 결과
<mark>AVS 세팅에서 기존 최강 LongVALE와 Qwen2.5-Omni를 크게 앞선다. 단, Visual-Only MR에서는 TRACE(128프레임) 대비 소폭 낮은데, TriSense가 64프레임만 사용하고 멀티모달 최적화 모델이기 때문이다.</mark>

*"TriSense consistently outperforms existing video LLMs across nearly all evaluated tasks. It also significantly surpasses latest omni-modal models like LongVALE and Qwen2.5-Omni, particularly in the audio-visual-speech (AVS) setting."*

### LongVALE 제로샷
<mark>LongVALE 학습 데이터 없이 제로샷으로 평가. MR에서 LongVALE(자체 데이터 학습)와 비슷한 성능. SC는 캡션 스타일 차이로 격차가 있다.</mark>

*"Our zero-shot performance on the Moment Retrieval task is comparable to LongVALE's performance, even though their model is trained on the same dataset."*

### 공개 벤치마크 제로샷 (Charades-STA, ActivityNet)
<mark>시각 전용 벤치마크에서도 경쟁력 있는 성능. 특히 **IoU=0.7** 고정밀도 기준에서 더 적은 프레임(64)으로 타 모델 대비 우위.</mark>

*"TriSense shows slightly inferior performance in Table 3, it still achieves competitive performance in visual-only settings, showing especially higher accuracy (IoU=0.7) than others, even with less frames used."*

### Ablation Studies
<mark>① 학습 단계: Stage 1만으로는 ~50% 성능. Stage 2 추가 후 급격히 향상. ② Connector: Addition(단순 합산) < Fixed Weights < 동적 가중치 순. ③ 프레임 수: 64→128 증가 시 MR에서 더 큰 향상.</mark>

Visual-Only 세팅에서는 고정 가중치(시각=1)가 동적 가중치보다 소폭 높다 — 멀티모달 최적화와 단일 모달리티 특화 간 트레이드오프.

## Conclusion

<mark>TriSense는 Query-Based Connector로 임의 모달리티 조합에서 강건한 시간 이해를 달성한다. TriSense-2M은 이 분야 가장 큰 규모의 멀티모달 롱 비디오 데이터셋이다.</mark>

*"Our modality-adaptive framework marks a substantial step toward more flexible and human-like video understanding systems."*

## Appendix A — 학습 레시피 (3단계)

<p><strong>흐름:</strong> Stage 1 Feature Alignment → Stage 2 Connector Generalization → Stage 3 Instruction Tuning</p>

### Stage 1: Feature Alignment
<mark>Query-Based Connector와 LM Head만 학습. 단일 모달리티 입력으로 각 모달리티별 특징을 개별적으로 학습. 학습 데이터: Clotho(오디오), LLaVA-LCS558K(시각), Valentini 음성 데이터셋.</mark>

4×A100-80GB GPU, 배치 512, 단일 프레임 입력, 약 10시간 소요.

### Stage 2: Connector Generalization
<mark>혼합 모달리티 데이터 투입. Connector·Time Encoder·Time Head·LM Head 학습(LLM 백본 고정). 약 880K 샘플. 16×A100-80GB GPU, 약 3.5일.</mark>

### Stage 3: Instruction Tuning
<mark>Connector를 고정하고 Time Encoder·Time Head·LM Head·LLM 백본을 학습. 약 1.12M 샘플 + LLaVA-Video-178K 일부 혼합(일반 이해 능력 유지). 16×A100-80GB GPU, 약 7일.</mark>

LLM 백본: Mistral-7B(TRACE 초기화). 시각 인코더: CLIP-ViT-L/14-336. 오디오: BEATs_iter3+. 음성: Whisper-large-V3. 최대 컨텍스트 4096토큰. 학습 시 1fps 리샘플링(추론 시 제외).

## Appendix B — TriSense-2M 상세

<p><strong>흐름:</strong> Generator·Judger 학습 과정 → 데이터 포맷</p>

### Generator·Judger 학습
<mark>GPT-o1으로 고품질 reference 데이터 생성 → 수동 필터링 → SFT. 배치 1,000개 단위로 생성 후 500개 무작위 샘플 수동 검수. 80% 이상이 기준 충족 시 배치 유지, 아니면 폐기.</mark>

<figure>
<img src="img/trisense/prompt.jpg" alt="Appendix Figure 1">
<figcaption><strong>Appendix Figure 1</strong> — Generator(좌)와 Judger(우) 학습에 사용된 GPT 프롬프트. Generator는 오디오·시각·음성 입력을 받아 옴니모달 캡션을 생성하고, Judger는 캡션 품질을 커버리지·정확성·패러프레이징 기준으로 평가한다.</figcaption>
</figure>

### 데이터 포맷
<mark>ShareGPT 스타일, 8라운드 대화. 각 라운드는 서로 다른 모달리티 조합의 태스크(예: VS-SC → AVS-MR)로 랜덤 구성. ⟨sync⟩·⟨time⟩ 특수 토큰으로 예측 헤드 전환 신호.</mark>

<figure>
<img src="img/trisense/data_format.jpg" alt="Appendix Figure 2">
<figcaption><strong>Appendix Figure 2</strong> — 학습 데이터 포맷 예시. 멀티턴 대화 구조에서 모달리티 조합과 태스크가 라운드마다 바뀐다(공간 제약으로 처음 3라운드만 표시).</figcaption>
</figure>

## Appendix C — 추가 실험

<p><strong>흐름:</strong> Slot 압축 비율 어블레이션 → Task-specific Head 어블레이션 → Query-Based Connector 브랜치 어블레이션 → VideoMME 일반 이해</p>

### Slot 압축 어블레이션
<mark>Slot 수를 8→64로 늘릴수록 성능이 소폭 향상되지만 계산 비용이 급증. Slot=16이 성능-비용 최적 균형.</mark>

### Task-specific Head 어블레이션
<mark>통합 헤드(LM Head만) 대비 Time Head + LM Head 조합이 MR에서 크게 앞선다. Time Head가 타임스탬프 정보를 추가로 제공하는 역할이 핵심.</mark>

### Query-Based Connector 브랜치 어블레이션
<mark>세 모달리티가 모두 있는 AVS-Full이 모든 태스크에서 최고. V-Only 브랜치만 써도 V-MR에서는 경쟁력 있으나 AVS 태스크에선 큰 폭으로 뒤처짐.</mark>

### VideoMME 일반 이해
<mark>일반 비디오 이해 벤치마크에서도 경쟁력 있는 성능. TRACE-uni(0.9M 일반 데이터)보다 훨씬 적은 500K 일반 데이터만 사용.</mark>

## Appendix D — 케이스 스터디

<figure>
<img src="img/trisense/avs_sc.jpg" alt="Appendix Figure 3">
<figcaption><strong>Appendix Figure 3</strong> — AVS-SC 케이스 스터디. 세 모달리티를 모두 활용한 Segment Captioning 예시. 시각·오디오·음성이 어떻게 캡션에 반영되는지 보여준다.</figcaption>
</figure>

<figure>
<img src="img/trisense/v_sc.jpg" alt="Appendix Figure 4">
<figcaption><strong>Appendix Figure 4</strong> — V-SC 케이스 스터디. 시각 전용 Segment Captioning. 오디오·음성 없이도 강건하게 동작함을 확인한다.</figcaption>
</figure>

<figure>
<img src="img/trisense/avs_mr.jpg" alt="Appendix Figure 5">
<figcaption><strong>Appendix Figure 5</strong> — AVS-MR 케이스 스터디. 세 모달리티를 쿼리와 결합해 정확한 시간 구간을 검색한다.</figcaption>
</figure>

<figure>
<img src="img/trisense/vs_mr.jpg" alt="Appendix Figure 6">
<figcaption><strong>Appendix Figure 6</strong> — VS-MR 케이스 스터디. 음성+시각 조합 Moment Retrieval.</figcaption>
</figure>

<figure>
<img src="img/trisense/av_mr.jpg" alt="Appendix Figure 7">
<figcaption><strong>Appendix Figure 7</strong> — AV-MR 케이스 스터디. 오디오+시각 조합 Moment Retrieval.</figcaption>
</figure>

<figure>
<img src="img/trisense/gen_qa.jpg" alt="Appendix Figure 8">
<figcaption><strong>Appendix Figure 8</strong> — General QA 케이스 스터디. 시간 이해를 넘어 일반적인 멀티모달 QA 태스크에서의 동작을 보여준다.</figcaption>
</figure>
