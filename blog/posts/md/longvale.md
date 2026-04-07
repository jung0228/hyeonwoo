---
title: "LongVALE: 오디오·영상·언어를 아우르는 첫 번째 옴니모달 벤치마크"
dek: "시각만 보던 비디오 AI가 소리까지 듣기 시작했다 — 105K 이벤트, 8.4K 롱 비디오, 그리고 교차 모달 추론의 시작"
tags: ["Multimodal", "Video"]
date: "2026-04-07"
readtime: "18"
slug: "longvale"
katex: false
---

## 논문 한눈에 보기

**LongVALE** (CVPR 2025)는 비디오 이해 연구의 맹점을 정면으로 겨냥한다.
기존 연구는 "시각 전용(visual-only)" 혹은 "짧은 클립" 수준에 머물렀다.
하지만 실제 세계의 비디오는 시각·오디오·음성이 뒤섞인 채 수 분에 걸쳐 흘러간다.
이 논문은 그 갭을 메우기 위해 **자동 파이프라인 → 벤치마크 → 모델**로 이어지는 완결된 연구를 제시한다.

<figure>
<img src="img/longvale/fig1.jpg" alt="Figure 1">
<figcaption><strong>Figure 1</strong> — LongVALE 데이터 예시. 악기 연주·웃음·공구 소음 등 다양한 오디오가 있는 동적인 시각 장면들. 오디오-비주얼 상관(동기성, 보완)의 실제 예를 보여주며, 내레이션 위주의 기존 데이터셋과 대비된다.</figcaption>
</figure>

## Abstract

<p><strong>흐름:</strong> 기존 한계 지적 → 데이터 부재 문제 → 파이프라인 제안 → 벤치마크 소개 → 모델 제안 → 실험 결과 예고</p>

### 기존 한계 지적
<mark>비디오 이해 연구가 발전했지만, 여전히 거친 해상도(coarse-grained)나 시각 전용 태스크에 갇혀 있다. 실제 비디오는 시각·오디오·음성이 복합된 일련의 사건들로 이루어져 있다.</mark>

> *"Despite impressive advancements in video understanding, most efforts remain limited to coarse-grained or visual-only video tasks. However, real-world videos encompass omni-modal information (vision, audio, and speech) with a series of events forming a cohesive storyline."*

### 데이터 부재 문제
<mark>옴니모달 + 세밀한 이벤트 어노테이션이 달린 데이터가 없고, 수동 레이블링 비용이 너무 높다.</mark>

> *"The lack of multi-modal video data with fine-grained event annotations and the high cost of manual labeling are major obstacles to comprehensive omni-modality video perception."*

### 파이프라인 제안
<mark>이를 해결하기 위해 고품질 필터링 → 이벤트 경계 검출 → 교차 모달 캡셔닝으로 이어지는 자동화 파이프라인을 제안한다.</mark>

> *"To address this gap, we propose an automatic pipeline consisting of high-quality multi-modal video filtering, semantically coherent omni-modal event boundary detection, and cross-modal correlation-aware event captioning."*

### 벤치마크 소개
<mark>그 결과물이 LongVALE다 — 8.4K 롱 비디오 안에 105K 옴니모달 이벤트, 정밀한 시간 경계와 관계 인식 캡션.</mark>

> *"In this way, we present LongVALE, the first-ever Vision-Audio-Language Event understanding benchmark comprising 105K omni-modal events with precise temporal boundaries and detailed relation-aware captions within 8.4K high-quality long videos."*

### 모델 + 결과 예고
<mark>LongVALE로 학습한 LongVALE-LLM은 옴니모달 세밀 시간 이해를 최초로 달성했으며, AVQA 제로샷에서도 압도적 성능을 보였다.</mark>

> *"Further, we build a baseline that leverages LongVALE to enable video large language models (LLMs) for omni-modality fine-grained temporal video understanding for the first time."*

## Introduction

<p><strong>흐름:</strong> 비디오 이해의 중요성 → 이상적 에이전트의 조건 → 기존 연구의 두 가지 한계 → 데이터 공백의 구체적 분석 → 파이프라인 세 축 설명 → LongVALE 수치 제시 → LongVALE-LLM 제시 → 기여 요약</p>

### 비디오 이해의 중요성 + 이상적 에이전트
<mark>소셜 미디어의 비디오 폭증 속에서, 이상적인 지능 에이전트는 교차 모달 추론과 세밀한 시간 이해를 동시에 갖춰야 한다.</mark>

> *"An ideal intelligent video agent should imitate it, capable of both cross-modal reasoning and fine-grained temporal understanding."*

### 기존 연구의 두 가지 한계
<mark>현재 연구는 (1) 짧은 클립의 거친 태스크(검색/캡셔닝), (2) 시각 전용 세밀 태스크(temporal grounding) 중 하나에 치우쳐, 두 능력을 동시에 갖추지 못하고 있다.</mark>

> *"However, current research is limited to coarse-grained tasks (e.g., video retrieval/captioning) or visual-only fine-grained tasks (e.g., temporal grounding/dense captioning), remaining far from enough to achieve both the capabilities."*

<div class="table-caption"><strong>Table 1</strong> — 기존 벤치마크 비교. LongVALE만이 Vision·Audio·Speech 전부 + 이벤트 타임스탬프 + A-V 상관관계를 동시에 제공한다.</div>

<div style="overflow-x:auto;margin:1.2rem 0;">
<table>
<thead>
<tr><th>Dataset</th><th>Ann.</th><th>#Videos</th><th>Avg.Len</th><th>#Events</th><th>V</th><th>A</th><th>S</th><th>Captions</th><th>Timestamps</th><th>A-V Corr.</th></tr>
</thead>
<tbody>
<tr><td>InternVid</td><td>G</td><td>234M</td><td>11.7s</td><td>1</td><td>✓</td><td>✗</td><td>✗</td><td>V</td><td>—</td><td>✗</td></tr>
<tr><td>Panda-70M</td><td>G</td><td>70.8M</td><td>8.5s</td><td>1</td><td>✓</td><td>✗</td><td>✗</td><td>V</td><td>—</td><td>✗</td></tr>
<tr><td>AudioCaps</td><td>M</td><td>51.3K</td><td>10s</td><td>1</td><td>✗</td><td>✓</td><td>✗</td><td>A</td><td>—</td><td>✗</td></tr>
<tr><td>VALOR</td><td>M</td><td>1.18M</td><td>10s</td><td>1</td><td>✓</td><td>✓</td><td>✗</td><td>VA</td><td>—</td><td>✗</td></tr>
<tr><td>VAST</td><td>G</td><td>27M</td><td>5~30s</td><td>1</td><td>✓</td><td>✓</td><td>✓</td><td>VAS</td><td>—</td><td>✗</td></tr>
<tr><td>ActivityNet Caps</td><td>M</td><td>20K</td><td>180s</td><td>3.7</td><td>✓</td><td>✗</td><td>✗</td><td>V</td><td>V</td><td>✗</td></tr>
<tr><td>UnAV-100</td><td>M</td><td>10,790</td><td>42.1s</td><td>2.8</td><td>✓</td><td>✓</td><td>✗</td><td>—</td><td>VA</td><td>✗</td></tr>
<tr style="background:#fef9c3;font-weight:700;"><td>LongVALE (Ours)</td><td>G+M</td><td>8,411</td><td>235s</td><td>12.6</td><td>✓</td><td>✓</td><td>✓</td><td>VAS</td><td>VAS</td><td>✓</td></tr>
</tbody>
</table>
</div>

### 데이터 공백 + 파이프라인 세 축
<mark>공백을 메우기 위해 세 가지 축으로 파이프라인을 설계했다: ① 고품질 필터링, ② 옴니모달 이벤트 경계 검출, ③ 오디오-비주얼 상관 추론 캡셔닝.</mark>

> *"Our pipeline includes three distinct aspects: 1) High-quality video filtering for rich audio-visual semantics and temporal dynamics. 2) Omni-modal event boundary detection for semantic coherence in both visual and audio scenes. 3) Omni-modal event captioning emphasizing audio-visual correlation reasoning."*

### 기여 요약
<mark>① 자동 파이프라인, ② LongVALE 벤치마크, ③ LongVALE-LLM의 세 가지 기여.</mark>

> *"We introduce LongVALE, the first-ever benchmark providing omni-modal event temporal boundaries and cross-modal correlation-aware captions for 105K omni-modal events within 8.4K high-quality multi-modal long videos."*

## Related Work

<p><strong>흐름:</strong> 멀티모달 비디오 벤치마크 리뷰 → 세밀한 비디오 이해 리뷰 → LongVALE의 위치 정리</p>

### 멀티모달 벤치마크의 한계
<mark>InternVid, VAST 등 대규모 데이터셋은 짧은 클립에 거친 캡션만 제공하고, 모달리티를 단순 연결(concatenation)할 뿐 교차 모달 추론이 없다. 세밀한 어노테이션이 있는 ActivityNet/Charades-STA는 시각 전용.</mark>

> *"These benchmarks offer only coarse-grained captions for short clips, which are unsuitable for fine-grained long video understanding... fine-grained video benchmarks like ActivityNet Caps and Charades-STA focus only on visual modality."*

### 세밀한 비디오 이해 연구
<mark>Temporal grounding, dense captioning 등 세밀 태스크 연구들이 존재하지만 모두 시각 전용. 최근 Video LLM들도 마찬가지다.</mark>

> *"Recent video large language models (video LLMs) have shown promise in visual-only fine-grained video understanding. In contrast, we aim to pioneer omni-modality fine-grained video understanding for a more holistic video comprehension."*

## Section 3: The LongVALE Benchmark

<p><strong>흐름:</strong> 데이터 수집·필터링 → 옴니모달 이벤트 경계 검출(시각 / 오디오 / 합성) → 옴니모달 이벤트 캡셔닝(시각 / 오디오+음성 / 관계 인식) → 데이터 분할 + 수동 검수 → 통계 분석</p>

### 데이터 수집 및 필터링
<mark>ACAV-100M에서 YouTube 원본 영상을 수집하고, 4단계 필터링(해상도·음성 비율·정적 장면·오디오-비주얼 일치도)으로 100K → 8.4K로 압축. 엄격한 품질 기준.</mark>

> *"We source videos from ACAV-100M... Then, we design a filtering strategy to obtain high-quality videos containing rich visual and audio semantics, as well as temporal dynamic information."*

**필터링 4단계:**
1. 해상도 360p 미만 제거, 영어 자막 보유 영상만 선택
2. 음성이 95% 이상 차지하는 영상 제거 (다양한 소리 확보)
3. PySceneDetect로 정적 슬라이드쇼 제거
4. C-MCR 모델로 오디오-비주얼 유사도 측정 → 관련 없는 배경음악/더빙 제거

### 시각 이벤트 경계 검출
<mark>기존 Panda 2단계 방식(기본 장면 분할 → 유사 장면 병합)을 롱 비디오에 맞게 개선. 정적 장면·전환 클립 후처리로 제거.</mark>

> *"Using only visual cues, we apply a two-stage detection method which includes splitting basic visual scenes and then merging semantically similar ones."*

### 오디오 이벤트 경계 검출 ← 핵심 기여
<mark>사전 정의된 카테고리 없이 일반 오디오 이벤트 경계를 검출하는 최초의 방법. MFCC로 스펙트럼 변화 감지 → CLAP으로 의미 유사도 계산 → 인접 클립 병합.</mark>

> *"Although audio is crucial for temporal video understanding, no method exists for detecting generic audio event boundaries without pre-defined categories. To fill this gap, we design a generic method that segments long audio sequences into semantically coherent clips leveraging distinct audio properties."*

### 옴니모달 이벤트 경계 합성
<mark>시각 경계를 기준으로 하되, 그 안의 오디오 이벤트를 잘라내지 않고 끝까지 포함시켜 오디오 의미 무결성 보장.</mark>

> *"To define omni-modal event boundaries, we primarily rely on visual boundaries while preserving the integrity of audio events. Specifically, for each visual event, we set its start time as the beginning of the omni-modal event and include all overlapping audio events without truncation."*

<figure>
<img src="img/longvale/fig2.jpg" alt="Figure 2">
<figcaption><strong>Figure 2</strong> — 전체 파이프라인. 시각/오디오 경계를 각각 검출 후 합성하고, 모달리티별 캡션을 생성한 뒤 Gemini로 교차 모달 추론 캡션을 최종 생성한다.</figcaption>
</figure>

### 모달리티별 캡셔닝
<mark>시각(LLaVA-NeXT-Video + GPT-4o 키프레임), 오디오(Qwen-Audio), 음성(Whisper-Large)으로 각각 캡션 생성. 단순 연결이 아니라 다음 단계에서 Gemini로 통합.</mark>

> *"Specifically, we employ LLaVA-NeXT-Video to caption each video event... Qwen-Audio, a general large audio-language model, to obtain detailed audio descriptions... Whisper-Large, a strong automatic speech recognition (ASR) model."*

### 관계 인식 옴니모달 캡셔닝
<mark>Gemini-1.5-Pro가 세 모달리티 캡션을 통합하면서 오디오-비주얼 상관관계(동기성, 보완, 인과성 등)를 명시적으로 추론. 이전 이벤트 캡션을 컨텍스트로 제공해 일관성 확보.</mark>

> *"Instead of simply concatenating modality-specific captions, we instruct Gemini-1.5-Pro to establish meaningful connections between them, such as analyzing whether audio events are visible, identifying sound sources, and reasoning about causality."*

### 통계 분석
<mark>8,411개 비디오, 105,730개 이벤트, 평균 영상 길이 3.9분, 평균 이벤트 12.6개/영상, 오디오-비주얼 상관 7가지 유형 분류.</mark>

> *"It includes 8,411 long videos spanning over 549 hours, with an average video duration of 3.9 minutes. The dataset contains 105,730 omni-modal events, each annotated with accurate temporal boundaries and omni-modality relation-aware captions."*

<figure>
<img src="img/longvale/fig3.jpg" alt="Figure 3">
<figcaption><strong>Figure 3</strong> — LongVALE 통계. (a) 영상 길이 분포, (b) 이벤트 수 분포, (c) 이벤트 지속시간 분포, (d) 오디오-비주얼 상관 유형 분포. 보완(complementary)·동기성(synchronicity)·강화(enhancement)가 가장 흔한 유형.</figcaption>
</figure>

## Section 4: LongVALE-LLM

<p><strong>흐름:</strong> 전체 아키텍처 설명 → 학습 레시피 (경계 인식 튜닝 → 명령어 튜닝)</p>

### 전체 아키텍처
<mark>멀티모달 인코더(CLIP/BEATs/Whisper)로 각 모달리티 특징 추출 → 어댑터로 LLM 임베딩 공간 매핑 → 시퀀스 차원으로 연결 → Vicuna-7B가 자기회귀적으로 응답 생성.</mark>

> *"Given a long video, the multi-modal encoders first extract modality-specific token features, which are then mapped into the LLM's embedding space via the multi-modal adapters. The embeddings from different modalities are concatenated along the sequence dimension and combined with the task instruction to form the prefix input to the LLM."*

<figure>
<img src="img/longvale/fig4.jpg" alt="Figure 4">
<figcaption><strong>Figure 4</strong> — LongVALE-LLM 아키텍처. 시각(CLIP), 오디오(BEATs), 음성(Whisper) 세 인코더가 각 어댑터를 거쳐 LLM에 합쳐진다. 두 단계 학습(경계 인식 튜닝 → 명령어 튜닝)도 함께 도식화.</figcaption>
</figure>

### 2단계 학습
<div class="summary-highlight">

- **1단계 (경계 인식 튜닝):** 옴니모달 이벤트와 시간 경계를 대화 형식으로 변환 (7,240개). 템플릿 기반 단일/다중 턴 QA.
- **2단계 (명령어 튜닝):** Gemini로 자유 형식 대화 생성 (25,400개). 명령어 따르기 + 추론 능력 향상.
</div>

> *"Although our model demonstrates the ability to perceive omni-modal event boundaries after boundary perception tuning, its outputs tend to overfit to templated answers. To improve the model's ability to follow human instructions... we create high-quality instruction-tuning data based on our LongVALE."*

## Section 5: Experiments

<p><strong>흐름:</strong> 실험 설정 → 메인 결과 (3가지 옴니모달 태스크) → 제로샷 AVQA → 절제 연구 (학습 데이터 / AVC 효과 / 모달리티) → 정성적 결과</p>

### 실험 설정
<mark>시각(CLIP ViT-L/14, 100프레임), 오디오(BEATs), 음성(Whisper-Large-v2), LLM(Vicuna-7B). 세 가지 평가 태스크: Omni-TVG(R@IoU, mIoU), Omni-DVC(SODA_c, CIDEr, METEOR), Omni-SC(BLEU-4, ROUGE-L, METEOR, CIDEr).</mark>

### 메인 결과 — 압도적 성능 차이
<mark>LongVALE-LLM이 기존 모든 Video LLM을 세 태스크에서 큰 차이로 앞선다. 오디오-비주얼 입력을 지원하는 VideoLLaMA/NExT-GPT도 프레임 수 제한(8프레임)으로 세밀 태스크에서 실패. 시간 이해 특화 VTimeLLM/TimeChat은 오디오 정보를 못 쓴다.</mark>

> *"Our LongVALE-LLM (7B) supports video, audio and speech input with fine-grained temporal understanding ability, and outperforms other video LLMs by a significant margin across all three tasks."*

### 제로샷 AVQA — 데이터 효율성
<mark>OneLLM(460K 오디오 데이터), AVicuna(350K)보다 적은 32.7K 샘플만 써도 AVSD에서 SOTA 달성. LongVALE의 교차 모달 추론 캡션이 일반화 능력을 끌어올림.</mark>

> *"Despite using significantly less data, our model surprisingly achieves state-of-the-art performance on AVSD... using only total 32.7K audio-visual samples from our LongVALE dataset, accounting for less than 10% of the data they use."*

### 절제 연구
<mark>세 가지 절제로 각 설계 결정을 검증.</mark>

1. **학습 단계별 데이터 영향:** 두 단계 모두에 LongVALE 데이터 추가 시 모든 태스크 큰 향상. 명령어 튜닝은 특히 세그먼트 캡셔닝 대폭 개선.
2. **AVC(오디오-비주얼 상관) 효과:** 단순 연결 대비 AVC 추론 포함 캡션으로 학습 시 캡셔닝 태스크에서 현저한 향상.
3. **모달리티 추가 효과:** V → V+A → V+S → V+A+S 순서로 모든 태스크 성능이 일관되게 향상.

### 정성적 결과
<mark>VTimeLLM은 시각만 보고 "손을 드는 행동"으로 오인. LongVALE-LLM은 오디오(남성 노래, 군중 환호)를 통합해 정확히 묘사.</mark>

<figure>
<img src="img/longvale/fig5.jpg" alt="Figure 5">
<figcaption><strong>Figure 5</strong> — 정성적 비교. VTimeLLM(시각 전용) vs LongVALE-LLM. 주황색 텍스트가 오디오-비주얼 교차 모달 추론의 핵심 부분. 오른쪽은 Music-AVQA 제로샷 예시.</figcaption>
</figure>

## Conclusion

<p><strong>흐름:</strong> 기여 재요약 → 모델 능력 강조 → 향후 방향</p>

### 기여 + 의의
<mark>LongVALE는 교차 모달 추론과 세밀한 시간 이해라는 두 가지 능력을 동시에 갖춘 첫 번째 시도. 지능형 비디오 어시스턴트로 가는 중요한 발걸음.</mark>

> *"Our model exhibits distinct capabilities of both cross-modal reasoning and fine-grained temporal understanding that are absent in existing video LLMs, making a crucial step toward an intelligent video assistant."*

### 향후 방향
<mark>더 많은 고품질 데이터로 LongVALE 확장 + 비디오 의미 밀도와 교차 모달 상호작용 개선을 위한 아키텍처 고도화.</mark>

> *"In the future, we will expand our LongVALE with more high-quality data and advance the model's architecture to improve video semantic density and cross-modal interaction."*


## Supplement A: LongVALE 벤치마크 상세

<p><strong>흐름:</strong> 이벤트 경계 정량 검증 → 추가 통계 → 수동 검수 과정 → 캡셔닝 프롬프트</p>

### 이벤트 경계 정량 분석 — MRSD

<mark>각 모달리티 이벤트의 의미적 일관성을 정량화하기 위해 MRSD(Max Running Semantic Difference) 지표를 도입. 이벤트 내 최대 의미 변화량을 측정한다.</mark>

MRSD가 낮을수록 이벤트 내 의미가 일관됨. 1차 분할(splitting) 후 유사 클립 병합(stitching)을 거치면 이벤트가 지나치게 짧아지지 않으면서도 의미 일관성을 유지함을 수치로 확인.

<div style="overflow-x:auto;margin:1.2rem 0;">
<table>
<thead>
<tr><th>Method</th><th>MRSD-V ↓</th><th>MRSD-A ↓</th><th>Avg.Len</th></tr>
</thead>
<tbody>
<tr><td>Visual boundary (splitting)</td><td>0.531</td><td>—</td><td>3.0s</td></tr>
<tr><td>Visual boundary (stitching)</td><td>0.532</td><td>—</td><td>10.7s</td></tr>
<tr><td>Audio boundary (splitting)</td><td>—</td><td>0.676</td><td>1.5s</td></tr>
<tr><td>Audio boundary (stitching)</td><td>—</td><td>0.703</td><td>5.8s</td></tr>
<tr style="background:#fef9c3;font-weight:700;"><td>Omni-modal boundary</td><td>0.601</td><td>0.784</td><td>16.7s</td></tr>
</tbody>
</table>
</div>

### 추가 통계 — 카테고리 분포 & 워드 클라우드

<mark>LongVALE는 YouTube 메타데이터 기준 다양한 카테고리를 포괄. 옴니모달 캡션의 단어 분포는 시각·오디오·음성이 고루 반영된 풍부한 내용을 담고 있다.</mark>

<figure>
<img src="img/longvale/video_category.png" alt="Video Category Distribution">
<figcaption><strong>Fig. S1</strong> — 비디오 카테고리 분포. LongVALE가 광범위한 주제를 커버함을 보여준다.</figcaption>
</figure>

<figure>
<img src="img/longvale/word_cloud.png" alt="Word Cloud">
<figcaption><strong>Fig. S2</strong> — 옴니모달 캡션 길이 분포 및 워드 클라우드. 시각·오디오·음성 정보가 고루 담긴 풍부한 어휘를 확인할 수 있다.</figcaption>
</figure>

### 수동 검수 과정

<mark>테스트 셋 2K 영상을 두 그룹이 검수·수정. 1그룹은 경계·캡션 정확도 검토, 2그룹은 오류 수정. 총 115인시(human hours), 약 300개 오류 수정.</mark>

<figure>
<img src="img/longvale/manual.png" alt="Manual Check Interface">
<figcaption><strong>Fig. S3</strong> — 수동 검수·수정 인터페이스 스크린샷.</figcaption>
</figure>

### 캡셔닝 & AV 상관 분석 프롬프트

<mark>LLaVA-NeXT-Video(동적 정보) + GPT-4o(키프레임 공간 정보) + Qwen-Audio + Whisper로 모달리티별 캡션 생성 후, Gemini-1.5-Pro가 교차 모달 상관을 추론해 최종 캡션 생성.</mark>

> *"Note that we found that the performance of the audio captioner lags significantly behind that of visual models, leading to more hallucination issues... we cleaned up these generations, retaining only general descriptions for each audio event while removing the specific speech content. Accurate ASR outputs generated by the advanced speech recognition model were used as replacements."*

<figure>
<img src="img/longvale/caption_prompt.png" alt="Captioning Prompts">
<figcaption><strong>Fig. S4</strong> — 비디오 클립·키프레임·오디오 캡셔닝 프롬프트 및 옴니모달 통합 프롬프트.</figcaption>
</figure>

<figure>
<img src="img/longvale/avc.png" alt="AV Correlation Prompt">
<figcaption><strong>Fig. S5</strong> — 오디오-비주얼 상관관계 및 시간적 다이나믹스 분석 프롬프트.</figcaption>
</figure>


## Supplement B: 태스크·모델·학습 상세

<p><strong>흐름:</strong> 3가지 태스크 정의 → 모델 아키텍처 수식 → 학습 데이터 구성</p>

### 3가지 옴니모달 태스크 정의

<div class="summary-highlight">

- **Omni-TVG (Temporal Video Grounding):** 텍스트 쿼리가 기술하는 옴니모달 이벤트의 시작·끝 타임스탬프를 찾아라.
- **Omni-DVC (Dense Video Captioning):** 비디오 내 모든 옴니모달 이벤트를 시간 위치와 함께 캡셔닝하라.
- **Omni-SC (Segment Captioning):** 주어진 시간 구간의 옴니모달 이벤트를 설명하라.
</div>

### 모델 아키텍처 수식

<mark>세 인코더(CLIP / BEATs / Whisper)가 각 어댑터를 거쳐 토큰 시퀀스로 변환되고, 단순 연결(concat)되어 LLM에 입력된다. 수식: Z = Concat(F_V, F_A, F_S), Z ∈ ℝ^(N×d).</mark>

BEATs(비음성 오디오)와 Whisper(음성)는 상호 보완적으로 동작해 일반 오디오 입력을 모두 처리. 모달리티가 없는 경우 단일·이중 모달 입력도 지원.

### 학습 데이터 구성

<mark>경계 인식 튜닝: 20% 단일 턴(Omni-DVC), 80% 다중 턴(Omni-TVG + Omni-SC). 명령어 튜닝 프롬프트는 Gemini가 이벤트 경계·캡션을 분석해 자유 형식 대화 생성.</mark>

<figure>
<img src="img/longvale/instruction_prompt.png" alt="Instruction Tuning Prompt">
<figcaption><strong>Fig. S6</strong> — 옴니모달 명령어 튜닝 데이터 생성 프롬프트.</figcaption>
</figure>


## Supplement C: 실험 상세

<p><strong>흐름:</strong> 구현 세부사항 → 평가 쿼리 설계 (자사 모델 / 비교 모델)</p>

### 구현 세부사항

<mark>2 에폭, 배치 128, AdamW + 코사인 LR 감쇠, LR=1e-4, LoRA rank=64 / alpha=128. 경계 인식 단계 LoRA를 LLM에 병합 후 명령어 튜닝 단계에서 새 LoRA 추가. RTX A100 40G GPU 1장, 30시간 학습.</mark>

### 평가 쿼리 설계

<mark>각 태스크별로 세심하게 설계된 쿼리 사용. 비교 모델들은 학습 방식이 달라 최적 쿼리를 따로 설계하고 GPT-4o mini로 타임스탬프 추출.</mark>

예시 쿼리:
- **Omni-DVC:** *"Could you please detail the events that took place during different time segments in the video? List the events in the format: From xx to xx, event..."*
- **Omni-TVG:** *"During which frames does \<event\> occur in the video? Give the timestamps in the format: From xx to xx."*
- **Omni-SC:** *"Can you describe what occurred from \<start\> to \<end\> in the video? Please give the event description directly."*


## Supplement D: 추가 정성적 결과

<p><strong>흐름:</strong> Segment Captioning → Temporal Grounding → AVQA → Dense Video Captioning</p>

<mark>네 태스크 모두에서 LongVALE-LLM이 시각 전용 모델(VTimeLLM) 대비 오디오-비주얼 통합 추론으로 더 풍부하고 정확한 결과를 제공함을 보여준다.</mark>

<figure>
<img src="img/longvale/segment_visual.png" alt="Segment Captioning">
<figcaption><strong>Fig. S7 — Omni-SC.</strong> VTimeLLM은 시각 이벤트만 간략히 묘사. LongVALE-LLM은 동적·청각 정보를 포함한 풍부한 설명 제공.</figcaption>
</figure>

<figure>
<img src="img/longvale/grounding_visual.png" alt="Temporal Grounding">
<figcaption><strong>Fig. S8 — Omni-TVG.</strong> 옴니모달 이벤트 캡션이 주어졌을 때 정확한 시간 구간 탐지. 초록색이 정답 경계.</figcaption>
</figure>

<figure>
<img src="img/longvale/avqa_visual.png" alt="AVQA">
<figcaption><strong>Fig. S9 — AVQA.</strong> 시각·청각 단서를 통합해 "가장 큰 소리의 악기 위치"와 같은 일반 오디오-비주얼 질문에 정확히 답변.</figcaption>
</figure>

<figure>
<img src="img/longvale/dense_visual.png" alt="Dense Video Captioning">
<figcaption><strong>Fig. S10 — Omni-DVC.</strong> 더 많은 옴니모달 이벤트를 탐지하고 시각·오디오 핵심 정보를 담은 세밀한 캡션 생성.</figcaption>
</figure>


## Supplement E: 사회적 영향

<mark>데이터 생성 시 Gemini의 안전 필터로 유해 콘텐츠 차단. NLTK로 개인 이름 제거해 프라이버시 보호. 데이터 소스: ACAV-100M (MIT License). LongVALE 어노테이션 공개 라이선스: CC BY-NC-SA 4.0.</mark>
