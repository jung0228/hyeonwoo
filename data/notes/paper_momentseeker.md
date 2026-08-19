# 📄 [Paper] Momentseeker: A Benchmark for Long-Video Moment Retrieval
- **Authors & Venue**: CVPR 2025
- **Domain**: Multimodal / Video Understanding / Temporal Grounding
- **Connected**: [[long_video_understanding]], [[streamkv]], [[vision_encoder]], [[rq_video_temporal_grounding]]

---

## 1. Problem Formulation & Frontier Blind Spot (문제 정의 및 기존 연구 결함)
- **Unaddressed Bottleneck**: 기존 비디오 벤치마크(Charades-STA, ActivityNet Captions 등)는 평균 1~3분의 극단적으로 짧은 클립에 편향되어 있어, 실제 실생활(유튜브, 영화, 수술 영상, 감시 카메라 등)의 수십 분~수 시간짜리 초장거리 비디오를 처리하는 능력을 전혀 평가하지 못함.
- **Core Limitation of Prior Art**: 기존 SOTA VLM들은 장시간 비디오에서 균등 프레임 샘플링(Uniform Sampling)을 적용하여, 1초 미만의 결정적 찰나 액션(Moment)을 통째로 누락하는 구조적 결함을 가짐.

---

## 2. Core Benchmark Architecture (데이터셋 및 평가 메커니즘)
- **벤치마크 구성**:
  - 평균 영상 길이 25분 ~ 2시간 이상의 다양한 장르(다큐멘터리, 스포츠, 일상 VLOG, 게임 등) 수집.
  - 시간적 인과관계(Temporal Causality)와 미세 액션(Fine-grained Action)을 요구하는 정밀한 자연어 쿼리 쌍 구축.
- **평가 지표**:
  - $\text{Recall}@K, \text{IoU}=m$: 상위 $K$개 예측 구간 중 정답 구간과의 Temporal IoU(tIoU)가 $m$ 이상인 비율.
  - Mean Average Precision (mAP) over multiple IoU thresholds $[0.3, 0.5, 0.7]$.

---

## 3. Findings & Failure Modes (발견점 및 실패 모드 군집)
- ⚠️ **SOTA 모델들의 참패**: GPT-4V, Gemini 1.5 Pro, Video-LLaVA 등 당대 최고 모델들도 비디오 길이가 15분을 초과하는 순간 정확도가 50% 이상 급락.
- ⚠️ **실패 모드 군집 (Failure Clusters)**:
  1. **초기/말미 편향(Lost in the Middle)**: 비디오 앞부분과 뒷부분에만 어텐션이 쏠려 중간 구간의 중요한 사건을 놓침.
  2. **Transient Clue Eviction**: KV Cache 메모리 제약으로 인해 순간적으로 지나간 핵심 시각 단서가 퇴출되어 오답 발생.

---

## 4. Hyeonwoo's Research Vector (나의 연구 아이디어 연계 - KAIST DAVIAN 랩 연계)
- **발굴 벡터**: [실패 모드 군집화] + [기저 가정 파괴]
- **연구 가설**: KAIST DAVIAN 랩에서 주도했던 LVMR 평가 파이프라인 경험을 기반으로, 고정 샘플링 대신 사건 변화율(Motion Entropy) 기반의 가변 샘플링과 Dynamic Temporal Gating 메커니즘을 통합 설계 $\rightarrow$ [[rq_video_temporal_grounding]]
