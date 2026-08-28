# 📑 [연구제안서] Omni-modal Video Token Compression: 시공간 2축 적응형 토큰 압축 기반 차세대 온디바이스 옴니모달 아키텍처
- **연구 책임자**: 정현우 (POSTECH / AI 대학원 지원 연구계획서)
- **작성 일자**: 2026년 08월 26일
- **분야**: Multimodal Large Language Models (MLLMs) / Efficient Inference / Token Compression / On-Device Computing
- **참조 논문군**: [[paper_fastv]], [[paper_qwen2_vl]], [[paper_llava]], [[paper_universal_audio_generation]], [[paper_marmot_masked_autoencoder_for_modeling_transient_i]]

---

## 1. Executive Summary & Problem Formulation (연구 배경 및 수치화된 극한 병목)

### 1.1 트랜스포머 $O(N^2)$ 연산 복잡도와 멀티모달 토큰 폭증의 충돌
최신 멀티모달 대형 언어 모델(MLLM)은 텍스트를 넘어 초고해상도 이미지, 장시간 비디오, 실시간 음성을 포괄하는 Omni-modal 환경으로 급속히 진화하고 있다. 그러나 트랜스포머의 Self-Attention 메커니즘은 시퀀스 길이 $N$에 대해 **$O(N^2)$의 계산 복잡도와 $O(N)$의 KV Cache 메모리 풋프린트**를 요구한다.

- **극단적 토큰 팽창 수치**: 텍스트 프롬프트가 수백 토큰에 불과한 반면, **90분 분량의 단일 비디오는 최대 5,400만($54\text{M}$) 토큰을 생성**한다.
- **온디바이스의 물리적 한계**: 스마트 글래스, 로봇, 엣지 보드(Jetson, NPU)의 제한된 SRAM(수십 MB) 및 DRAM 대역폭 환경에서 이러한 초장길이 시퀀스는 실시간 추론을 완벽히 마비시키는 '연산 및 전력 절벽'을 야기한다.

---

## 2. Frontier Blind Spots & Prior Art Critique (선행 연구 맹점 비판 및 연구 갭)

본 제안서는 최근 제안된 최전선 가속 기법들의 상호 모순과 해결되지 않은 미개척 영역(Research Gap)을 다음과 같이 규명한다:

| 선행 연구 패러다임 | 대표 논문 | 핵심 장점 | 결정적 맹점 (Frontier Blind Spot) |
| :--- | :--- | :--- | :--- |
| **Attention-based Pruning** | **FastV** (ECCV 2024) | Layer 2 이후 하위 50% 토큰 드롭, FLOPs 45% 감축 | • **정적 단일 이미지 국한**: 비디오 시간축 인과관계(Temporal Causality) 왜곡<br>• 질문 복잡도 무관 고정 $K=2, R=50\%$ 컷의 정보 소실 |
| **Time-wise Cache Eviction** | **StreamKV** | 슬라이딩 윈도우 기반 중요 KV 토큰만 보존 | • 토큰 텐서 자체의 FFN 연산은 줄이지 못하고 캐시 크기만 제한<br>• 시각 토큰의 공간적 중복성을 반영하지 못함 |
| **Modality Gating** | **OmniSelect** (2026) | AudioCLIP 기반 음성/영상 프루닝 비율 동적 분기 | • 거대 LLM 내부 레이어 간 깊이별 정보 집약(Anchor Sink) 특성을 활용하지 못함 |

> 🚨 **핵심 연구 갭(Research Gap)**:  
> 기존 연구들은 **'공간 축(Depth-wise FastV)'** 또는 **'시간 축(Time-wise StreamKV)'** 중 단 하나만을 파편적으로 다루었으며, **사용자 질문의 의도(Query Complexity)에 따라 시공간 2축을 동시에 입체적으로 압축하는 통합 프레임워크**는 전무하다.

---

## 3. Proposed Methodology: Spatiotemporal 2-Axis Adaptive Compression (핵심 제안 기법)

본 연구는 **[FastV의 깊이 축 토큰 프루닝] $\times$ [StreamKV의 시간 축 캐시 관리] $\times$ [OmniSelect의 질문 반응형 모달리티 게이팅]**을 수학적으로 융합한 **`Dual-Axis Adaptive MLLM (DA-MLLM)`** 아키텍처를 제안한다.

### 3.1 수학적 수식화 (Mathematical Formulation)

1. **질문 기반 모달리티 중요도 벡터 산출**:
   입력 텍스트 질문 $Q$와 비디오-오디오 멀티모달 스트림 $V = \{v_t\}, A = \{a_t\}$에 대해, 경량 프로젝터를 통해 모달리티별 민감도 계수 $\alpha_v, \alpha_a \in [0, 1]$를 동적 산출:
   $$\alpha_v, \alpha_a = \text{Softmax}(\mathbf{W}_g \cdot [\text{Embed}(Q); \text{Pooling}(V); \text{Pooling}(A)])$$

2. **레이어 축 깊이별 적응형 프루닝 (Depth-wise Adaptive Pruning)**:
   고정 $K=2$ 레이어가 아닌, 질문 복잡도 $\mathcal{C}(Q)$에 따라 프루닝 시점 $K^*$와 비율 $R^*$를 가변 결정:
   $$K^* = \lfloor K_0 + \beta \cdot \mathcal{C}(Q) \rfloor, \quad R^* = R_0 \cdot (1 - \alpha_v)$$
   Layer $K^*$ 통과 후 평균 어텐션 점수 $\phi_{\text{attn}}(t)$ 하위 $R^*$ 토큰을 FFN 통과 직전 텐서에서 영구 배제.

3. **시간 축 캐시 슬라이딩 정렬 (Time-wise KV Cache Alignment)**:
   가변 토큰 프루닝으로 인해 발생하는 PagedAttention 메모리 인덱싱 충돌을 방지하기 위해, **Time-Anchor Indexing Mask**를 설계하여 공간 토큰은 삭제하되 각 타임스탬프의 기준 앵커 토큰만 KV 캐시에 압축 유지.

---

## 4. Quantitative Impact & Verification Plan (정량적 목표 및 실험 검증)

### 4.1 정량적 목표 지표 (Target Metrics)
* **연산량(FLOPs)**: 바닐라 LLaVA/Qwen2-VL 대비 **65% 이상 절감**.
* **메모리 풋프린트**: 1시간 비디오 스트리밍 시 KV Cache 메모리 **80% 이상 감축** (16GB GPU 내 상주 달성).
* **추론 속도 (FPS)**: 초당 프레임 처리 속도 **3.2배 향상** (Jetson Orin 엣지 기준 15 FPS ➔ 48 FPS 실시간성 확보).
* **정확도 보존**: Video-MME, MMMU, ActivityNet-QA 등 핵심 벤치마크에서 풀 토큰 대비 **98.5% 이상 성능 유지**.

### 4.2 실험 설계 (Evaluation Protocol)
1. **Ablation Study**:
   - Depth-wise Pruning만 적용 vs Time-wise Cache만 적용 vs 제안 2축 통합 아키텍처 비교.
   - 고정 비율(FastV 스타일) 대비 질문 반응형 동적 게이팅의 정확도 방어율 입증.
2. **On-Device 실기기 벤치마크**:
   - Apple Silicon (M-series MLX) 및 NVIDIA Jetson 엣지 디바이스에서 실제 전력 소모(Watt), 발열, Latency 측정.

---

## 5. Killer Application: On-Device Multimodal Life-Logging (실사용 파급 효과)

* **24시간 올데이 스마트 글래스 (AI Life-Companion)**:
  - 사용자가 착용한 안경에서 하루 종일 유입되는 고화질 시야와 음성을 배터리 방전 및 발열 없이 실시간 인과 추론.
  - "내가 3시간 전에 열쇠를 어디 뒀더라?"라는 질문에 3시간 치의 잉여 비디오 토큰은 모두 쳐내고, 핵심 앵커 타임스탬프만 역추적하여 0.5초 만에 정확한 답변 도출.
