# 📄 [Paper] FastV: An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models
- **Authors**: Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Haozhe Jia, Junyang Lin, Chang Zhou, Baobao Chang (Peking University, Alibaba Group)
- **Venue / Year**: ECCV 2024 / arXiv 2024
- **Domain**: Multimodal / Vision-Language Model / Plug-and-Play Inference Acceleration & Token Pruning
- **Connected**: [[paper_llava]], [[paper_qwen2_vl]], [[transformer]], [[kv_cache]], [[streamkv]], [[long_video_understanding]], [[ondevice_multimodal_lifelogging]]

---

## 1. Problem Formulation & Frontier Blind Spot (문제 정의 및 선행 연구 결함)
- **Unaddressed Bottleneck**: LLaVA, GPT-4V 등 현대 LVLM은 이미지를 수백~수천 개(예: LLaVA 576개, Video-LLaVA 2,048개)의 시퀀스 토큰으로 변환해 LLM에 입력함. 트랜스포머의 계산 복잡도는 시퀀스 길이의 제곱($O(N^2)$)에 비례하므로, 시각 토큰 수가 늘어날수록 Self-Attention 및 거대한 FFN(Feed-Forward Network) 연산량과 KV 캐시 메모리 병목이 극심해짐.
- **Core Limitation of Prior Art**: 기존 LLM 추론 가속 기법(FlashAttention, vLLM, Sparse Attention 등)은 주로 순수 텍스트 모델에 집중되어 있었음. 반면 멀티모달 모델에서 "이미지 토큰이 트랜스포머 레이어를 통과하며 실제로 어떻게 소비되는가"에 대한 근본적인 메커니즘 분석과 LVLM 전용 가속 연구는 부재했음.

---

## 2. Core Hypothesis & Architecture (핵심 기법 및 발견)

### 🔍 핵심 발견: 비효율적인 시각 어텐션 (Inefficient Visual Attention & Figure 3)
- 저자들은 LLaVA-1.5-7B/13B의 전 레이어에 걸쳐 토큰 유형별 어텐션 분배($\lambda$) 및 토큰당 어텐션 효율($\epsilon$)을 정량화함.
  1. **초반 층 (Layer 1~2)**: 이미지 토큰이 비교적 높은 어텐션을 받음 (시스템 프롬프트 대비 약 50% 수준).
  2. **깊은 층 (Layer 3~32)**: 이미지 토큰이 받는 어텐션이 시스템 프롬프트 대비 **단 0.21%**로 떡락함. 반면 시스템 프롬프트(`sys`)는 전체 어텐션의 85% 이상을 독점(Attention Sink).
  3. **어텐션 효율 ($\epsilon = \frac{\text{총 어텐션 점수}}{\text{토큰 개수}}$)**: 깊은 층에서 시스템 프롬프트 토큰 1개의 효율이 이미지 토큰 1개보다 무려 **472배** 높음.
- **메커니즘 원인 (Anchor Token Aggregation)**:
  - 이미지의 고유한 공간적 중복성(Redundancy)으로 인해, 모델은 Layer 1~2의 Self-Attention 단계에서 이미지의 핵심 정보를 시퀀스 앞단의 **앵커 토큰(시스템 프롬프트 및 텍스트 앵커)**으로 대부분 압축·흡수(Aggregation)시킴.
  - 따라서 Layer 3 이후의 언어 생성 단계에서는 원본 이미지 토큰을 사실상 쳐다보지 않고(투명인간 취급) 텍스트 앵커만 보며 디코딩을 수행함.

### ⚡ FastV 알고리즘 메커니즘
- **핵심 가설**: "어차피 Layer 2 이후에는 보지도 않을 이미지 토큰이라면, Layer 2가 끝난 직후 하위 50%를 삭제(Pruning)해도 생성 성능은 보존될 것이다."
- **알고리즘 파이프라인**:
  1. **초기 통과 ($1 \le l \le K$)**: 하이퍼파라미터 $K$ (기본값 $K=2$) 레이어까지는 원본 이미지 토큰 전체를 정상 연산하여 시각 정보를 앵커 토큰에 충분히 전이시킴.
  2. **재정렬 및 필터링 (Ranking & Pruning at Layer $K$)**: Layer $K$ 출력 시점에서 각 이미지 토큰이 다른 모든 토큰으로부터 받은 평균 어텐션 점수 $\phi_{\text{attn}}$를 계산하여 순위를 매김.
  3. **하위 토큰 영구 삭제**: 하위 $R\%$ (기본값 $R=50\%$)의 이미지 토큰을 텐서에서 완전히 드롭(Discard).
  4. **후반 레이어 추론 ($K < l \le L$)**: Layer 3부터 최종 레이어까지는 살아남은 절반의 토큰만 가지고 연산 수행.
- **Sparse Attention 대비 본질적 우위**:
  - Sparse Attention은 어텐션 행렬 연산만 일부 마스킹할 뿐 토큰 자체는 남아있어 모델 파라미터의 2/3를 차지하는 **FFN(Feed-Forward Network) 연산을 건너뛰지 못함**.
  - FastV는 토큰 자체를 지워버리므로 **Self-Attention과 FFN 연산 모두를 완전히 스킵**하여 진정한 연산량 절감을 달성함.

---

## 3. Findings & Quantitative Impact (주요 결과)
- **45% FLOPs 절감 & 무손실 성능**:
  - LLaVA-1.5-13B 및 Qwen-VL-Chat에서 $K=2, R=50\%$ 설정 시, VQA(A-OKVQA, MMMU), 캡셔닝(Flickr30K), 체화 추론(PCA-Bench), 세밀한 OCR(OCR-VQA) 등 전 분야에서 성능 저하 없이 **이론상 연산량(FLOPs) 45% 감축**.
- **레이턴시 역전 (13B가 7B보다 빨라짐)**:
  - 실제 Latency 측정 결과, **LLaVA-13B + FastV**가 **기본 LLaVA-7B보다 더 빠른 추론 속도**를 기록하면서도 13B 고유의 고성능을 완벽히 유지.
- **비디오 모델(Video-LLaVA)에서의 극대화**:
  - 프레임 토큰이 2,048개에 달하는 비디오 이해에서는 중복성이 더 심하므로, 40% 이상의 연산을 절감하면서도 오히려 노이즈 제거 효과로 인해 일부 벤치마크(TGIF 등)에서 성능이 소폭 상승.
- **고해상도 트레이드오프 해소**:
  - 고해상도 입력 시 발생하는 토큰 폭증 문제를 FastV로 상쇄하여, 동일한 연산 비용 예산 내에서 더 높은 해상도의 이미지를 처리해 세밀한 인식 성능 향상.

---

## 4. Limitations & Frontier Blind Spot (한계점 ➔ 후속 연구 기회)
- ⚠️ **가변 시퀀스 길이로 인한 실제 서빙 배치(Batching) 오버헤드**:
  - 중간 레이어에서 시퀀스 길이가 달라지므로, vLLM의 PagedAttention이나 TensorRT-LLM 환경에서 정적 텐서 형태를 깨뜨려 패딩(Padding) 오버헤드나 복잡한 메모리 재할당이 필요함. (실제 서빙 레벨의 커널 최적화 과제).
- ⚠️ **정적 하이퍼파라미터($K=2, R=50\%$)의 경직성**:
  - 이미지의 복잡도나 질문의 난이도(예: 빽빽한 영수증 OCR vs 단순 풍경 묘사)에 관계없이 무조건 2번째 레이어에서 50%를 날리므로, 극도로 미세한 단서가 필요한 태스크에서는 핵심 정보 소실 위험 존재. (➔ Query-Adaptive Dynamic Pruning 필요).
- ⚠️ **비디오 시공간 구조 붕괴 위험**:
  - 비디오에서 단순 전체 어텐션 평균으로 토큰을 날리면 특정 시간(Time) 프레임 전체의 시각 토큰이 소실되어 시간적 인과관계 추론이 망가질 수 있음.

---

## 5. Hyeonwoo's Research Vector (나의 연구 아이디어 연계)
- **발굴 벡터**: **[병목 역전 (Bottleneck Targeting)]** + **[이종 결합]**
- **1) FastV + StreamKV 결합 (시공간 2축 압축 온디바이스 아키텍처)**:
  - FastV의 **'레이어 축 토큰 프루닝(Depth-wise Token Discard)'**과 StreamKV의 **'시간 축 KV Cache 윈도우 압축(Time-wise Cache Eviction)'**을 결합.
  - 온디바이스 멀티모달 라이프로깅 환경([[ondevice_multimodal_lifelogging]])에서 초장시간 비디오 스트림을 처리할 때, 메모리 풋프린트를 80% 이상 감축하는 극한의 가속 프레임워크 설계.
- **2) Query-Guided Modality-Decoupled Dynamic Pruning**:
  - 질문의 의미적 복잡도에 따라 토큰 제거 시점 $K$와 비율 $R$을 동적으로 결정하는 모달리티 게이팅([[rq_modality_decoupled_moe]]) 연구로 확장.
