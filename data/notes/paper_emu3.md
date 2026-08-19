# 📄 [Paper] Emu3: Next-Token Prediction is All You Need
- **Authors**: BAAI (Beijing Academy of Artificial Intelligence) Emu3 Team
- **Venue / Year**: ArXiv / ICLR 2025
- **Domain**: Multimodal / Any-to-Any Foundation Model / Autoregressive Generation
- **Connected**: [[hcx_omni]], [[diffusion]], [[discrete_token]], [[vision_encoder]], [[rq_cross_modal_alignment]]

---

## 1. Problem Formulation & Frontier Blind Spot (문제 정의 및 기존 모델의 결함)
- **Unaddressed Bottleneck**: 멀티모달 모델들은 전통적으로 "이해(Understanding)는 LLM/ViT", "생성(Generation)은 Diffusion Model(확산 모델)"이라는 분리된 이원화 아키텍처(Hybrid Dual Architecture)를 취해옴 (HCX SEED Omni 역시 이해용 Continuous Encoder와 생성용 Discrete VQ를 분리 운용).
- **Core Limitation of Prior Art**: 구조의 복잡성, 모달리티 간 표현 공간의 불일치, 학습 파이프라인의 다단계 파편화(8단계 이상) 발생.

---

## 2. Core Hypothesis & Architecture (핵심 제안 기법)
- **핵심 가설**: 확산 모델(Diffusion)이나 별도의 비전 인코더(CLIP) 없이, 텍스트, 이미지, 비디오를 모두 이산 토큰(Discrete Tokens)으로 변환한 뒤 **오직 단 하나의 Transformer와 Next-Token Prediction 손실 함수만으로** 사전학습해도 SOTA 수준의 멀티모달 이해와 고화질 이미지/비디오 생성을 동시에 달성할 수 있다.
- **Unified Objective**:
  $$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t \mid x_{<t}; \Theta)$$
  - $x_t \in \mathcal{V}_{\text{text}} \cup \mathcal{V}_{\text{vision}}$ (단일 어휘 공간)
- **Vision Tokenizer**:
  - $512 \times 512$ 또는 $1024 \times 1024$ 이미지를 $4096$ 토큰으로 이산화하는 고성능 SBER-MoVQ 코덱.
  - 시간 축 압축을 적용하여 비디오 토큰을 동일한 어휘 공간에서 처리.

---

## 3. Findings & Quantitative Impact (주요 결과)
- **이미지 생성**: SDXL(Stable Diffusion XL) 및 Flux를 능가하는 시각적 퀄리티 달성 (GenEval 점수 0.66 vs SDXL 0.55).
- **시각적 이해**: LLaVA-1.6 및 Qwen-VL과 동등 수준의 VQA/MMBench 벤치마크 점수 기록.
- 단 하나의 손실 함수로 Any-to-Any 패러다임의 극단적 단순화와 확장성(Scaling Laws) 입증.

---

## 4. Limitations & Frontier Blind Spot (한계점 ➔ 후속 연구 기회)
- ⚠️ **연산 비용과 시퀀스 길이**: 이미지를 수천 개의 토큰으로 펼쳐 Autoregressive하게 생성하므로, 추론 시 KV Cache 메모리와 생성 속도(Latency)가 확산 모델 대비 느림.
- ⚠️ **세부 해상도 텍스트(OCR) 세밀도**: 순수 이산 토큰화 과정에서 미세 폰트나 초소형 객체 정보의 손실이 일부 잔존.

---

## 5. Hyeonwoo's Research Vector (HCX SEED Omni 개선 연구 방향)
- **발굴 벡터**: **[기저 가정 파괴 (Assumption Inversion)]**
- **HCX 개선 아이디어**: HCX의 복잡한 8단계 파이프라인(Stage 2 이산 토큰 + Stage 5 연속 인코더의 이중 구조)을 Emu3 스타일의 **Unified Next-Token Prediction 3단계 단일화 파이프라인**으로 압축하여 훈련 안정성 및 Any-to-Any 생성 해상도 극대화.
