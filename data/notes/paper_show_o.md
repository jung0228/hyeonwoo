# 📄 [Paper] Show-o: One Single Transformer to Unify Multimodal Understanding and Generation
- **Authors**: Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhicheng Chen, Zhenheng Yang, Mike Zheng Shou (Show Lab, NUS)
- **Venue / Year**: ICLR 2025
- **Domain**: Unified Multimodal Foundation Model / Discrete Diffusion / Autoregressive Modeling
- **Connected**: [[paper_dynin_omni]], [[paper_emu3]], [[diffusion]], [[discrete_token]], [[rq_cross_modal_alignment]]

---

## 1. Problem Formulation & Frontier Blind Spot (문제 정의)
- **Unaddressed Bottleneck**: 기존 멀티모달 시스템은 시각 이해(Visual Understanding)를 위해 Autoregressive LLM을, 시각 생성(Visual Generation)을 위해 Diffusion Model(확산 모델)을 별도의 두 개 모델로 분리하여 사용하여 파라미터 낭비와 모달리티 간 표현 공간의 단절이 극심했음.
- **Core Limitation of Prior Art**: 단순 Next-Token 예측만으로는 연속적인 고화질 시각 정보를 효율적으로 생성하기 어렵고, 순수 Diffusion 모델은 복잡한 텍스트 논리 추론(Reasoning)에 취약함.

---

## 2. Core Architecture & Hybrid Objective (핵심 제안 기법)
- **핵심 가설**: 텍스트 토큰은 **Autoregressive (Causal Attention)**로 모델링하고, 시각 토큰은 **Discrete Denoising Diffusion (Full Bi-directional Attention)**으로 모델링하되, 이를 **단 하나의 Transformer 백본(Single Transformer)** 내부에서 통합할 수 있다.
- **Unified Attention Mechanism (Omni-Attention)**:
  - Text Tokens: Causal Masking (과거 토큰만 참조).
  - Image Tokens: Full-Attention Masking (모든 이미지 패치 토큰 간의 양방향 상관관계 계산).
  - Text-to-Image Cross Attention: 텍스트 조건 토큰을 이미지 디노이징 시 전면 참조.
- **수식 (하이브리드 손실 함수)**:
  $$\mathcal{L}_{\text{Show-o}} = \mathcal{L}_{\text{AR}}(\text{Text} \mid \text{Context}) + \mathcal{L}_{\text{Discrete-Diffusion}}(\text{Image}_0 \mid \text{Image}_t, \text{Text})$$

---

## 3. Findings & Lineage Impact (주요 결과)
- 단일 1.3B / 7B 파라미터 모델로 LLaVA 수준의 VQA 이해 능력과 SD v1.5 수준의 고화질 텍스트-투-이미지 생성을 완벽히 양립.
- **Dynin-Omni로의 발전 계보**: Show-o의 "텍스트 AR + 이미지 이산 디퓨전" 구조는 2026년 서울대 AIDAS 랩 **Dynin-Omni**가 텍스트, 음성, 비디오 전 모달리티를 포괄하는 **완전 단일화 마스크드 디퓨전(Unified Masked-Diffusion)**으로 도약하는 결정적 이론적 토대가 됨.
