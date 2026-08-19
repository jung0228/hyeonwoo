# 🔭 [RQ] Multimodal Data Mixture Recipe Optimization
- **Researcher**: 정현우 (Jeong Hyeonwoo)
- **Domain**: Multimodal AI / Large-Scale Training / Data Engineering
- **Industry Context**: 네이버클라우드 HyperCLOVA X Omni팀 인턴십 연계
- **Connected**: [[data_recipe]], [[sft]], [[hcx_omni]], [[rq_cross_modal_alignment]]

---

## 1. Macro Why (거시적 당위성: 왜 이 문제를 풀어야 하는가?)
멀티모달 파운데이션 모델 학습에서 모델 아키텍처보다 더 결정적인 성능 차이를 낳는 것은 **"어떤 모달리티 데이터를 어떤 비율로 섞어 훈련시키는가(Data Mixture Ratio)"**입니다. 데이터 레시피의 원리를 규명하지 못한 채 수백억 원의 GPU 연산 비용을 주먹구구식 실험에 낭비하는 문제를 해결하고, 데이터 효율적인 최적의 믹스 법칙(Scaling Law for Data Mixtures)을 도출해야 합니다.

---

## 2. Prior Art Pathology & Frontier Blind Spot (기존 SOTA의 결함)
- **경험주의적 튜닝 한계**: 대부분의 프런티어 랩(OpenAI, Google, NAVER)은 경험적 시도와 에러(Trial and Error)로 토큰 비율을 결정하며, 왜 특정 단계에서 비디오 데이터 비율을 40% 이상으로 높여야 텍스트 추론 능력이 함께 상승하는지 이론적 규명이 부재함.
- **모달리티 간 간섭(Cross-Modal Interference)**: 특정 모달리티(예: 이미지 캡션)가 과다 투입되면 텍스트 수학/코딩 추론 성능이 망각(Catastrophic Forgetting)되는 현상 발생.

---

## 3. Hyeonwoo's Core Hypothesis & 4-Vector Strategy (핵심 가설 및 4대 발굴 벡터)
- **발굴 벡터**: **[이종 결합 (Cross-Pollination)]** + **[병목 역전 (Bottleneck Targeting)]**
- **핵심 가설**:
  1. *정보 이론적 결합*: 각 모달리티의 토큰 엔트로피와 그래디언트 다양성(Gradient Diversity)을 실시간 모니터링하여, 학습 단계별로 망각을 최소화하고 시너지(Cross-modal Synergy)를 극대화하는 **Dynamic Online Token Re-weighting** 메커니즘을 수립한다.
  2. *HCX SEED Omni 분석*: Stage 1(사전 정렬) $\to$ Stage 2(멀티모달 지시 튜닝) $\to$ Stage 3(고난도 추론 강화)의 3단계 전이 과정에서 모달리티별 최적 파레토 프런티어(Pareto Frontier)를 체계적 Ablation으로 증명.

---

## 4. Evaluation & Verification Plan (검증 파이프라인)
- **Benchmarks**: MME, MMBench, Video-ChatGPT, GSM8K (Text Reasoning 보존율 검증).
- **Success Metric**: 고정 레시피 대비 학습 토큰 수 30% 절감 상태에서 동등 SOTA 달성, Catastrophic Forgetting 0% 유지.
