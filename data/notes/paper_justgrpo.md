# JustGRPO: Simple & Scalable Post-Training RL for Reasoning

> **ICML 2026 Outstanding Paper & Highlight** 🌟  
> **저자**: DeepSeek, Tsinghua, Peking University  
> **키워드**: `Post-Training RL`, `GRPO (Group Relative Policy Optimization)`, `Reasoning Alignment`, `GSM8K / MATH-500`

---

## 💡 핵심 아이디어

기존 PPO(Proximal Policy Optimization)는 생성 정책(Policy) 외에 별도의 가치 평가 모델(Critic / Value Network)을 동시 학습시켜야 하므로 GPU 메모리 소모가 극심하고 수렴이 불안정했습니다.

**GRPO (Group Relative Policy Optimization)**는 Critic 모델을 완전히 제거하고, 단일 프롬프트에 대해 $G$개의 응답 그룹 샘플을 생성한 뒤 **그룹 내 상대적 보상 통계량(Group Advantage)**만을 계산하여 정책을 업데이트하는 획기적인 RLHF 강화학습 알고리즘입니다.

---

## 📐 수식 및 메커니즘

하나의 질문 $q$에 대해 $G$개의 답변 $\{o_1, o_2, \dots, o_G\}$을 생성하고 각 보상 $R_i$를 평가합니다:

$$A_i = \frac{R_i - \text{mean}(R)}{\text{std}(R)}$$

목적 함수(Objective Function):

$$J_{\text{GRPO}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \min \left( \frac{\pi_\theta(o_i|q)}{\pi_{\text{old}}(o_i|q)} A_i, \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\text{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right) A_i \right) - \beta D_{\text{KL}}(\pi_\theta || \pi_{\text{ref}}) \right]$$

---

## 🎯 주요 성과
- Critic 모델 제거로 **GPU 메모리 약 50% 절감** 및 학습 속도 2배 향상.
- 수학 추론 Benchmark (GSM8K 89.1%, MATH-500)에서 PPO 대비 월등한 성능 달성.

---

## 🔗 연결 개념
- [[paper_dllm]] (Diffusion Language Models & JustGRPO)
- [[rq_data_recipe_optimization]] (현우의 연구 과제: 데이터 레시피 및 RLHF)
