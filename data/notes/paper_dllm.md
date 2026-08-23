# Diffusion LLM (dLLM) & The Flexibility Trap

> **ICML 2026 Outstanding Paper Award** 🏆  
> **논문명**: *The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models*  
> **키워드**: `dLLM`, `Non-Autoregressive Diffusion`, `Flexibility Trap`, `Parallel Decoding`

---

## 💡 핵심 아이디어

디퓨전 언어 모델(dLLM)은 기존 Autoregressive 모델(좌$\to$우 순차 생성)과 달리 토큰을 임의의 순서(Arbitrary Order)로 병렬 생성할 수 있는 유연성을 제공합니다.

하지만 본 논문은 **"The Flexibility Trap (유연성의 함정)"** 현상을 최초로 규명했습니다:
dLLM이 임의 순서 생성의 자유도를 오용하여 `"Therefore"`, `"Since"`, `"Thus"` 같은 높은 불확실성을 가진 **핵심 분기 토큰(Forking Tokens)** 생성을 회피하고 쉬운 토큰부터 먼저 채워버림으로써, 논리적 추론 솔루션 공간이 조기에 붕괴되는 현상입니다.

---

## 🛠️ 해결책: JustGRPO Autoregressive Scaffold

- **RL Rollout 시 좌$	o$우 제약**: RLHF 강화학습(JustGRPO) 과정에서는 정책 탐색 시 Autoregressive 순서를 강제하여 논리 분기 토큰을 정면으로 학습하게 만듭니다.
- **추론 시 병렬성 유지**: 학습 스캐폴딩 후 실제 추론 시에는 디퓨전 고유의 빠른 병렬 데코딩(Parallel Decoding) 성능을 100% 유지합니다.

---

## 🔗 연결 개념
- [[paper_justgrpo]] (JustGRPO RL Framework)
- [[rq_modality_decoupled_moe]] (모달리티 분리형 라우팅)
