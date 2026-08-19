# 🔭 [RQ] Counterfactual Spatiotemporal Causal Reasoning in Long Videos
- **Researcher**: 정현우 (Jeong Hyeonwoo)
- **Target Labs**: 서울대학교 AIDAS Lab (도재영 교수님) / KAIST DAVIAN 랩
- **Connected**: [[paper_virst]], [[paper_mevis]], [[long_video_understanding]], [[paper_momentseeker]]

---

## 1. Macro Why (거시적 당위성)
자율주행 사고 원인 분석, 의료 수술 영상 감사, 스포츠 전략 분석 등 고위험 의사결정에서 AI는 단순히 "무슨 일이 일어났는가"뿐만 아니라 **"만약 A 행동을 하지 않았다면 어떤 결과가 발생했을까? (What if?)"**라는 반사실적(Counterfactual) 인과 추론을 수행할 수 있어야 합니다.

---

## 2. Prior Art Pathology (기존 SOTA의 한계)
- 현재의 모든 비디오 VLM(VIRST, Qwen2-VL, Gemini 1.5)은 시간을 1차원/3차원 텐서의 정적 시퀀스로 취급하며, 통계적 상관관계(Correlation)에 기반해 다음 프레임을 예측합니다.
- 시간 순서를 역전시키거나 중간 프레임에 가상의 개입(Intervention)을 가했을 때, 인과 화살표(Arrow of Time)를 위배하고 그럴듯한 환각(Hallucination)을 생성하는 심각한 한계를 가집니다.

---

## 3. Hyeonwoo's Core Hypothesis & 4-Vector Strategy (핵심 가설)
- **발굴 벡터**: **[기저 가정 파괴 (Assumption Inversion)]** + **[실패 모드 군집화]**
- **가설**:
  1. *구조적 인과 모델(SCM) 주입*: 펄(Judea Pearl)의 $do(\cdot)$ 연산자를 멀티모달 RoPE 어텐션 마스크에 매핑하여, 특정 이벤트 토큰의 인과적 개입(Intervention) 시 시공간 피처 맵의 반사실적 분기를 수학적으로 추론.
  2. *VIRST 3D 세그멘테이션 결합*: 1차원 타임스탬프 수준의 VMR을 넘어, 개입에 따른 3차원 픽셀 마스크의 시공간 변화 궤적을 오차 없이 분할·추적.
