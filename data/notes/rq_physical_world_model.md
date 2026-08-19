# 🔭 [RQ] Physics-Grounded 4D Omnimodal World Model for Robotics
- **Researcher**: 정현우 (Jeong Hyeonwoo)
- **Target Labs**: 서울대학교 AIDAS Lab (도재영 교수님) / 포스텍
- **Connected**: [[paper_cosmos]], [[paper_dynin_omni]], [[paper_virst]], [[paper_flow_matching]]

---

## 1. Macro Why (거시적 당위성: 왜 이 문제를 풀어야 하는가?)
2026년 현재 비디오 생성 모델은 영화 예고편 수준의 유려한 2D 영상을 만들지만, 로봇이 실제로 물체를 쥐고 옮길 때 질량, 마찰력, 강체 충돌 역학을 무시하여 컵을 깨뜨리거나 물체를 놓치는 치명적 결함을 보입니다. 단순 픽셀 생성을 넘어 실제 물리 법칙이 수학적으로 보존되는 **Physics-Grounded 4D World Model** 없이는 안전한 휴머노이드 로봇과 물리적 인공지능(Physical AI)은 불가능합니다.

---

## 2. Prior Art Pathology (기존 SOTA의 한계)
- **NVIDIA Cosmos / Sora 계열**: 수천억 개의 파라미터로 물리 현상을 '그럴듯하게 흉내(Visual Plausibility)' 낼 뿐, 물리적 불변량(Conservation Laws of Momentum & Energy)을 명시적으로 제약하지 않아 극한 상황에서 환각 발생.
- **Dynin-Omni / OpenVLA 계열**: 로봇 액션 생성 시 10~30스텝 디퓨전 지연으로 인해 50Hz 실시간 반사 제어(Reactive Control) 불가.

---

## 3. Hyeonwoo's Core Hypothesis & 4-Vector Strategy (핵심 가설)
- **발굴 벡터**: **[이종 결합 (Cross-Pollination)]** + **[병목 역전 (Bottleneck Targeting)]**
- **가설**:
  1. *물리 제약 결합*: 3D Gaussian Splatting 기반 공간 표현에 미분 가능한 물리 시뮬레이터(Differentiable Physics Engine)의 잔차 에너지 손실(Residual Energy Loss)을 주입.
  2. *초고속 1-Step 샘플링*: Flow Matching 직선 궤적에 Consistency Distillation을 적용하여, **20ms 이내에 50Hz 로봇 액션 토큰과 시각 예측을 동시 생성하는 온로봇 World Model** 구축.
