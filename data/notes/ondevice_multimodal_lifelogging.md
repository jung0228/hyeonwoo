# 온디바이스 멀티모달 라이프로깅 (On-Device Multimodal Life-Logging)

## 핵심 아이디어
스마트 안경 및 웨어러블 폼팩터를 통해 사용자의 1인칭 시야(First-Person View) 영상, 주변 음성, 공간 맥락을 24시간 실시간으로 흡수(Life-Logging)하고, 네트워크 지연과 프라이버시 침해 없이 초저지연 온디바이스(On-device) 신경망으로 연산하여 적시에 능동적 개입(Real-Time Contextual Nudge)을 수행하는 차세대 지능 인터페이스입니다.

---

## 핵심 파이프라인 수식화

$$\mathcal{S}_{\text{life}} = \{ \mathbf{v}_t \in \mathbb{R}^{H \times W \times C}, \mathbf{a}_t \in \mathbb{R}^{L}, \mathbf{c}_t \}_{\text{continuous}}$$

1. **상시 엣지 인지 (Always-on Edge Perception)**:
   $$\mathbf{z}_t = \text{Encoder}_{\text{On-Device}}(\mathbf{v}_t, \mathbf{a}_t, \mathbf{c}_t)$$
2. **이상 징후 및 개입 감지 (Intervention Triggering)**:
   $$\mathcal{T}_{\text{nudge}} = \mathbb{I}\left( \text{KL}(P(\text{Intention} | \mathbf{z}_{\le t}) \parallel Q(\text{Action} | \text{Context})) > \tau \right)$$
3. **무마찰 넛지 전달 (Frictionless Nudge)**:
   - 시야(HUD) 또는 골전도 오디오로 0.1초 이내에 결정적 팩트/경고 전송

---

## 직관적 설명
스마트폰을 꺼내 잠금을 풀고 검색창에 타이핑하는 물리적 단절을 완전히 제거합니다. 내가 세상을 바라보고 숨 쉬는 일상 자체가 AI의 입력이 되며, AI가 지루한 기억과 검색 노동을 100% 흡수함으로써 인간은 '오직 순수한 사고와 거시적 판단의 유희(Intellectual Play)'만을 누리게 됩니다.

---

## 연결 개념
- [[video_llm]] : 비디오 스트림 실시간 의미 분석 및 Video Moment Retrieval
- [[agent_memory]] : 과거 대화 및 시각 경험의 장기 온톨로지 축적
- [[multimodal]] : 시각-음성-텍스트 옴니모달 융합
- [[systems]] : 초저지연 온디바이스 엣지 NPU 및 에너지 최적화

---

## 참고
- OpenGlass / Ego4D: First-Person Multimodal Life-Logging Benchmark
- Project Astra / Meta Orion: Real-time Contextual Spatial Computing
- 정현우 칼럼: AI에게 "인간의 개입을 줄이고, 너의 개입을 늘려라"고 말할 수 있을까 (2026)
