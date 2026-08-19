# 📄 [Paper/Model] NVIDIA Cosmos: Physical AI & World Foundation Models
- **Authors & Org**: NVIDIA Research & Physical AI Team
- **Year**: 2026
- **Domain**: Physical AI / Omnimodal World Foundation Models / Robotics Simulation
- **Connected**: [[paper_dynin_omni]], [[paper_virst]], [[rq_physical_world_model]], [[diffusion]]

---

## 1. Problem Formulation & Macro Why (왜 이 연구가 시작되었는가?)
- **Unaddressed Bottleneck**: 기존 텍스트/이미지 옴니 모델은 모니터 안의 2D 픽셀 생성에 갇혀 있어, 실제 물리적 세계(중력, 마찰, 질량, 충돌)와 상호작용해야 하는 자율주행, 휴머노이드 로봇, 공장 자동화의 두뇌 역할을 수행할 수 없음.
- **Core Limitation of Prior Art**: 비디오 생성 시 물리 법칙이 위배되거나(물체 관통, 비정상적 중력 왜곡) 로봇의 모터 토크 액션과 동기화되지 않음.

---

## 2. Core Architecture & World Modeling (핵심 메커니즘)
- **Omnimodal Physics World Model**:
  - 시각, 음향, 텍스트뿐만 아니라 **3D 깊이(Depth), 포인트 클라우드, 로봇 제어 액션(End-effector Action Trajectory)**을 단일 연속-이산 잠재 공간에서 모델링.
  - 시간적 인과성(Temporal Causality)과 물리적 제약 조건(Physics Constraints)을 강화학습 환경 시뮬레이터와 직접 연동.
- **Flow-Matching Video-World Tokenizer**:
  - $1024 \times 1024$ 해상도의 고화질 물리 시뮬레이션 비디오를 초고속 Flow Matching으로 렌더링.

---

## 3. Limitations & Hyeonwoo's Research Takeaway (남겨진 한계 ➔ 후속 연구 기회)
- ⚠️ **초거대 연산 비용 및 폐쇄성**: 수만 장의 GPU 클러스터를 요구하며, 온디바이스 로봇 엣지 디바이스(RTX Embedded)에서의 50Hz 실시간 추론 불가.
- ⚠️ **접촉 역학(Contact Dynamics)의 미세 오차**: 유리잔을 쥐거나 미끄러운 물체를 잡는 미세 마찰력 예측에서 환각 잔존.
- **후속 연구 가설**: 서울대 AIDAS 랩 **Dynin-Robotics** 및 VIRST 시공간 세그멘테이션과 결합하여, **경량 온로봇 1-Step Flow-Matching World Model** 도출 $\rightarrow$ [[rq_physical_world_model]].
