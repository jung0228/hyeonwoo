# 📄 [Paper/Model] Dynin-Omni: Masked-Diffusion Unified Omnimodal Foundation Model
- **Authors & Lab**: 서울대학교 AIDAS Lab (지도교수: 도재영 교수님)
- **Year**: 2026
- **Domain**: Unified Omnimodal Foundation Model / Any-to-Any / Embodied Physical AI
- **Connected**: [[hcx_omni]], [[paper_virst]], [[paper_emu3]], [[diffusion]], [[rq_cross_modal_alignment]], [[rq_video_temporal_grounding]]

---

## 1. Problem Formulation & HCX SEED Omni 대비 핵심 차별점 (Why Dynin-Omni?)
- **HCX SEED Omni (2024)의 한계**:
  - 이해(Continuous Feature)와 생성(Discrete VQ-VAE)의 분리로 인한 8단계 복잡 파이프라인.
  - 인과적 Autoregressive 생성으로 인한 양방향 문맥(Bidirectional Context) 활용 제한 및 음성/시각 생성의 순차적 지연.
- **Dynin-Omni (2026)의 혁신**:
  - **단일 마스크드 디퓨전(Masked-Diffusion) 아키텍처**: 텍스트, 이미지, 음성(Speech), 비디오(Video)를 모두 포괄하는 **공유 이산 토큰 공간(Shared Discrete Token Space)** 구축.
  - 단 하나의 통합 아키텍처 내에서 전 모달리티의 동시적 양방향 이해(Understanding)와 비동기 고화질 생성(Generation)을 단일화.
  - 19개 멀티모달 벤치마크에서 글로벌 오픈 SOTA 달성.

---

## 2. Core Architecture & Mechanisms (핵심 기술 메커니즘)
- **Unified Masked-Diffusion Objective**:
  임의의 모달리티 $m \in \{\text{Text}, \text{Image}, \text{Audio}, \text{Video}\}$에 대해 마스킹된 토큰 집합을 점진적 디퓨전 디노이징으로 복원:
  $$\mathcal{L}_{\text{Dynin}} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{\epsilon}} \left[ \|\mathbf{\epsilon} - \mathbf{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}_{\text{omni}})\|^2 \right]$$
- **Any-to-Any Cross-Modal Flow**:
  - 비디오와 음성의 시간적 궤적을 동일한 잠재 공간(Latent Flow)에서 동기화하여 HCX의 1Hz 압축 손실을 원천 차단.
  - **Dynin-Robotics(물리적 AI)**로의 확장을 위해 센서 및 액션 토큰까지 통합 가능한 인터페이스 설계.

---

## 3. Findings & Comparative Impact
- 한국어-영어 멀티모달 및 음성-영상 동시 생성에서 기존 Autoregressive 모델 대비 **추론 속도 3.2배 향상 및 크로스모달 일관성 극대화**.
- 로보틱스 및 의료(MEDIC-AD) 등 피지컬 AI 도메인으로의 무손실 전이 능력 입증.

---

## 4. Hyeonwoo's Research Takeaway (도재영 교수님 연구실 컨택 & 연구 연계 포인트)
- **연구 어필 포인트**:
  - *"HCX SEED Omni 인턴십에서 분석한 8단계 SFT 파이프라인의 전이 손실과 MambaMia 1Hz 압축 한계를, 도재영 교수님의 Dynin-Omni 마스크드 디퓨전 공유 토큰 공간 및 VIRST 시공간 세그멘테이션 연구와 결합하여 극복하고 싶습니다."*
