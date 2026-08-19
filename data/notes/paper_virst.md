# 📄 [Paper] VIRST: Video-Instructed Reasoning Assistant for SpatioTemporal Segmentation
- **Authors & Lab**: 서울대학교 AIDAS Lab (지도교수: 도재영 교수님)
- **Venue / Year**: CVPR 2026 (Oral Presentation)
- **Domain**: Spatiotemporal Video Reasoning / Referring Video Object Segmentation (RVOS) / Embodied AI
- **Connected**: [[paper_dynin_omni]], [[long_video_understanding]], [[paper_momentseeker]], [[rq_video_temporal_grounding]]

---

## 1. Problem Formulation & Frontier Blind Spot (문제 정의)
- **Unaddressed Bottleneck**: 기존의 비디오 LLM(Video-ChatGPT, LLaVA-Video, HCX SEED Omni)은 영상 속 사건을 텍스트로만 묘사할 뿐, *"사용자가 지목한 특정 객체가 영상의 어느 시간(When), 어느 픽셀 영역(Where)에서 움직이고 있는가"*를 정밀하게 분할(Segmentation) 및 추적하지 못함.
- **Core Limitation of Prior Art**: 단순 타임스탬프 예측(Moment Retrieval)을 넘어선 픽셀 단위의 시공간 의미론적 분할(RVOS)과 다단계 인과 추론의 결합 부재.

---

## 2. Core Architecture & Mechanisms (핵심 기술 메커니즘)
- **Semantic-to-Segmentation Representation Bridge**:
  - LLM의 고수준 의미적 추론 토큰(Reasoning Embeddings)과 고해상도 시공간 피처 맵(Spatiotemporal Feature Maps)을 직접 정렬하는 경량 브리지 레이어 도입.
- **VIRST-Audio 확장**:
  - 자연어 텍스트뿐만 아니라 **음향 소리(Audio Events, e.g. "개 짖는 소리가 나는 위치의 동물")**를 쿼리로 받아 영상 내 객체를 시공간 픽셀 단위로 즉각 추적.
- **수식 (SpatioTemporal Mask Loss)**:
  $$\mathcal{L}_{\text{VIRST}} = \lambda_{\text{dice}} \mathcal{L}_{\text{Dice}}(\mathbf{M}_t, \mathbf{M}_t^{\text{gt}}) + \lambda_{\text{bce}} \mathcal{L}_{\text{BCE}}(\mathbf{M}_t, \mathbf{M}_t^{\text{gt}}) + \mathcal{L}_{\text{text\_QA}}$$

---

## 3. Findings & Quantitative Impact (주요 결과)
- **CVPR 2026 Oral 선정**: 세계 최고 권위 컴퓨터 비전 학회에서 구두 발표(상위 3% 이내)로 학술적 우수성 입증.
- Ref-YouTube-VOS, MeViS 등 최고난도 비디오 객체 분할 벤치마크에서 SOTA 갱신.
- **PVUW MeViS-Audio Challenge 3위 수상**: 음향-영상 복합 세그멘테이션 세계 대회 입증.

---

## 4. Hyeonwoo's Research Vector (KAIST DAVIAN 랩 VMR 경험과의 시너지)
- **연구 연결점**:
  - 현우 님의 KAIST DAVIAN 랩 1저자 경험인 **Video Moment Retrieval (VMR)** 평가 파이프라인과 도재영 교수님의 **VIRST 시공간 픽셀 세그멘테이션**은 완벽한 상호보완 관계!
  - 1차원 시간 구간 예측(VMR)을 3차원 시공간 픽셀 마스크(VIRST)로 승격시키는 후속 연구 제안 가능.
