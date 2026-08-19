# 📄 [Paper] Mini-Omni 2: Towards Real-Time Omni-modal Interactive LLMs
- **Authors**: Zhifei Xie, Changqiao Wu, et al.
- **Venue / Year**: ArXiv / ICLR 2025
- **Domain**: Multimodal / Real-Time Omni-modal Interactive LLM (Vision + Audio + Text)
- **Connected**: [[hcx_omni]], [[paper_moshi]], [[paper_qwen2_vl]], [[rq_video_temporal_grounding]]

---

## 1. Problem Formulation & Frontier Blind Spot (문제 정의 및 기존 모델의 결함)
- **Unaddressed Bottleneck**: GPT-4o가 실시간 비디오+음성 동시 대화의 비전을 제시했으나, 오픈소스 진영에서는 카메라 스트림(Vision)과 마이크 스트림(Audio)을 동시에 실시간으로 수신하며 저지연 음성으로 대답하는 완전한 오픈 엔드투엔드 모델이 전무했음.
- **Core Limitation of Prior Art**: 비전 모델(VLM)과 음성 대화 모델(Spoken Dialogue)이 각각 분리되어 실시간 멀티모달 상호작용의 시너지가 부재함.

---

## 2. Core Hypothesis & Architecture (핵심 기법)
- **핵심 가설**: 오디오 토큰 예측과 텍스트 토큰 예측을 직렬/병렬로 결합한 **Any-to-Any Task-Oriented SFT**와 **Cross-Modal Interruption Mechanism**을 구축하면, 소형 모델(0.5B~7B)에서도 실시간 시각+청각 동시 반응 및 인터럽트 처리가 가능하다.
- **아키텍처**:
  - Speech Encoder: Whisper / Qwen2-Audio
  - Vision Encoder: CLIP / SigLIP
  - Speech Decoder: SNAC / Mimi 기반의 병렬 멀티코드북 오디오 디코더
  - State Machine: Listening $\leftrightarrow$ Thinking $\leftrightarrow$ Speaking $\leftrightarrow$ Interrupted 상태 전이 관리.

---

## 3. Findings & Quantitative Impact (주요 결과)
- 최초의 엔드투엔드 오픈소스 Real-Time Vision+Audio Interactive Omni 모델 구현.
- 실시간 카메라 영상(30fps)을 보며 사용자의 음성 질문에 sub-300ms 지연 시간으로 즉각 음성 답변.

---

## 4. Limitations & Frontier Blind Spot (한계점 ➔ 후속 연구 기회)
- ⚠️ **스케일 및 장기 컨텍스트 한계**: 경량 0.5B/7B 베이스로, 복잡한 비디오 추론이나 다단계 계획 수립 능력은 상대적으로 제한적임.
- ⚠️ **고해상도 비디오 프레임 메모리 관리**: 초당 여러 장의 프레임이 유입될 때 KV Cache 관리가 미흡하여 5분 이상 연속 대화 시 메모리 과부하.

---

## 5. Hyeonwoo's Research Vector (HCX SEED Omni 개선 연구 방향)
- **발굴 벡터**: **[실패 모드 군집화]** + **[이종 결합]**
- **HCX 개선 아이디어**: HCX SEED Omni의 강력한 8B 파운데이션 지식에 Mini-Omni 2의 **Real-Time Cross-Modal Interruption State Machine**을 결합하여, 네이버 하이퍼클로바 X의 실제 서비스형 실시간 음성-영상 비서 시스템 구축.
