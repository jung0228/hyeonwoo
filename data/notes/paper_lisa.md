# 📄 [Paper] LISA: Reasoning Segmentation via Large Language Models
- **Authors**: Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, Jiaya Jia (HKUST, SmartMore)
- **Venue / Year**: CVPR 2024
- **Domain**: Multimodal / Reasoning Segmentation / Pixel-level Grounding
- **Connected**: [[paper_virst]], [[paper_llava]], [[paper_mevis]], [[rq_video_temporal_grounding]]

---

## 1. Problem Formulation & Motivation (왜 이 문제를 풀려고 했는가?)
- **Unaddressed Bottleneck**: 기존 시각 분할 모델(SAM, Mask R-CNN, Referring Segmentation)은 "빨간 사과", "왼쪽 강아지"처럼 명시적이고 단순한 지시어(Explicit Referring)에만 반응할 뿐, **"비타민 C가 풍부하고 껍질을 깎아 먹는 과일을 찾아 분할하라"**와 같은 복합 지식 추론 기반의 세그멘테이션(Reasoning Segmentation)을 수행하지 못함.
- **Core Limitation of Prior Art**: LLM의 고수준 인지 추론 능력과 컴퓨터 비전의 고해상도 픽셀 분할 능력 간의 인터페이스 부재.

---

## 2. Core Architecture & `[SEG]` Token Mechanism (핵심 기법)
- **`[SEG]` 토큰 임베딩**:
  - LLM의 어휘집(Vocabulary)에 특수 토큰 `[SEG]`를 추가.
  - LLM이 복합 질문에 대해 텍스트 답변을 생성하다가, 분할해야 할 타겟 객체가 등장하면 `[SEG]` 토큰을 출력하도록 유도.
- **SAM (Segment Anything Model) 결합**:
  - `[SEG]` 토큰의 최종 레이어 은닉 상태 벡터 $\mathbf{h}_{\text{seg}}$를 추출하여 프로젝션 계층(MLP)을 통과.
  - 이를 SAM의 프롬프트 인코더(Prompt Encoder)에 마스크 프롬프트로 주입하여 고해상도 바이너리 세그멘테이션 마스크 $\mathbf{M}$을 출력.
- **수식 (End-to-End Joint Loss)**:
  $$\mathcal{L}_{\text{LISA}} = \mathcal{L}_{\text{text\_autoregressive}} + \lambda_{\text{dice}} \mathcal{L}_{\text{Dice}}(\mathbf{M}, \mathbf{M}^{\text{gt}}) + \lambda_{\text{bce}} \mathcal{L}_{\text{BCE}}(\mathbf{M}, \mathbf{M}^{\text{gt}})$$

---

## 3. Findings & Lineage Impact
- 복합 추론 세그멘테이션(Reasoning Segmentation Benchmark)에서 기존 SOTA 대비 mIoU 20%p 이상 압도적 성능 달성.
- **VIRST로의 발전 계보**: LISA의 단일 이미지 `[SEG]` 토큰 개념은 2026년 서울대 AIDAS 랩 **VIRST**로 계승되어, **동적 비디오 시공간 3D 마스크(SpatioTemporal RVOS) 및 음향(Audio Event) 쿼리 기반 픽셀 추적**으로 완벽하게 승격됨.
