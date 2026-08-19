# 📄 [Paper] Qwen2-VL: To See the World More Clearly
- **Authors**: Qwen Team (Alibaba Cloud)
- **Venue / Year**: ArXiv 2024-2025
- **Domain**: Multimodal / Vision-Language Model / Dynamic Resolution & Video Understanding
- **Connected**: [[hcx_omni]], [[long_video_understanding]], [[vit]], [[streamkv]], [[rq_video_temporal_grounding]]

---

## 1. Problem Formulation & Frontier Blind Spot (문제 정의 및 기존 모델의 결함)
- **Unaddressed Bottleneck**: 대부분의 VLM(LLaVA, HCX SEED Omni)은 고정 해상도($336 \times 336$ or $448 \times 448$) 또는 단순 고정 그리드 분할을 사용하여 왜곡(Distortion)과 과도한 연산 낭비가 발생하고, 20분 이상의 장시간 비디오에서 시공간 타임스탬프의 정밀한 동기화가 불가능했음.
- **Core Limitation of Prior Art**: 1D 텍스트 RoPE(Rotary Position Embedding)를 비디오 프레임에 그대로 적용하여 시간축과 공간축의 기하학적 상관관계가 왜곡됨.

---

## 2. Core Hypothesis & Architecture (핵심 제안 기법)
- **핵심 가설**: 이미지 원본의 종횡비와 해상도를 있는 그대로 보존하는 **Naive Dynamic Resolution** 메커니즘과, 시간(Time)·높이(Height)·너비(Width)를 3차원으로 분해해 회전 위치 임베딩을 부여하는 **Multimodal Rotary Position Embedding (M-RoPE)**를 도입하면 수 시간 분량의 비디오와 초고화질 이미지를 무손실 정밀 추론할 수 있다.
- **Multimodal RoPE (M-RoPE)**:
  헤드 차원 $D$를 텍스트($D/4$), 시간($D/4$), 높이($D/4$), 너비($D/4$)로 분할:
  $$\mathbf{R}_{\text{M-RoPE}}(t, h, w) = \text{diag}\left( R(t), R(h), R(w), R(\text{text}) \right)$$
- **Dynamic Resolution ViT**:
  - 임의 해상도의 이미지/비디오를 패치 크기에 맞춰 동적 2D/3D 토큰 그리드로 변환.
  - 패치 윈도우 어텐션으로 $O(N^2)$ 연산량 폭증 방지.

---

## 3. Findings & Quantitative Impact (주요 결과)
- Open-source VLM 중 압도적 1위 달성 (MathVista, DocVQA, Video-MME 등 전 벤치마크에서 GPT-4o 수준에 육박).
- 20분 이상의 비디오에서 초 단위의 세밀한 이벤트 탐색(Video Grounding) 및 초고해상도 OCR 완벽 수행.

---

## 4. Limitations & Frontier Blind Spot (한계점 ➔ 후속 연구 기회)
- ⚠️ **동시 음향(Audio) 모달리티 부재**: Qwen2-VL은 시각-언어 전용이며, 오디오는 별도의 Qwen2-Audio로 분리되어 있어 Any-to-Any 동시 융합 모델이 아님.
- ⚠️ **초장시간 스트리밍 시 KV Cache 제약**: 비디오 길이가 1시간을 넘어가면 M-RoPE에도 불구하고 KV Cache 메모리 병목 발생 (➔ StreamKV 결합 필요).

---

## 5. Hyeonwoo's Research Vector (HCX SEED Omni 개선 연구 방향)
- **발굴 벡터**: **[병목 역전 (Bottleneck Targeting)]** + **[이종 결합]**
- **HCX 개선 아이디어**: HCX의 고정 32K 컨텍스트 및 단순 프레임 다운샘플링 구조에 Qwen2-VL의 **M-RoPE (Time-Height-Width 3D 분해)**와 **Dynamic Resolution ViT**를 결합하여 영상-음성 타임스탬프 동기화 및 초장시간 비디오 인과 추론 능력 획기적 제고.
