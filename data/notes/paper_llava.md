# 📄 [Paper] LLaVA: Visual Instruction Tuning
- **Authors**: Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee (UW-Madison, Microsoft Research, Columbia Univ.)
- **Venue / Year**: NeurIPS 2023 (Oral)
- **Domain**: Multimodal / Vision-Language Model (VLM)
- **Connected**: [[clip]], [[llm]], [[sft]], [[vision_encoder]], [[long_video_understanding]], [[rq_cross_modal_alignment]]

---

## 1. Problem Formulation & Frontier Blind Spot (문제 정의 및 선행 연구 결함)
- **Unaddressed Bottleneck**: GPT-4와 같은 LLM은 텍스트 영역에서 뛰어난 명령 이행(Instruction Following) 능력을 보였으나, 시각 정보를 대화형으로 해석하고 사용자의 복합 지시를 수행하는 범용 멀티모달 인터페이스는 부재했음.
- **Core Limitation of Prior Art**: 기존 멀티모달 모델(Flamingo, BLIP-2)은 복잡한 Q&A나 다단계 시각 추론 데이터를 체계적으로 학습하지 못해, 단순 이미지 캡셔닝 수준에 머물렀음.

---

## 2. Core Hypothesis & Architecture (핵심 기법)
- **핵심 가설**: GPT-4를 활용해 이미지-텍스트 쌍으로부터 고품질 '시각 지시 데이터(Visual Instruction Dataset)' 158K개를 자동 합성하고, 단순한 선형 투영 계층(Linear Projection Layer) 하나만으로 CLIP 비전 인코더와 LLM(Vicuna/Llama)을 연결해도 강력한 멀티모달 대화 능력이 창발한다.
- **2단계 학습 레시피 (Two-Stage Training Recipe)**:
  1. **Stage 1: Feature Alignment Pre-training**:
     - CC-595K 필터링 이미지-텍스트 캡션 데이터 사용.
     - Vision Encoder와 LLM을 모두 동결(Frozen)하고, 프로젝션 행렬 $\mathbf{W}$만 학습하여 시각 토큰을 LLM 임베딩 공간으로 정렬.
  2. **Stage 2: Visual Instruction Tuning (SFT)**:
     - 158K 시각 지시 데이터(대화, 상세 묘사, 복합 추론)로 학습.
     - Vision Encoder는 동결 유지, Projection Matrix $\mathbf{W}$와 LLM 가중치를 End-to-End로 파인튜닝.
- **수식 (Projection)**:
  $$\mathbf{H}_v = \mathbf{W} \cdot \mathbf{Z}_v \quad (\mathbf{Z}_v = \text{CLIP-ViT}(I))$$
  $$\mathbf{X} = [\mathbf{H}_v; \mathbf{H}_q] \longrightarrow \text{LLM}(\mathbf{X}) \longrightarrow \mathbf{Y}_{\text{response}}$$

---

## 3. Findings & Quantitative Impact (주요 결과)
- LLaVA-Bench(In-the-Wild)에서 독보적인 멀티모달 대화 성능 달성 (GPT-4 대비 85.1% 상대 점수 기록).
- 단 1개의 단순 Linear Projection 레이어만으로도 복잡한 Perceiver Resampler나 Q-Former보다 뛰어난 효율성과 성능 입증.

---

## 4. Limitations & Failure Modes (한계점 ➔ 후속 연구 기회)
- ⚠️ **저해상도 정보 손실**: 고정 해상도($224 \times 224$ or $336 \times 336$)로 이미지를 축소하여 입력하므로, 작은 텍스트(OCR)나 미세 객체 인식 실패. (➔ LLaVA-NeXT의 AnyRes 그리드 분할로 발전)
- ⚠️ **정적 단일 이미지 한계**: 프레임 시퀀스와 오디오가 포함된 비디오/스트리밍 환경에 직접 적용 불가. (➔ Video-LLaVA 및 LVMR로 확장 필요)

---

## 5. Hyeonwoo's Research Vector (나의 연구 아이디어 연계)
- **발굴 벡터**: [이종 결합] + [병목 역전]
- **후속 확장**: LLaVA의 시각 지시 튜닝 패러다임을 비디오 프레임 시퀀스와 KV Cache 스트리밍 최적화로 확장 $\rightarrow$ [[long_video_understanding]], [[streamkv]], [[rq_video_temporal_grounding]]
