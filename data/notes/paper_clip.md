# 📄 [Paper] CLIP: Learning Transferable Visual Models From Natural Language Supervision
- **Authors**: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, et al. (OpenAI)
- **Venue / Year**: ICML 2021
- **Domain**: Multimodal / Vision-Language Representation
- **Connected**: [[contrastive_learning]], [[vit]], [[vision_encoder]], [[paper_llava]], [[rq_cross_modal_alignment]]

---

## 1. Problem Formulation & Motivation (왜 이 문제를 풀려고 했는가?)
- **Unaddressed Bottleneck**: 기존 컴퓨터 비전 모델은 ImageNet 1,000개 클래스와 같이 고정된 카테고리 라벨(Fixed Discrete Labels)에 국한되어, 새로운 개념이나 범주에 대한 Zero-shot 전이(Transfer)가 불가능했음.
- **Core Limitation of Prior Art**: 라벨링 비용이 매우 비싸고, 자연어에 담긴 풍부한 의미적 맥락(Semantic Context)을 전혀 활용하지 못함.

---

## 2. Core Hypothesis & Architecture (핵심 제안 기법)
- **핵심 가설**: 웹에서 수집한 4억 개(WIT, 400M)의 (이미지, 텍스트) 자연어 쌍을 대상으로 대규모 대조 학습(Contrastive Pre-training)을 수행하면, 별도의 미세조정(Fine-tuning) 없이도 텍스트 프롬프트를 통해 임의의 시각 개념을 Zero-shot으로 분류할 수 있다.
- **아키텍처**:
  - Image Encoder $f(\cdot)$: Vision Transformer(ViT) 또는 ResNet
  - Text Encoder $g(\cdot)$: Transformer 기반 텍스트 인코더
  - Multi-modal Embedding Space: 두 인코더의 출력을 동일한 차원 $d$로 선형 투영(Projection) 후 L2 정규화.
- **Objective Function (Symmetric InfoNCE)**:
  배치 크기 $N$에 대해 이미지-텍스트 유사도 행렬 $S_{i, j} = \cos(f(I_i), g(T_j)) / \tau$ 계산:
  $$\mathcal{L}_{\text{CLIP}} = \frac{1}{2} \left( \mathcal{L}_{\text{image}\to\text{text}} + \mathcal{L}_{\text{text}\to\text{image}} \right)$$

---

## 3. Findings & Quantitative Impact (주요 결과)
- ImageNet Zero-shot 분류에서 지도학습된 ResNet-50과 동등한 성능(76.2% Top-1 Accuracy) 달성.
- OCR, 지리적 위치 파악, 액션 인식 등 30개 이상의 다양한 비전 벤치마크에서 강력한 범용 전이 능력 입증.

---

## 4. Limitations & Frontier Blind Spots (한계점 ➔ 후속 연구 기회)
- ⚠️ **Coarse Alignment의 한계**: 이미지 전체와 텍스트 문장 전체를 하나의 글로벌 벡터로 매핑하므로, 객체의 세부 속성(색상, 재질, 수량, 공간적 상대 위치)을 구별하는 Fine-grained 추론 능력이 취약함 (e.g., "오른쪽의 빨간 사과와 왼쪽의 파란 컵" 구분 실패).
- ⚠️ **생성(Generation) 불가**: 오직 대조 평가 및 검색/분류에만 특화되어 있어, 텍스트나 이미지를 직접 생성하는 대화형 VLM으로 확장하기 위해 별도의 생성형 디코더(LLM) 연결 필요.

---

## 5. Hyeonwoo's Research Vector (나의 연구 아이디어 연계)
- **발굴 벡터**: [이종 결합] + [기저 가정 파괴]
- **후속 발전 방향**:
  1. CLIP ViT 인코더를 Frozen LLM과 Projection Layer로 결합하여 생성형 VLM을 구축하는 방향 $\rightarrow$ [[paper_llava]]
  2. 글로벌 벡터 매핑을 넘어 토큰/영역 수준의 정렬을 수행하는 연구 과제 $\rightarrow$ [[rq_cross_modal_alignment]]
