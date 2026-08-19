# 🔭 [RQ] Fine-grained Cross-Modal Alignment
- **Researcher**: 정현우 (Jeong Hyeonwoo)
- **Domain**: Multimodal Representation / Vision-Language-Audio
- **Target Labs**: 포스텍 손진희 교수님 연구실 / 서울대 AIDAS 도재영 교수님 연구실
- **Connected**: [[contrastive_learning]], [[clip]], [[paper_llava]], [[rq_data_recipe_optimization]]

---

## 1. Macro Why (거시적 당위성: 왜 이 문제를 풀어야 하는가?)
인간의 사고는 시각, 언어, 음향의 미세한 속성을 동시에 정밀하게 지각하지만, 현재의 멀티모달 AI는 전체 이미지와 전체 문장을 뭉뚱그려 대조하는 Coarse Alignment에 머물러 있습니다. 의료 영상의 미세 병변 지목, 자율주행의 돌발 객체 식별, 복합 다이어그램 추론 등 정밀 인지 작업에서 환각(Hallucination) 없이 안전하게 작동하는 차세대 멀티모달 지능을 구현하기 위해 반드시 해결해야 할 핵심 병목입니다.

---

## 2. Prior Art Pathology & Frontier Blind Spot (기존 SOTA의 결함)
- **CLIP 계열**: 이미지 전역 벡터(Global Pooling)와 문장 전역 벡터 간 코사인 유사도만 최대화하므로, "빨간 차 옆의 파란 자전거"와 "파란 차 옆의 빨간 자전거"를 동일한 의미로 혼동함 (Compositionality Failure).
- **LLaVA 계열**: 단순 선형 투영(Linear Projection)에 의존하여 시각 패치 토큰의 세부 공간 좌표 및 객체 간 관계(Relation) 정보가 LLM 텍스트 레이어로 전이되는 과정에서 소실됨.

---

## 3. Hyeonwoo's Core Hypothesis & 4-Vector Strategy (핵심 가설 및 4대 발굴 벡터)
- **발굴 벡터**: **[이종 결합 (Cross-Pollination)]** + **[기저 가정 파괴 (Assumption Inversion)]**
- **핵심 가설**:
  1. *가정 파괴*: "이미지와 텍스트는 단일 전역 벡터로 정렬되어야 한다"는 가정을 깨고, 토큰 레벨의 최적 수송(Optimal Transport / Earth Mover's Distance) 기반의 세분화된 정렬 손실을 도입한다.
  2. *이종 결합*: 그래프 신경망(GNN)의 구조적 릴레이션 임베딩을 비전 패치 어텐션 맵에 결합하여 속성-객체-위치 바인딩을 명시적으로 학습한다.

---

## 4. Evaluation & Verification Plan (검증 파이프라인)
- **Benchmarks**: Winoground, ARO (Attribution, Relation, Order), SugarCrepe, Visual Genome Relation.
- **Success Metric**: Winoground Compositional Accuracy 45% $\rightarrow$ 60% 이상 향상, VLM Object Hallucination Rate(POPE) 30% 이상 저감.
