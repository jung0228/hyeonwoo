# 대조 학습 (Contrastive Learning)

## 핵심 아이디어
라벨이 없는 대규모 데이터에서 서로 유사한 샘플(Positive Pair)은 임베딩 공간에서 가깝게 끌어당기고(Attract), 서로 다른 샘플(Negative Pair)은 멀리 밀어내는(Repel) 방식으로 고차원 의미 공간의 표현(Representation)을 학습하는 자기지도학습(Self-Supervised Learning)의 핵심 방법론입니다.

---

## 핵심 수식 및 손실 함수

### 1. InfoNCE (Information Noise-Contrastive Estimation) Loss
쿼리 표현 $\mathbf{q}$, 정답 양성 샘플 $\mathbf{k}_+$, 그리고 $K$개의 음성 샘플 $\{\mathbf{k}_1, \dots, \mathbf{k}_K\}$가 주어졌을 때:

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\mathbf{q} \cdot \mathbf{k}_+ / \tau)}{\exp(\mathbf{q} \cdot \mathbf{k}_+ / \tau) + \sum_{i=1}^{K} \exp(\mathbf{q} \cdot \mathbf{k}_i / \tau)}$$

- $\tau > 0$: 온도 하이퍼파라미터 (Temperature), 점수 분포의 뾰족한 정도(Sharpness)를 조절.
- 소프트맥스 Cross-Entropy 손실 함수와 수학적으로 동일한 형태를 취하여 상호 정보량(Mutual Information)의 하한(Lower Bound)을 최대화합니다.

### 2. Triplet Loss (삼중항 손실)
$$\mathcal{L}_{\text{triplet}} = \max\left(0, \|\mathbf{a} - \mathbf{p}\|_2^2 - \|\mathbf{a} - \mathbf{n}\|_2^2 + \alpha\right)$$
- $\mathbf{a}$: Anchor, $\mathbf{p}$: Positive, $\mathbf{n}$: Negative, $\alpha$: Margin.

---

## 직관적 설명
수많은 사진이 흩어져 있는 방에서, "같은 고양이의 다른 각도 사진"은 서로 자석처럼 붙이고, "강아지나 자동차 사진"은 척력으로 밀어내어, 책상 위에 사물별로 완벽하게 군집화된 지도를 스스로 만들어내는 과정입니다.

---

## 연결 개념 및 논문
- [[mle_map]] : 상호 정보량 최대화와 확률적 우도 최적화의 수학적 기반
- [[cross_entropy]] : InfoNCE의 소프트맥스 정규화 수식 근원
- [[clip]] : 이미지-텍스트 대규모 멀티모달 쌍에 InfoNCE를 적용한 대표 마일스톤 논문
- [[rq_cross_modal_alignment]] : Coarse Alignment를 넘어선 Fine-grained 대조 학습 연구 과제

---

## 참고
- Representation Learning with Contrastive Predictive Coding (CPC, Oord et al., 2018)
- A Simple Framework for Contrastive Learning of Visual Representations (SimCLR, Chen et al., ICML 2020)
