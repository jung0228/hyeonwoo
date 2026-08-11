---
title: "INSID3: 단 하나의 자기지도 모델로 In-context Segmentation"
dek: DINOv3 피처만으로 파인튜닝 없이 객체·부분·개인화 세그멘테이션을 모두 달성한 CVPR 2026 Oral 논문.
tags: [Vision, Multimodal]
date: Apr 2026
readtime: 12 min read
slug: insid3
katex: true
---

## 한 줄 요약

<p><strong>흐름:</strong> 문제 정의 → 기존 방법의 한계 → INSID3의 접근 → 성능</p>

<mark>DINOv3 하나만으로 — 어떤 학습도, SAM도 없이 — 원샷 세그멘테이션 SOTA를 평균 +7.5% mIoU 차이로 갱신한다.</mark>

In-context Segmentation(ICS)은 주석이 달린 예시 이미지(reference) 한 장을 참고해서 target 이미지의 임의 개념(객체, 부분, 특정 인스턴스)을 분할하는 task다. 기존 연구들은 크게 두 갈래였다.

- **파인튜닝 방식**: DINOv2 위에 세그멘테이션 디코더를 학습. 인도메인 성능↑, 일반화↓
- **학습 없는 방식**: DINOv2(대응) + SAM(마스크) 조합. 일반화↑, 구조 복잡↑, 두 모델 간 정보 손실

INSID3는 세 번째 길을 제시한다: **DINOv3만 쓰되, 그 안에서 대응(correspondence)과 세그멘테이션을 동시에 해결한다.**

<figure>
<img src="img/insid3/teaser.jpg" alt="Teaser">
<figcaption><strong>Figure 1</strong> — 거미줄 차트: INSID3(보라)는 파인튜닝 방법(주황)과 SAM 기반 학습 없는 방법(파랑) 모두를 전 데이터셋에서 앞선다. 파라미터는 3배 적고, 어떤 mask/category 감독도 사용하지 않는다.</figcaption>
</figure>

<div class="ornament">· · ·</div>

## 왜 DINOv3인가?

<p><strong>흐름:</strong> DINOv3의 특성 → clustering으로 확인</p>

DINOv3는 대규모 이미지 코퍼스에서 순수 자기지도 학습된 Vision Foundation Model이다. 기존 DINO 계열과 다른 핵심 특성은 **Dense localized features**다. 패치 임베딩이 공간 구조를 강하게 보존하며, 같은 물체/부분에 속하는 패치들이 feature space에서 자연스럽게 뭉친다.

아래 그림은 DINOv3 피처에 agglomerative clustering을 적용했을 때의 결과다. 학습 없이, 별도 레이블 없이 물체와 부분이 색깔별로 분리된다.

<figure>
<img src="img/insid3/clustering.png" alt="DINOv3 Clustering">
<figcaption><strong>Figure 2</strong> — DINOv3 피처에 agglomerative clustering을 적용한 결과. 음식, 소화전, 말, 피자, 버스, 실내 등 다양한 도메인에서 의미 단위로 클러스터가 형성된다. 어떤 감독도 없이 이 수준의 공간 분할이 가능하다는 것이 INSID3의 출발점이다.</figcaption>
</figure>

<div class="ornament">· · ·</div>

## 핵심 문제: Positional Bias

<p><strong>흐름:</strong> 문제 발견 → 원인 분석 → 해결책</p>

DINOv3의 피처를 그대로 cross-image matching에 쓰면 치명적인 문제가 생긴다. **두 이미지에서 의미와 무관하게 같은 절대 좌표 위치끼리 유사하게 매칭되는 현상 — Positional Bias**다.

아래 두 예시를 보자. Reference는 초록색 마스크로 개념을 표시하고, query는 같은 개념이 포함된 다른 이미지다.

### 예시 1: 야구 배트

<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:1.5rem 0">
  <figure style="margin:0">
    <img src="img/insid3/568_support.png" alt="Reference" style="width:100%;border-radius:4px">
    <figcaption style="font-size:0.78rem;color:#666;text-align:center;margin-top:4px">Reference<br>(야구 배트 마스킹)</figcaption>
  </figure>
  <figure style="margin:0">
    <img src="img/insid3/568_query.png" alt="Query" style="width:100%;border-radius:4px">
    <figcaption style="font-size:0.78rem;color:#666;text-align:center;margin-top:4px">Query<br>(아이 + 야구 배트)</figcaption>
  </figure>
  <figure style="margin:0">
    <img src="img/insid3/568_sim.png" alt="Similarity (original)" style="width:100%;border-radius:4px">
    <figcaption style="font-size:0.78rem;color:#666;text-align:center;margin-top:4px">Original sim<br>(debiasing 전)</figcaption>
  </figure>
  <figure style="margin:0">
    <img src="img/insid3/568_sim_debias.png" alt="Similarity (debiased)" style="width:100%;border-radius:4px">
    <figcaption style="font-size:0.78rem;color:#666;text-align:center;margin-top:4px">Debiased sim<br>(debiasing 후)</figcaption>
  </figure>
</div>

### 예시 2: 초록 박스 → 짐가방

<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:1.5rem 0">
  <figure style="margin:0">
    <img src="img/insid3/854_support.png" alt="Reference" style="width:100%;border-radius:4px">
    <figcaption style="font-size:0.78rem;color:#666;text-align:center;margin-top:4px">Reference<br>(초록 박스)</figcaption>
  </figure>
  <figure style="margin:0">
    <img src="img/insid3/854_query.png" alt="Query" style="width:100%;border-radius:4px">
    <figcaption style="font-size:0.78rem;color:#666;text-align:center;margin-top:4px">Query<br>(짐가방)</figcaption>
  </figure>
  <figure style="margin:0">
    <img src="img/insid3/854_sim.png" alt="Similarity (original)" style="width:100%;border-radius:4px">
    <figcaption style="font-size:0.78rem;color:#666;text-align:center;margin-top:4px">Original sim<br>(하단 바닥에 활성화!)</figcaption>
  </figure>
  <figure style="margin:0">
    <img src="img/insid3/854_sim_debias.png" alt="Similarity (debiased)" style="width:100%;border-radius:4px">
    <figcaption style="font-size:0.78rem;color:#666;text-align:center;margin-top:4px">Debiased sim<br>(가방에 집중)</figcaption>
  </figure>
</div>

예시 2가 특히 충격적이다. Reference의 초록 박스가 이미지 하단에 위치하니, query의 짐가방(이미지 중앙)을 찾아야 하는데 오히려 query 이미지 하단 바닥에 활성화가 몰린다. 의미가 아니라 **위치** 때문이다.

<div class="callout">
  <strong>왜 생기나?</strong> DINOv3의 학습 목표 중 <em>Gram anchoring</em>은 패치 임베딩의 공분산 행렬을 안정시킨다. 공간 일관성을 높이는 이 목표가 부작용으로 절대 위치 정보를 피처에 과하게 녹아들게 한다. DINOv2에서는 이 현상이 훨씬 약하다.
</div>

<figure>
<img src="img/insid3/comparison_dinov2_dinov3.jpg" alt="DINOv2 vs DINOv3 positional bias">
<figcaption><strong>Figure 3</strong> — DINOv3 original(왼쪽)과 debiased(오른쪽), 그리고 DINOv2(하단) 비교. DINOv3는 positional bias가 강하게 나타나지만 debiasing 후 올바른 semantic 위치를 찾는다. DINOv2는 bias 자체가 약해 debiasing 효과가 작다.</figcaption>
</figure>

### Debiasing: 노이즈 이미지로 위치 부분공간 제거

노이즈 이미지를 DINOv3에 통과시키면 의미 정보는 없고 위치 정보만 남은 피처를 얻는다.

$$
\mathbf{F}^{\text{noise}} = \Phi(\mathbf{I}^{\text{noise}}), \quad \mathbf{I}^{\text{noise}} \sim \mathcal{N}(\mathbf{0}, \mathbf{1})
$$

여기에 SVD를 적용해 상위 $s$개 우특이벡터 $\mathbf{B}$를 추출한다. 이것이 **위치 부분공간**이다. 실제 피처에서 이 부분공간을 orthogonal projection으로 제거한다:

$$
\tilde{\mathbf{F}} = \mathbf{F}(\mathbf{I}_D - \mathbf{B}\mathbf{B}^\top)
$$

이렇게 얻은 **debiased feature** $\tilde{\mathbf{F}}$는 cross-image matching에, 원래 피처 $\mathbf{F}$는 intra-image clustering에 사용한다. 위치 정보가 intra-image grouping에서는 오히려 도움이 되기 때문이다.

<div class="ornament">· · ·</div>

## INSID3 파이프라인

<p><strong>흐름:</strong> Clustering → Seed 선택 → Aggregation → 최종 마스크</p>

<figure>
<img src="img/insid3/method.jpg" alt="INSID3 Method">
<figcaption><strong>Figure 4</strong> — INSID3 전체 파이프라인. Reference와 Target 모두 DINOv3를 통과하고, cross-image matching에는 debiased 피처를, clustering에는 원본 피처를 사용한다. 세 단계(Clustering → Seed 선택 → Aggregation)를 거쳐 최종 마스크를 예측한다.</figcaption>
</figure>

### Fine-grained Clustering

Target 이미지의 원본 DINOv3 피처 $\mathbf{F}^t$에 agglomerative clustering을 적용한다. K-means와 달리 클러스터 수를 사전 지정할 필요 없고, 임계값 $\tau$ 하나로 granularity를 조절한다.

$$
\bigcup_{k=1}^K \mathcal{G}_k = \Omega, \quad \mathcal{G}_i \cap \mathcal{G}_j = \emptyset \quad \forall i \neq j
$$

DINOv3의 공간적 일관성 덕분에 클러스터가 자연스럽게 의미 단위로 묶인다.

### Seed-cluster Selection

**후보 추리기 — Backward Matching**: 각 target 패치 $i$에서 reference 피처로 nearest neighbor를 찾는다. 이 NN이 reference 마스크 안에 있는 경우만 후보 $\mathcal{C}_\text{NN}$으로 남긴다.

$$
\text{NN}(i) = \arg\max_{j} \langle \tilde{\mathbf{F}}^t_i, \tilde{\mathbf{F}}^r_j \rangle, \quad \mathcal{C}_\text{NN} = \{i \mid \mathbf{M}^r_{\text{NN}(i)} = 1\}
$$

Forward matching("reference prototype → target 전체")보다 backward matching이 유리한 이유: target 패치들이 reference의 non-mask 영역을 **암묵적으로 negative로 활용**하기 때문이다. 특히 personalized segmentation에서 distractor 억제에 효과적이다.

**Seed 선택**: 후보 클러스터 각각의 prototype과 reference prototype 사이 cross-image similarity를 계산해 가장 높은 클러스터를 seed $\mathcal{G}^*$로 선택한다.

$$
s^\text{cross}_k = \langle \tilde{\mathbf{p}}^t_k,\, \tilde{\mathbf{p}}^r \rangle, \quad \mathcal{G}^* = \arg\max_{\mathcal{G}_k \in \mathcal{C}_\text{cand}} s^\text{cross}_k
$$

### Cluster Aggregation

Seed 클러스터만으로는 개념의 일부(예: 기린 목)만 덮는 경우가 많다. 나머지 후보 클러스터를 병합할지 결정하는 combined score:

$$
S_k = s^\text{cross}_k \cdot s^\text{intra}_k
$$

- $s^\text{cross}_k$: debiased 피처 기반 reference와의 **의미적 유사도** (cross-image)
- $s^\text{intra}_k$: 원본 피처 기반 seed 클러스터와의 **구조적 일관성** (intra-image)

두 score를 곱해서 둘 다 높은 클러스터만 병합하므로, 뷰포인트 변화나 occlusion이 있어도 개념의 전체 범위를 깔끔하게 복원한다.

$$
\mathcal{M}_\text{final} = \mathcal{G}^* \cup \{\mathcal{G}_k \in \mathcal{C}_\text{cand} \mid S_k \geq \alpha\}
$$

<div class="ornament">· · ·</div>

## 실험 결과

<p><strong>흐름:</strong> Semantic → Part → Personalized 세그멘테이션 순서로 결과 확인</p>

<figure>
<img src="img/insid3/qualitatives.jpg" alt="Qualitative results">
<figcaption><strong>Figure 5</strong> — GF-SAM, Matcher, INSID3의 정성적 비교. 왼쪽: semantic segmentation (胸部 X-ray 포함). 오른쪽 상단: part segmentation. 오른쪽 하단: personalized segmentation. INSID3의 마스크가 훨씬 정확하고 깔끔하다.</figcaption>
</figure>

<table style="width:100%;border-collapse:collapse;font-size:0.88rem;margin:1.5rem 0">
  <thead>
    <tr style="background:#f4f4f4">
      <th style="padding:8px;border:1px solid #ddd;text-align:left">방법</th>
      <th style="padding:8px;border:1px solid #ddd;text-align:center">학습</th>
      <th style="padding:8px;border:1px solid #ddd;text-align:center">COCO-20ⁱ</th>
      <th style="padding:8px;border:1px solid #ddd;text-align:center">Chest X-ray</th>
      <th style="padding:8px;border:1px solid #ddd;text-align:center">PASCAL-Part</th>
      <th style="padding:8px;border:1px solid #ddd;text-align:center">PerMIS</th>
      <th style="padding:8px;border:1px solid #ddd;text-align:center">파라미터</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:8px;border:1px solid #ddd">SegIC</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">✅ 파인튜닝</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">76.1</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">낮음</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">39.9</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">51.8</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">~950M</td>
    </tr>
    <tr>
      <td style="padding:8px;border:1px solid #ddd">GF-SAM</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">❌</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">55.1</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">낮음</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">44.5</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">54.1</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">945M</td>
    </tr>
    <tr style="background:#f0fdf4;font-weight:bold">
      <td style="padding:8px;border:1px solid #ddd">INSID3 (ours)</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">❌</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">57.6</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center"><span style="background:#d4f7d4;padding:1px 4px;border-radius:3px">+27.8%↑</span></td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">50.5</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center"><span style="background:#d4f7d4;padding:1px 4px;border-radius:3px">67.0</span></td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center"><span style="background:#d4f7d4;padding:1px 4px;border-radius:3px">304M</span></td>
    </tr>
  </tbody>
</table>

주목할 점들:

- **파라미터 3배 절감**: 945M → 304M, 성능은 오히려 높음
- **Chest X-ray +27.8%**: 학습 데이터에 없는 도메인에서 일반화가 얼마나 중요한지 보여줌
- **PerMIS +12.9%**: Backward matching이 distractor를 암묵적으로 억제해 개인화 세그멘테이션에서 압도적
- **파인튜닝 모델 vs 인도메인**: SegIC는 COCO에서 76.1%로 앞서지만, 다른 도메인에서 급락 — 특화의 트레이드오프

<div class="ornament">· · ·</div>

## Debiasing의 범용성: Semantic Correspondence

<p><strong>흐름:</strong> ICS 너머 다른 task에도 통하는가</p>

Positional Bias는 ICS에만 국한된 문제가 아니다. 이미지 간 keypoint를 매칭하는 **semantic correspondence** task에서도 동일한 문제가 생긴다.

아래 4쌍의 예시에서, 각 이미지 쌍의 파란 점이 reference keypoint이고 빨간 점이 모델이 예측한 target keypoint다.

<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:1.5rem 0">
  <figure style="margin:0">
    <img src="img/insid3/dinov3-semantic_correspondence_0.jpg" alt="Correspondence 0" style="width:100%;border-radius:4px">
    <figcaption style="font-size:0.78rem;color:#666;text-align:center;margin-top:4px">새 → 새 (부리)</figcaption>
  </figure>
  <figure style="margin:0">
    <img src="img/insid3/dinov3-semantic_correspondence_1.jpg" alt="Correspondence 1" style="width:100%;border-radius:4px">
    <figcaption style="font-size:0.78rem;color:#666;text-align:center;margin-top:4px">자전거 → 자전거</figcaption>
  </figure>
  <figure style="margin:0">
    <img src="img/insid3/dinov3-semantic_correspondence_2.jpg" alt="Correspondence 2" style="width:100%;border-radius:4px">
    <figcaption style="font-size:0.78rem;color:#666;text-align:center;margin-top:4px">배 → 배</figcaption>
  </figure>
  <figure style="margin:0">
    <img src="img/insid3/dinov3-semantic_correspondence_3.jpg" alt="Correspondence 3" style="width:100%;border-radius:4px">
    <figcaption style="font-size:0.78rem;color:#666;text-align:center;margin-top:4px">병 → 병</figcaption>
  </figure>
</div>

SPair-71k 벤치마크에서 DINOv3에 debiasing을 적용하면 PCK가 **+0.9~+6.6%** 향상된다. 별도 학습 없이 피처 전처리만으로 얻는 공짜 성능 향상이다.

<div class="callout">
  <strong>시사점:</strong> DINOv3를 cross-image matching에 사용하는 모든 downstream task — optical flow, pose estimation, 3D reconstruction 등 — 에서 이 debiasing이 효과적일 수 있다.
</div>

<div class="ornament">· · ·</div>

## 요약

INSID3는 "더 많은 모델을 조합할수록 좋다"는 통념을 뒤집는다.

- **Positional Bias 발견 & 제거**: 노이즈 이미지 한 장으로 DINOv3의 위치 편향을 추정·제거. ICS 너머 semantic correspondence에도 즉시 적용 가능
- **Agglomerative Clustering**: 클러스터 수 미리 정하지 않아도 DINOv3 공간 일관성과 시너지로 의미 단위 분할
- **Seed + Aggregation**: Cross-image와 intra-image similarity를 곱셈 결합해 개념의 전체 범위를 정확하게 복원

단 하나의 자기지도 모델로, 복잡한 파이프라인 없이 SOTA를 달성한다.

<div class="footnote">
  논문: INSID3: In-context Segmentation with DINOv3 (CVPR 2026 Oral)
</div>
