---
title: "INSID3: 단 하나의 자기지도 모델로 In-context Segmentation"
dek: DINOv3 피처만으로 파인튜닝 없이 객체·부분·개인화 세그멘테이션을 모두 달성한 CVPR 2026 Oral 논문.
tags: [Vision, Multimodal]
date: Apr 2026
readtime: 10 min read
slug: insid3
katex: true
---

## 한 줄 요약

> **INSID3**는 DINOv3 하나만으로 — 어떤 학습도, SAM도 없이 — 원샷 세그멘테이션 SOTA를 +7.5% mIoU 차이로 갱신한다.

In-context Segmentation(ICS)은 주석이 달린 예시 이미지 한 장을 참고해서 대상 이미지의 임의 개념(객체, 부분, 특정 인스턴스)을 분할하는 task다. 기존 연구들은 크게 두 갈래로 나뉜다.

- **파인튜닝 방식**: DINOv2 위에 세그멘테이션 디코더를 학습 → 인도메인 성능↑, 일반화↓
- **학습 없는 방식**: DINOv2(대응) + SAM(마스크) 조합 → 일반화↑, 구조 복잡↑

INSID3는 두 방식의 단점을 모두 피하는 세 번째 길을 제시한다: **DINOv3만 쓰되, 그 안에서 대응과 세그멘테이션을 동시에 해결한다.**

<div class="ornament">· · ·</div>

## 왜 DINOv3인가?

DINOv3는 대규모 이미지 코퍼스에서 순수 자기지도 학습된 Vision Foundation Model이다. 기존 DINO 계열과 다른 핵심 특성이 있다.

- **Dense localized features**: 패치 임베딩이 공간 구조를 강하게 보존
- **강한 자기유사성(self-similarity)**: 같은 이미지 내 동일 물체/부분 패치들이 feature space에서 뭉친다
- **의미적 대응**: 다른 이미지에서도 같은 개념의 패치끼리 유사하다

아래 그림은 DINOv3 피처에 agglomerative clustering을 적용했을 때 객체와 부분이 자연스럽게 분리되는 모습을 보여준다.

<svg viewBox="0 0 720 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:720px;display:block;margin:2rem auto">
  <!-- 원본 이미지 영역 -->
  <rect x="20" y="30" width="160" height="160" rx="8" fill="#f0f0f0" stroke="#ccc"/>
  <text x="100" y="18" text-anchor="middle" font-size="12" fill="#666" font-family="sans-serif">Target Image</text>
  <!-- 사람 실루엣 -->
  <ellipse cx="100" cy="75" rx="20" ry="22" fill="#b0b0c0"/>
  <rect x="80" y="95" width="40" height="60" rx="6" fill="#8090b0"/>
  <rect x="65" y="98" width="18" height="45" rx="5" fill="#8090b0"/>
  <rect x="117" y="98" width="18" height="45" rx="5" fill="#8090b0"/>
  <rect x="80" y="150" width="18" height="38" rx="5" fill="#7080a0"/>
  <rect x="102" y="150" width="18" height="38" rx="5" fill="#7080a0"/>
  <!-- 화살표 -->
  <text x="200" y="115" text-anchor="middle" font-size="24" fill="#999">→</text>
  <text x="200" y="132" text-anchor="middle" font-size="10" fill="#888" font-family="sans-serif">DINOv3 +</text>
  <text x="200" y="144" text-anchor="middle" font-size="10" fill="#888" font-family="sans-serif">Clustering</text>
  <!-- 클러스터 결과 -->
  <rect x="230" y="30" width="160" height="160" rx="8" fill="#f8f8f8" stroke="#ccc"/>
  <text x="310" y="18" text-anchor="middle" font-size="12" fill="#666" font-family="sans-serif">Clusters (τ=0.6)</text>
  <!-- 머리 클러스터 -->
  <ellipse cx="310" cy="75" rx="20" ry="22" fill="#f4a261" opacity="0.85"/>
  <!-- 몸통 -->
  <rect x="290" y="95" width="40" height="60" rx="6" fill="#457b9d" opacity="0.85"/>
  <!-- 팔 -->
  <rect x="275" y="98" width="18" height="45" rx="5" fill="#2a9d8f" opacity="0.85"/>
  <rect x="327" y="98" width="18" height="45" rx="5" fill="#2a9d8f" opacity="0.85"/>
  <!-- 다리 -->
  <rect x="290" y="150" width="18" height="38" rx="5" fill="#e76f51" opacity="0.85"/>
  <rect x="312" y="150" width="18" height="38" rx="5" fill="#e76f51" opacity="0.85"/>
  <!-- 범례 -->
  <rect x="230" y="198" width="12" height="12" fill="#f4a261" rx="2"/>
  <text x="246" y="209" font-size="10" fill="#555" font-family="sans-serif">머리</text>
  <rect x="275" y="198" width="12" height="12" fill="#457b9d" rx="2"/>
  <text x="291" y="209" font-size="10" fill="#555" font-family="sans-serif">몸통</text>
  <rect x="315" y="198" width="12" height="12" fill="#2a9d8f" rx="2"/>
  <text x="331" y="209" font-size="10" fill="#555" font-family="sans-serif">팔</text>
  <rect x="352" y="198" width="12" height="12" fill="#e76f51" rx="2"/>
  <text x="368" y="209" font-size="10" fill="#555" font-family="sans-serif">다리</text>
  <!-- 파이프라인 나머지 -->
  <text x="415" y="115" text-anchor="middle" font-size="24" fill="#999">→</text>
  <text x="415" y="132" text-anchor="middle" font-size="10" fill="#888" font-family="sans-serif">Seed 선택</text>
  <text x="415" y="144" text-anchor="middle" font-size="10" fill="#888" font-family="sans-serif">+ 집계</text>
  <rect x="445" y="30" width="160" height="160" rx="8" fill="#f8f8f8" stroke="#ccc"/>
  <text x="525" y="18" text-anchor="middle" font-size="12" fill="#666" font-family="sans-serif">Final Mask</text>
  <!-- 머리만 강조 (part segmentation 예시) -->
  <ellipse cx="525" cy="75" rx="20" ry="22" fill="#f4a261" opacity="0.9"/>
  <rect x="505" y="95" width="40" height="60" rx="6" fill="#ddd" opacity="0.5"/>
  <rect x="490" y="98" width="18" height="45" rx="5" fill="#ddd" opacity="0.5"/>
  <rect x="542" y="98" width="18" height="45" rx="5" fill="#ddd" opacity="0.5"/>
  <rect x="505" y="150" width="18" height="38" rx="5" fill="#ddd" opacity="0.5"/>
  <rect x="527" y="150" width="18" height="38" rx="5" fill="#ddd" opacity="0.5"/>
  <text x="525" y="210" text-anchor="middle" font-size="10" fill="#888" font-family="sans-serif">ref: "person head" → head만 분리</text>
</svg>

<div class="ornament">· · ·</div>

## 핵심 문제: Positional Bias

DINOv3의 피처를 그대로 cross-image matching에 쓰면 심각한 문제가 생긴다. **두 이미지에서 의미와 무관하게 같은 절대 좌표 위치끼리 유사하게 매칭되는 현상**이다.

예를 들어 "사람 머리" 레퍼런스 마스크의 prototype을 구해서 target 이미지와 유사도를 계산하면, 의미적으로 매칭되어야 할 머리 영역 외에 target 이미지의 같은 위치 패치들도 함께 활성화된다.

<div class="callout">
  <strong>왜 생기나?</strong> DINOv3의 학습 목표 중 <em>Gram anchoring</em>은 패치 임베딩의 공분산 행렬을 안정시킨다. 이 목표가 공간 일관성을 높이는 동시에 절대 위치 정보가 피처에 과하게 녹아드는 부작용을 낳는다.
</div>

논문은 이를 **Positional Bias**라 부르고, 노이즈 이미지 하나로 이 편향을 추정해 제거하는 방법을 제안한다.

### Debiasing: 노이즈 이미지로 위치 부분공간 제거

$$
\mathbf{F}^{\text{noise}} = \Phi(\mathbf{I}^{\text{noise}}), \quad \mathbf{I}^{\text{noise}} \sim \mathcal{N}(\mathbf{0}, \mathbf{1})
$$

노이즈 이미지를 DINOv3에 통과시키면 의미 정보는 없고 위치 정보만 남은 피처 $\mathbf{F}^{\text{noise}}$를 얻는다. 여기에 SVD를 적용해 상위 $s$개 우특이벡터 $\mathbf{B}$를 추출한다. 이것이 **위치 부분공간(positional subspace)**이다.

실제 피처에서 이 부분공간을 투영 제거(orthogonal complement projection)한다:

$$
\tilde{\mathbf{F}} = \mathbf{F}(\mathbf{I}_D - \mathbf{B}\mathbf{B}^\top)
$$

이렇게 얻은 **debiased feature** $\tilde{\mathbf{F}}$는 cross-image matching에, 원래 피처 $\mathbf{F}$는 intra-image clustering에 사용한다. 위치 정보가 intra-image grouping에서는 오히려 도움이 되기 때문이다.

<div class="callout">
  <strong>간단한 비유:</strong> 지도에서 "한강 북쪽에 있는 건물"을 찾을 때 (cross-image), 방위(위치 편향)를 제거하고 건물 외관(의미)만 비교해야 정확하다. 반면 같은 지도 안에서 "이웃한 건물들 묶기" (intra-image)에는 방위 정보도 도움이 된다.
</div>

<div class="ornament">· · ·</div>

## INSID3 파이프라인 3단계

<svg viewBox="0 0 740 300" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:740px;display:block;margin:2rem auto">
  <!-- 배경 박스들 -->
  <rect x="10" y="50" width="155" height="200" rx="10" fill="#eef2ff" stroke="#a5b4fc" stroke-width="1.5"/>
  <rect x="195" y="50" width="155" height="200" rx="10" fill="#f0fdf4" stroke="#86efac" stroke-width="1.5"/>
  <rect x="380" y="50" width="155" height="200" rx="10" fill="#fefce8" stroke="#fde047" stroke-width="1.5"/>
  <rect x="565" y="50" width="155" height="200" rx="10" fill="#fdf4ff" stroke="#e879f9" stroke-width="1.5"/>
  <!-- 단계 번호 -->
  <circle cx="87" cy="75" r="14" fill="#6366f1"/>
  <text x="87" y="80" text-anchor="middle" font-size="13" fill="white" font-family="sans-serif" font-weight="bold">1</text>
  <circle cx="272" cy="75" r="14" fill="#22c55e"/>
  <text x="272" y="80" text-anchor="middle" font-size="13" fill="white" font-family="sans-serif" font-weight="bold">2</text>
  <circle cx="457" cy="75" r="14" fill="#eab308"/>
  <text x="457" y="80" text-anchor="middle" font-size="13" fill="white" font-family="sans-serif" font-weight="bold">3</text>
  <circle cx="642" cy="75" r="14" fill="#a855f7"/>
  <text x="642" y="80" text-anchor="middle" font-size="13" fill="white" font-family="sans-serif" font-weight="bold">✓</text>
  <!-- 제목 -->
  <text x="87" y="108" text-anchor="middle" font-size="12" fill="#4338ca" font-family="sans-serif" font-weight="bold">Fine-grained</text>
  <text x="87" y="122" text-anchor="middle" font-size="12" fill="#4338ca" font-family="sans-serif" font-weight="bold">Clustering</text>
  <text x="272" y="108" text-anchor="middle" font-size="12" fill="#15803d" font-family="sans-serif" font-weight="bold">Seed-cluster</text>
  <text x="272" y="122" text-anchor="middle" font-size="12" fill="#15803d" font-family="sans-serif" font-weight="bold">Selection</text>
  <text x="457" y="108" text-anchor="middle" font-size="12" fill="#854d0e" font-family="sans-serif" font-weight="bold">Cluster</text>
  <text x="457" y="122" text-anchor="middle" font-size="12" fill="#854d0e" font-family="sans-serif" font-weight="bold">Aggregation</text>
  <text x="642" y="108" text-anchor="middle" font-size="12" fill="#7e22ce" font-family="sans-serif" font-weight="bold">Final</text>
  <text x="642" y="122" text-anchor="middle" font-size="12" fill="#7e22ce" font-family="sans-serif" font-weight="bold">Mask</text>
  <!-- 설명 -->
  <text x="87" y="150" text-anchor="middle" font-size="10" fill="#555" font-family="sans-serif">원본 피처 F^t에</text>
  <text x="87" y="164" text-anchor="middle" font-size="10" fill="#555" font-family="sans-serif">agglomerative</text>
  <text x="87" y="178" text-anchor="middle" font-size="10" fill="#555" font-family="sans-serif">clustering 적용</text>
  <text x="87" y="192" text-anchor="middle" font-size="10" fill="#555" font-family="sans-serif">→ K개 클러스터</text>
  <text x="272" y="150" text-anchor="middle" font-size="10" fill="#555" font-family="sans-serif">debiased 피처로</text>
  <text x="272" y="164" text-anchor="middle" font-size="10" fill="#555" font-family="sans-serif">backward NN 매칭</text>
  <text x="272" y="178" text-anchor="middle" font-size="10" fill="#555" font-family="sans-serif">→ 후보 클러스터</text>
  <text x="272" y="192" text-anchor="middle" font-size="10" fill="#555" font-family="sans-serif">→ 최고 cross-sim</text>
  <text x="272" y="206" text-anchor="middle" font-size="10" fill="#555" font-family="sans-serif">클러스터 = seed</text>
  <text x="457" y="150" text-anchor="middle" font-size="10" fill="#555" font-family="sans-serif">cross-sim ×</text>
  <text x="457" y="164" text-anchor="middle" font-size="10" fill="#555" font-family="sans-serif">intra-sim으로</text>
  <text x="457" y="178" text-anchor="middle" font-size="10" fill="#555" font-family="sans-serif">주변 클러스터</text>
  <text x="457" y="192" text-anchor="middle" font-size="10" fill="#555" font-family="sans-serif">병합 여부 결정</text>
  <text x="642" y="150" text-anchor="middle" font-size="10" fill="#555" font-family="sans-serif">seed + 병합된</text>
  <text x="642" y="164" text-anchor="middle" font-size="10" fill="#555" font-family="sans-serif">클러스터 합집합</text>
  <text x="642" y="178" text-anchor="middle" font-size="10" fill="#555" font-family="sans-serif">→ CRF refinement</text>
  <text x="642" y="192" text-anchor="middle" font-size="10" fill="#555" font-family="sans-serif">→ 최종 마스크</text>
  <!-- 화살표 -->
  <line x1="165" y1="150" x2="190" y2="150" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="350" y1="150" x2="375" y2="150" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="535" y1="150" x2="560" y2="150" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
  <defs>
    <marker id="arr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#999"/>
    </marker>
  </defs>
</svg>

### Step 1: Fine-grained Clustering

target 이미지의 원본 DINOv3 피처 $\mathbf{F}^t$에 agglomerative clustering을 적용한다. K-means와 달리 클러스터 수를 사전 지정할 필요 없고, 임계값 $\tau$ 하나로 granularity를 조절한다. DINOv3의 공간적 일관성 덕분에 클러스터가 자연스럽게 의미 단위로 묶인다.

### Step 2: Seed-cluster Selection

후보 클러스터를 추리고, 그 중 레퍼런스와 가장 의미적으로 가까운 **seed 클러스터**를 선택한다.

**후보 추리기 (backward matching)**: 각 target 패치 $i$에서 reference 피처 $\tilde{\mathbf{F}}^r$로 nearest neighbor를 찾는다. 이 NN이 레퍼런스 마스크 안에 있는 패치들만 후보 $\mathcal{C}_\text{NN}$으로 남긴다.

$$
\text{NN}(i) = \arg\max_{j} \langle \tilde{\mathbf{F}}^t_i, \tilde{\mathbf{F}}^r_j \rangle, \quad \mathcal{C}_\text{NN} = \{i \mid \mathbf{M}^r_{\text{NN}(i)} = 1\}
$$

Forward matching("레퍼런스 prototype → target 전체")보다 backward matching이 유리한 이유: target 패치들이 암묵적으로 레퍼런스의 non-mask 영역을 negative로 활용하기 때문이다.

**seed 선택**: 후보 클러스터 각각의 prototype과 레퍼런스 prototype 사이의 cross-image similarity를 계산해 가장 높은 클러스터를 seed $\mathcal{G}^*$로 선택한다.

### Step 3: Cluster Aggregation

seed 클러스터만으로는 개념의 일부(예: 기린 목)만 덮는 경우가 많다. 나머지 후보 클러스터를 병합할지 결정하는 combinned score를 사용한다:

$$
S_k = s^\text{cross}_k \cdot s^\text{intra}_k
$$

- $s^\text{cross}_k$: debiased 피처 기반 reference와의 의미적 유사도
- $s^\text{intra}_k$: 원본 피처 기반 seed 클러스터와의 구조적 일관성

둘 다 높은 클러스터만 병합하므로 오염 없이 개념의 전체 범위를 복원한다.

<div class="ornament">· · ·</div>

## 실험 결과

INSID3는 학습 없는 방법으로 파인튜닝 기반 모델들까지 압도한다.

<table style="width:100%;border-collapse:collapse;font-size:0.88rem;margin:1.5rem 0">
  <thead>
    <tr style="background:#f4f4f4">
      <th style="padding:8px;border:1px solid #ddd;text-align:left">방법</th>
      <th style="padding:8px;border:1px solid #ddd;text-align:center">학습</th>
      <th style="padding:8px;border:1px solid #ddd;text-align:center">COCO-20ⁱ</th>
      <th style="padding:8px;border:1px solid #ddd;text-align:center">LVIS-92ⁱ</th>
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
      <td style="padding:8px;border:1px solid #ddd;text-align:center">—</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">39.9</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">51.8</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">~950M</td>
    </tr>
    <tr>
      <td style="padding:8px;border:1px solid #ddd">GF-SAM</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">❌ 학습 없음</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">55.1</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">—</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">44.5</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">54.1</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">945M</td>
    </tr>
    <tr style="background:#f0fdf4;font-weight:bold">
      <td style="padding:8px;border:1px solid #ddd">INSID3 (ours)</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">❌ 학습 없음</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">57.6</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">+6.6%↑</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">50.5</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center">67.0</td>
      <td style="padding:8px;border:1px solid #ddd;text-align:center"><span style="background:#d4f7d4;padding:1px 4px;border-radius:3px">304M</span></td>
    </tr>
  </tbody>
</table>

주목할 점들:

- **파라미터 3배 절감**: 945M(GF-SAM) → 304M(INSID3)
- **개인화 세그멘테이션 +12.9%**: PerMIS에서 압도적 차이. backward matching이 distractor 억제에 효과적
- **Chest X-ray +27.8%**: 도메인 외 이미지에서도 일반화 강력
- **파인튜닝 방법들 능가**: 인도메인 일부 데이터셋(COCO)에서만 SegIC가 앞서고, 나머지에서는 INSID3가 모두 앞선다

<div class="ornament">· · ·</div>

## Debiasing의 범용성: Semantic Correspondence에도 통한다

논문이 발견한 positional bias는 ICS에만 국한된 문제가 아니다. SPair-71k 벤치마크(semantic correspondence 표준)에서 DINOv3 피처에 debiasing을 적용하면 PCK가 **+0.9~+6.6%** 향상된다. 별도 학습 없이 피처 전처리만으로 얻는 공짜 성능 향상이다.

<div class="callout">
  <strong>시사점:</strong> DINOv3를 cross-image matching에 사용하는 모든 downstream task — optical flow, pose estimation, 3D reconstruction 등 — 에서 이 debiasing 기법이 효과적일 수 있다.
</div>

<div class="ornament">· · ·</div>

## 요약

INSID3는 "더 많은 모델을 조합할수록 좋다"는 통념을 뒤집는다. 단 하나의 자기지도 모델로, 복잡한 파이프라인 없이 SOTA를 달성한다.

핵심 기여 세 가지:

1. **Positional Bias 발견 & 제거**: 노이즈 이미지 한 장으로 DINOv3의 위치 편향을 추정·제거. Cross-image matching 정확도를 대폭 향상시키며 ICS 너머 다른 task에도 적용 가능
2. **Agglomerative Clustering**: 클러스터 수를 미리 정하지 않아도 되는 유연한 파트 분리. DINOv3의 공간 일관성과 시너지
3. **Seed + Aggregation**: Cross-image similarity와 intra-image self-similarity를 곱셈 결합해 개념의 전체 범위를 정확하게 복원

<div class="footnote">
  논문: <a href="https://arxiv.org/abs/2506.00000" target="_blank">INSID3: In-context Segmentation with DINOv3</a> (CVPR 2026 Oral)
</div>
