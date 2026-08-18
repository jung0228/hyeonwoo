# 📐 4.6, 4.7 & 4.8 Matrix Approximation, Phylogeny and Spectral Methods (행렬 저계수 근사와 에카르트-영 정리, 행렬 계통도, 스펙트럴 머신러닝)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 4.6, 4.7, 4.8 전수 분석 & 4단계 정밀 해설 노트


## 🌐 0. Chapter 4의 대단원: 압축, 계통도, 그리고 실전 머신러닝

우리는 지난 4.5절에서 모든 직사각형 행렬을 세 행렬의 곱으로 쪼개는 특이값 분해($A = U \Sigma V^\top$)를 완성했습니다.

Chapter 4의 마지막 세 절(4.6, 4.7, 4.8)은 이 모든 이론을 하나로 엮어 "실전 데이터 압축과 머신러닝 응용, 그리고 선형대수학 전체 행렬 분류의 완벽한 지도"를 완성합니다:

1. 4.6절 행렬 저계수 근사와 에카르트-영 정리 (Eckart-Young Theorem):
   거대한 데이터 행렬에서 가장 에너지가 큰 $k$개의 특이값 축만 남겨 원본 크기의 0.6% 수준으로 초경량 압축하면서도 정보 손실을 수학적으로 최소화하는 최적 근사 정리입니다.
2. 4.7절 행렬 계통도 (Matrix Phylogeny):
   일반 직사각형 행렬부터 정방, 가역, 비결함, 정규, 직교, 대칭, 양의 정정, 대각, 단위 행렬까지 모든 행렬 유형의 포함 관계와 연산 가능성을 트리 구조로 집대성합니다.
3. 4.8절 스펙트럴 머신러닝과 텐서 분해 (Further Reading):
   PCA, MDS, Isomap, 라플라시안 고유지도(Laplacian Eigenmaps), 스펙트럴 클러스터링, 그리고 3차원 이상 고차원 데이터를 분해하는 텐서 분해(Tucker, CP Decomposition)의 지평을 엽니다.


## 1. ⚔️ Section 4.6: Matrix Approximation (행렬 저계수 근사)


### 📌 1. 외적 합(Outer Product Sum) 표현과 Rank-1 행렬 (Eq 4.90~4.91)

행렬 $A \in \mathbb{R}^{m \times n}$ (계수 $r$) 의 SVD는 좌특이벡터 $\mathbf{u}_i$ 와 우특이벡터 $\mathbf{v}_i$ 의 외적으로 만들어진 Rank-1 행렬 $A_i$ 들의 가중치 합으로 완벽하게 전개됩니다:

$$A_i := \mathbf{u}_i \mathbf{v}_i^\top \in \mathbb{R}^{m \times n} \quad (\text{Rank-1 행렬, Eq 4.90})$$

$$A = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^\top = \sum_{i=1}^r \sigma_i A_i \quad (\text{Eq 4.91})$$

- 왜 이 식이 성립하는가?:
  특이값 행렬 $\Sigma$ 가 대각 행렬이므로 $i \neq j$ 인 교차 외적항 $\Sigma_{ij}\mathbf{u}_i \mathbf{v}_j^\top$ 은 모두 $0$ 으로 소거되고, $i > r$ 인 항들은 특이값 $\sigma_i = 0$ 이 되어 자동으로 사라집니다.


### 📌 2. Rank-k 저계수 근사 (Rank-k Approximation: Eq 4.92)

원본 행렬 $A$ 의 모든 $r$개 성분을 다 더하지 않고, 가장 중요한 상위 $k$개 ($k < r$) 의 특이값 성분만 잘라서 합산한 행렬을 Rank-$k$ 근사 행렬 $\hat{A}(k)$ 라 부릅니다:

$$\hat{A}(k) := \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top = \sum_{i=1}^k \sigma_i A_i \quad (\text{rk}(\hat{A}(k)) = k, \text{ Eq 4.92})$$

#### 💡 [스톤헨지(Stonehenge) 이미지 압축 사례: Figure 4.11 & 4.12]
- 원본 이미지 행렬 $A \in \mathbb{R}^{1432 \times 1910}$:
  저장해야 할 숫자 $= 1,432 \times 1,910 = \mathbf{2,735,120 \text{ 개 (약 270만 개)}}$.
- Rank-5 저계수 근사 $\hat{A}(5)$:
  5개의 특이값과 각각 1432차원, 1910차원인 5쌍의 좌우 특이벡터만 저장:
  저장해야 할 숫자 $= 5 \times (1,432 + 1,910 + 1) = \mathbf{16,715 \text{ 개}}$.
- 압축률: 원본 데이터 용량의 단 0.6% 만으로도 스톤헨지 바위의 윤곽과 형태를 선명하게 복원합니다!


### 📌 3. 행렬 스펙트럼 노름 (Matrix Spectral Norm: Definition 4.23 & Theorem 4.24)

근사 오차를 측정하기 위해 행렬의 크기를 재는 스펙트럼 노름을 정의합니다:

$$\VertA\Vert_2 := \max_{\mathbf{x} \neq \mathbf{0}} \frac{\VertA\mathbf{x}\Vert_2}{\Vert\mathbf{x}\Vert_2} \quad (\text{Definition 4.23 & Eq 4.93})$$

- Theorem 4.24: 행렬 $A$ 의 스펙트럼 노름은 $A$ 의 가장 큰 특이값 $\sigma_1$ 과 정확히 일치합니다:
  $$\VertA\Vert_2 = \sigma_1$$


### 📌 4. 에카르트-영 정리 (Eckart-Young Theorem: Theorem 4.25 & Eq 4.94~4.99)

에카르트와 영(Eckart & Young, 1936)이 증명한 이 정리는 SVD 저계수 근사가 "세상에 존재하는 모든 계수 $k$ 이하의 행렬들 중 원본 행렬과 오차가 가장 적은 유일한 최적해"임을 보장합니다:

$$\hat{A}(k) = \text{argmin}_{\text{rk}(B) = k} \VertA - B\Vert_2 \quad (\text{Eq 4.94})$$

$$\VertA - \hat{A}(k)\Vert_2 = \sigma_{k+1} \quad (\text{Eq 4.95})$$

- 오차의 수학적 증명 직관:
  원본 $A$ 와 근사 행렬 $\hat{A}(k)$ 의 차이 행렬은 버려진 나머지 특이값들의 합입니다:
  $$A - \hat{A}(k) = \sum_{i=k+1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^\top \quad (\text{Eq 4.96})$$
  이 차이 행렬의 스펙트럼 노름(최대 특이값)은 바로 첫 번째 남은 특이값인 $\sigma_{k+1}$ 이 됩니다.
  만약 이보다 오차가 더 작은 다른 행렬 $B$ 가 존재한다고 가정하면, 차원 정리(Rank-Nullity Theorem)와 코시-슈바르츠 부등식에 의해 차원의 합이 $n$을 초과하는 모순이 발생하여 증명됩니다.


### 💡 [Example 4.15: 넷플릭스 영화 평점 Rank-2 저계수 근사 전수 수치 분석]
앞선 Example 4.14의 영화 평점 행렬 $A \in \mathbb{R}^{4 \times 3}$ 에 대해:

1. 1번 SF 장르 Rank-1 근사 행렬 $A_1 = \mathbf{u}_1 \mathbf{v}_1^\top$ (Eq 4.100):
   $$A_1 = \begin{bmatrix} -0.6710 \\\\ -0.7197 \\\\ -0.0939 \\\\ -0.1515 \end{bmatrix} \begin{bmatrix} -0.7367 & -0.6515 & -0.1811 \end{bmatrix} = \begin{bmatrix} 0.4943 & 0.4372 & 0.1215 \\\\ 0.5302 & 0.4689 & 0.1303 \\\\ 0.0692 & 0.0612 & 0.0170 \\\\ 0.1116 & 0.0987 & 0.0274 \end{bmatrix}$$
   - Ali와 Beatrix의 SF 영화(Star Wars, Blade Runner) 선호 패턴을 정확히 포착합니다.

2. 2번 프랑스 예술영화 Rank-1 근사 행렬 $A_2 = \mathbf{u}_2 \mathbf{v}_2^\top$ (Eq 4.101):
   $$A_2 = \begin{bmatrix} 0.0236 \\\\ 0.2054 \\\\ -0.7705 \\\\ -0.6030 \end{bmatrix} \begin{bmatrix} 0.0852 & 0.1762 & -0.9807 \end{bmatrix} = \begin{bmatrix} 0.0020 & 0.0042 & -0.0231 \\\\ 0.0175 & 0.0362 & -0.2014 \\\\ -0.0656 & -0.1358 & 0.7556 \\\\ -0.0514 & -0.1063 & 0.5914 \end{bmatrix}$$
   - Chandra의 프랑스 예술영화(Amelie, Delicatessen) 선호 패턴을 정확히 포착합니다.

3. 최종 Rank-2 근사 행렬 $\hat{A}(2) = \sigma_1 A_1 + \sigma_2 A_2$ (Eq 4.102):
   $$\hat{A}(2) = 9.6438 A_1 + 6.3639 A_2 = \begin{bmatrix} 4.7801 & 4.2419 & 1.0244 \\\\ 5.2252 & 4.7522 & -0.0250 \\\\ 0.2493 & -0.2743 & 4.9724 \\\\ 0.7495 & 0.2756 & 4.0278 \end{bmatrix} \approx \begin{bmatrix} 5 & 4 & 1 \\\\ 5 & 5 & 0 \\\\ 0 & 0 & 5 \\\\ 1 & 0 & 4 \end{bmatrix} = A$$
   - 3번째 특이값 $\sigma_3 = 0.7056$ 은 매우 작으므로, 상위 2개 테마만으로 원본 평점 테이블을 거의 100% 완벽하게 복원할 수 있음을 실증합니다!


## 2. ⚔️ Section 4.7: Matrix Phylogeny (선형대수학 행렬 계통도)

교재 Figure 4.13의 행렬 계통도(Phylogenetic Tree)는 모든 행렬들의 포함 관계(Subset)와 적용 가능한 연산을 완벽하게 정리합니다:

```text
[모든 실수 행렬 A ∈ Rᵐˣⁿ] ──(SVD 분해 A = UΣVᵀ 100% 항상 존재)──┐
  │
  ▼ (정방행렬 조건 m = n)
[정방행렬 A ∈ Rⁿˣⁿ] ──(행렬식 det(A), 대각합 tr(A), 특성다항식)
  ├── 1. 가역 / 정칙 행렬 (Regular / Invertible): det(A) ≠ 0 ⟺ 역행렬 A⁻¹ 존재
  └── 2. 비결함 행렬 (Non-defective): n개 선형독립 고유벡터 존재 ⟺ 고유값 대각화 A = PDP⁻¹ 존재
        (주의: 가역성과 대각화 가능성은 별개! 회전행렬은 det=1 가역이지만 실수 대각화 불가)
        │
        ▼ (AᵀA = AAᵀ 조건 만족)
      [정규 행렬 (Normal Matrices)]
        ├── (AᵀA = AAᵀ = I 조건) ──> [직교 행렬 (Orthogonal)]: Aᵀ = A⁻¹, 길이/각도 보존 회전
        └── (S = Sᵀ 조건) ──> [대칭 행렬 (Symmetric)]: 스펙트럴 정리, 100% 실수 고유값 & ONB 대각화
                                │
                                ▼ (xᵀPx > 0 조건 만족)
                              [대칭 양의 정정 행렬 (SPD)]: 유일한 숄레스키 분해 A = LLᵀ, 모든 고유값 > 0
                                │
                                ▼ (비대각 성분 = 0 조건 만족)
                              [대각 행렬 (Diagonal D)]: 행렬식/거듭제곱/역행렬 O(n) 초고속
                                │
                                ▼ (모든 대각 성분 = 1)
                              [단위 행렬 (Identity I)]: 항등 변환
```


## 3. ⚔️ Section 4.8: Further Reading & Spectral Machine Learning (스펙트럴 머신러닝 총정리)


### 📌 1. 스펙트럴 방법론 (Spectral Methods in Machine Learning)
1. PCA (주성분 분석 - Ch 10): 공분산 행렬의 상위 $k$개 고유벡터 축으로 데이터를 투영하여 최대 분산 보존.
2. Fisher 판별 분석 (Fisher Discriminant Analysis - FDA): 클래스 간 분산 최대화 및 클래스 내 분산 최소화 분리 초평면 도출.
3. MDS (다차원 척도법 - Multidimensional Scaling): 고차원 데이터 간 거리 관계를 저차원 유클리드 공간에 최대한 보존하며 임베딩.
4. 비선형 매니폴드 학습 (Nonlinear Manifold Learning):
   - Isomap (2000): 최단 경로 측지선 거리(Geodesic distance) 기반 MDS 임베딩.
   - Laplacian Eigenmaps (2003): 그래프 라플라시안의 고유벡터를 이용해 인접한 데이터가 저차원에서도 가깝게 유지되도록 매핑.
   - Hessian Eigenmaps (2003): 매니폴드의 국소 곡률(Hessian)을 최소화하는 고유벡터 추출.
   - Spectral Clustering (Shi & Malik 2000): 그래프 라플라시안 고유벡터를 이용해 복잡한 비선형 군집 분할 수행.


### 📌 2. 텐서 분해 (Tensor Decompositions / Higher-Order SVD)
- 3차원 이상의 고차원 다차원 배열(Tensor)에 대해 SVD를 확장한 기법:
  - 터커 분해 (Tucker Decomposition, 1966): 고차원 텐서를 작은 코어 텐서와 각 모드별 직교 행렬들의 곱으로 압축하는 고차원 SVD (Higher-Order SVD).
  - CP 분해 (CANDECOMP/PARAFAC, 1970): 텐서를 최소 개수의 Rank-1 텐서들의 합으로 분해.


### 📌 3. 결측치 복원 (Matrix Completion)과 손실 압축
- 넷플릭스 추천 시스템처럼 빈칸(Missing values)이 많은 거대 행렬에서 저계수(Low-Rank) 가정을 통해 비어있는 값을 정밀 복원(Matrix Completion)하고 메모리를 획기적으로 절약합니다.


## 🧠 4. 4단계 정밀 개념 해설


### 1️⃣ [1단계 개념 정의]
- 행렬 저계수 근사 ($\hat{A}(k) = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$): SVD의 상위 $k$개 Rank-1 외적 성분만 취하여 원본 행렬을 최적의 $k$차원 부분공간으로 투영하는 손실 압축 기법입니다.
- 에카르트-영 정리 (Eckart-Young Theorem): 계수가 $k$인 모든 행렬 중 SVD로 만든 $\hat{A}(k)$ 가 원본과의 스펙트럼 노름 오차($\sigma_{k+1}$)를 최소화하는 유일한 최적해임을 보장하는 정리입니다.
- 행렬 계통도 (Matrix Phylogeny): 임의의 직사각형 행렬부터 단위 행렬까지 대수적 성질과 분해 기법의 포함 관계를 체계화한 분류도입니다.


### 2️⃣ [2단계 왜 쓰는가?]
- 초거대 데이터의 압축 및 계산 가속: $m \times n$ 원본을 $(m+n)k$ 로 기하급수적으로 축소하여 메모리와 곱셈 연산량을 대폭 절감합니다.
- 노이즈 필터링 및 잠재 구조 발견: 작은 특이값에 해당하는 고주파 노이즈를 날려버리고 지배적인 핵심 잠재 특징(Latent Features)만 추출하기 위해 사용합니다.


### 3️⃣ [3단계 상황별 직관 & Trade-off]
- 근사 차수 $k$ 의 결정 Trade-off:
  - $k$ 를 너무 작게 잡으면: 압축률은 극대화되지만 정보 손실 오차($\sigma_{k+1}$)가 커집니다.
  - $k$ 를 너무 크게 잡으면: 원본 복원율은 높아지지만 노이즈까지 보존되고 계산량이 증가합니다.
  - 실전 기준: 특이값의 누적 에너지 비율($\sum_{i=1}^k \sigma_i^2 / \sum_{i=1}^r \sigma_i^2 \ge 0.90 \sim 0.95$)을 기준으로 $k$ 를 결정합니다.


### 4️⃣ [4단계 실전 AI 연결고리]
- 거대 언어 모델(LLM) 가중치 압축 & LoRA: 트랜스포머 가중치 행렬 $W$ 를 에카르트-영 정리에 기반하여 $W \approx B A$ 로 저계수 분해함으로써 초경량 서빙 및 미세조정을 수행합니다.
- 이미지 및 비디오 노이즈 제거 (SVD Denoising): 이미지 패치 행렬의 저계수 근사를 통해 센서 노이즈를 완벽히 제거합니다.
- 매니폴드 학습 및 차원 축소 (PCA, Isomap, Laplacian Eigenmaps): 고차원 데이터의 비선형 다양체를 저차원 잠재 공간으로 임베딩하여 시각화 및 클러스터링을 수행합니다.
