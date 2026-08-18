# 📐 4.5 Singular Value Decomposition (특이값 분해, SVD)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 4.5 전수 분석 & 4단계 정밀 해설 노트


## 🌐 0. 선형대수학의 절대적 정점: 왜 "SVD(특이값 분해)"인가?

우리는 앞선 4.4절에서 정방행렬을 대각화하는 고유값 분해($A = PDP^{-1}$)를 배웠습니다.
하지만 고유값 분해에는 치명적인 3가지 한계가 존재합니다:
1. 오직 가로세로가 같은 정방행렬($n \times n$)에만 적용할 수 있습니다.
2. 정방행렬이라도 고유벡터가 부족한 결함 행렬(Defective Matrix)이면 분해가 불가능합니다.
3. 고유값이 음수이거나 복소수로 튀어나올 수 있습니다.

하지만 현실의 모든 머신러닝 데이터(사용자-아이템 평점 행렬, 이미지 픽셀 행렬, 단어-문서 행렬)는 대부분 직사각형 행렬($m \times n$)입니다.

특이값 분해(Singular Value Decomposition, SVD)는 "지구상에 존재하는 모든 임의의 직사각형 행렬에 대해 100% 예외 없이 존재하는 선형대수학의 기본 정리(Fundamental Theorem of Linear Algebra)"입니다!


## 1. ⚔️ Section 4.5: SVD의 정의와 3대 구성요소 (Theorem 4.22)


### 📌 1. SVD 정리 (SVD Theorem: Eq 4.64)

임의의 직사각형 행렬 $A \in \mathbb{R}^{m \times n}$ (계수 $r \le \min(m, n)$) 은 두 개의 직교 행렬 $U, V$ 와 하나의 직사각형 대각행렬 $\Sigma$ 의 곱으로 완벽히 분해됩니다:

$$A = U \Sigma V^\top \quad (\text{Eq 4.64})$$

1. 좌특이벡터 행렬 ($U \in \mathbb{R}^{m \times m}$):
   - $U^\top U = I_m$ 을 만족하는 직교 행렬(Orthogonal Matrix)입니다.
   - 열벡터 $\mathbf{u}_1, \dots, \mathbf{u}_m \in \mathbb{R}^m$ 을 좌특이벡터(Left-singular vectors)라 부르며, 공역 $\mathbb{R}^m$ 의 정규직교기저(ONB)를 형성합니다.
2. 우특이벡터 행렬 ($V \in \mathbb{R}^{n \times n}$):
   - $V^\top V = I_n$ 을 만족하는 직교 행렬(Orthogonal Matrix)입니다.
   - 열벡터 $\mathbf{v}_1, \dots, \mathbf{v}_n \in \mathbb{R}^n$ 을 우특이벡터(Right-singular vectors)라 부르며, 정의역 $\mathbb{R}^n$ 의 정규직교기저(ONB)를 형성합니다.
3. 특이값 행렬 ($\Sigma \in \mathbb{R}^{m \times n}$):
   - 행렬 $A$ 와 동일한 $m \times n$ 크기의 직사각형 대각 행렬입니다.
   - 주대각선 원소 $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$ 을 특이값(Singular Values)이라 부르며, 항상 0 이상의 실수(Non-negative reals)로 내림차순 정렬됩니다.
   - 나머지 $r+1$ 번째 이후 성분 및 비대각 성분은 모두 $0$ 으로 채워집니다 (Zero padding: Eq 4.65, 4.66).

- $m > n$ 일 때의 $\Sigma$ (행이 더 많은 경우: Eq 4.65):
  $$\Sigma = \begin{bmatrix} \sigma_1 & 0 & 0 \\\\ 0 & \ddots & 0 \\\\ 0 & 0 & \sigma_n \\\\ 0 & \dots & 0 \\\\ \vdots & \ddots & \vdots \\\\ 0 & \dots & 0 \end{bmatrix} \in \mathbb{R}^{m \times n}$$
- $m < n$ 일 때의 $\Sigma$ (열이 더 많은 경우: Eq 4.66):
  $$\Sigma = \begin{bmatrix} \sigma_1 & 0 & 0 & 0 & \dots & 0 \\\\ 0 & \ddots & 0 & \vdots & \ddots & \vdots \\\\ 0 & 0 & \sigma_m & 0 & \dots & 0 \end{bmatrix} \in \mathbb{R}^{m \times n}$$


## 2. ⚔️ Section 4.5.1: SVD의 기하학적 3단계 변환 직관 (Figure 4.8 & 4.9)

SVD는 정의역 $\mathbb{R}^n$ 의 단위 초구(Unit Sphere)를 공역 $\mathbb{R}^m$ 의 초타원체(Hyper-ellipsoid)로 사상하는 "회전 ➡️ 축 스케일링/차원 변환 ➡️ 회전"의 3단계 기하학적 변환입니다:

1. 1단계: $V^\top$ (정의역 내 회전 기저 변환):
   정의역 $\mathbb{R}^n$ 의 표준 기저를 우특이벡터 축 $\mathbf{v}_1, \dots, \mathbf{v}_n$ 방향으로 회전 정렬합니다.
2. 2단계: $\Sigma$ (독립된 축 팽창 및 차원 확장/축소):
   회전된 축들을 따라 각각의 특이값 $\sigma_i$ 배만큼 순수하게 늘리거나 줄이며, $n$차원 공간을 $m$차원 공간으로 임베딩(차원 변경)합니다. 단위 구가 $m$차원 타원체로 변형됩니다.
3. 3단계: $U$ (공역 내 최종 회전 기저 변환):
   팽창된 타원체를 공역 $\mathbb{R}^m$ 의 표준 좌표계에 맞추어 좌특이벡터 $\mathbf{u}_1, \dots, \mathbf{u}_m$ 축 방향으로 회전시킵니다.


### 💡 [Example 4.12: 2차원 평면에서 3차원 공간으로의 SVD 기하학적 사상 수치]
$A = \begin{bmatrix} 1 & -0.8 \\\\ 0 & 1 \\\\ 1 & 0 \end{bmatrix} \in \mathbb{R}^{3 \times 2}$ 의 SVD 수치 전개 (Eq 4.67):
$$A = U \Sigma V^\top = \begin{bmatrix} -0.79 & 0 & -0.62 \\\\ 0.38 & -0.78 & -0.49 \\\\ -0.48 & -0.62 & 0.62 \end{bmatrix} \begin{bmatrix} 1.62 & 0 \\\\ 0 & 1.0 \\\\ 0 & 0 \end{bmatrix} \begin{bmatrix} -0.78 & 0.62 \\\\ -0.62 & -0.78 \end{bmatrix}^\top$$
- 2차원 사각 격자점 집합 $X \in \mathbb{R}^2$ 에 $V^\top$ 을 적용하여 회전시킵니다.
- $\Sigma$ 를 통해 $x_1, x_2$ 축 방향으로 각각 1.62배, 1.0배 팽창시키며 3차원 공간 $\mathbb{R}^3$ ($x_3 = 0$ 평면)으로 매핑합니다.
- 마지막으로 $U$ 가 3차원 공간 내에서 타원 평면을 최종 회전시킵니다.


## 3. ⚔️ Section 4.5.2: SVD의 수학적 구성 및 유도 증명


### 📌 1. 우특이벡터 $V$ 와 특이값 $\sigma_i$ 의 도출 ($A^\top A$)

임의의 행렬 $A$ 에 대해 $A^\top A \in \mathbb{R}^{n \times n}$ 은 항상 대칭 반양의 정정 행렬(SPSD)입니다.
스펙트럴 정리(Theorem 4.15)에 의해 $A^\top A$ 를 직교 대각화하면:

$$A^\top A = (U \Sigma V^\top)^\top (U \Sigma V^\top) = V \Sigma^\top U^\top U \Sigma V^\top = V (\Sigma^\top \Sigma) V^\top \quad (\text{Eq 4.72~4.73})$$

- 우특이벡터 $V$: 대칭 행렬 $A^\top A$ 의 정규직교 고유벡터 행렬과 100% 일치합니다 ($V = P$).
- 특이값 $\sigma_i$: $A^\top A$ 의 고유값 $\lambda_i$ 의 양의 제곱근입니다:
  $$\sigma_i = \sqrt{\lambda_i(A^\top A)} \quad (\text{Eq 4.75})$$


### 📌 2. 좌특이벡터 $U$ 의 도출 ($AA^\top$)

마찬가지로 대칭 행렬 $AA^\top \in \mathbb{R}^{m \times m}$ 에 대해 SVD를 대입하면:

$$AA^\top = (U \Sigma V^\top)(U \Sigma V^\top)^\top = U (\Sigma \Sigma^\top) U^\top \quad (\text{Eq 4.76})$$

- 좌특이벡터 $U$: 대칭 행렬 $AA^\top$ 의 정규직교 고유벡터 행렬과 100% 일치합니다.


### 📌 3. 특이값 방정식과 두 기저의 완벽한 결합 (Eq 4.78~4.79)

우특이벡터 $\mathbf{v}_i$ 의 $A$ 에 의한 변환 상 $A\mathbf{v}_i$ 들은 서로 직교합니다:

$$(A\mathbf{v}_i)^\top (A\mathbf{v}_j) = \mathbf{v}_i^\top (A^\top A) \mathbf{v}_j = \mathbf{v}_i^\top (\lambda_j \mathbf{v}_j) = \lambda_j \mathbf{v}_i^\top \mathbf{v}_j = 0 \quad (i \neq j)$$

이 직교 벡터 $A\mathbf{v}_i$ 를 단위 길이로 정규화한 것이 바로 좌특이벡터 $\mathbf{u}_i$ 가 됩니다:

$$\mathbf{u}_i := \frac{A\mathbf{v}_i}{\VertA\mathbf{v}_i\Vert} = \frac{1}{\sqrt{\lambda_i}} A\mathbf{v}_i = \frac{1}{\sigma_i} A\mathbf{v}_i \quad (\text{Eq 4.78})$$

이 식을 정리하면 선형대수학의 황금 방정식인 특이값 방정식(Singular Value Equation)이 완성됩니다:

$$A \mathbf{v}_i = \sigma_i \mathbf{u}_i \quad (i = 1, \dots, r) \quad (\text{Eq 4.79})$$

이를 행렬 형태로 묶으면 $A V = U \Sigma$ 가 되며, 우변에 $V^\top$ 을 곱하면 $A = U \Sigma V^\top$ 가 완성됩니다!


### 📌 4. SVD와 4대 기본 부분공간 / 영공간(Kernel)의 완벽한 일치

SVD는 행렬 $A$ 의 4대 기본 부분공간(Fundamental Subspaces)의 정규직교기저(ONB)를 동시에 완벽하게 제공합니다:
1. 열공간(Column Space / $\text{Im}(A)$): 처음 $r$개의 좌특이벡터 $\{\mathbf{u}_1, \dots, \mathbf{u}_r\}$ 가 ONB를 형성합니다.
2. 행공간(Row Space): 처음 $r$개의 우특이벡터 $\{\mathbf{v}_1, \dots, \mathbf{v}_r\}$ 가 ONB를 형성합니다.
3. 영공간(Null Space / $\ker(A)$): 나머지 $n-r$개의 우특이벡터 $\{\mathbf{v}_{r+1}, \dots, \mathbf{v}_n\}$ 가 $\ker(A)$ 의 ONB를 형성합니다 ($A\mathbf{v}_i = \mathbf{0}$).
4. 좌영공간(Left Null Space / $\ker(A^\top)$): 나머지 $m-r$개의 좌특이벡터 $\{\mathbf{u}_{r+1}, \dots, \mathbf{u}_m\}$ 가 $\ker(A^\top)$ 의 ONB를 형성합니다.


## 4. ⚔️ SVD 손풀기 수치 계산 전수 분석 (Example 4.13)

행렬 $A = \begin{bmatrix} 1 & 0 & 1 \\\\ -2 & 1 & 0 \end{bmatrix} \in \mathbb{R}^{2 \times 3}$ 의 SVD 도출 과정:

1. 1단계: $A^\top A$ 계산 및 고유값/고유벡터(우특이벡터 $V$) 도출:
   $$A^\top A = \begin{bmatrix} 1 & -2 \\\\ 0 & 1 \\\\ 1 & 0 \end{bmatrix} \begin{bmatrix} 1 & 0 & 1 \\\\ -2 & 1 & 0 \end{bmatrix} = \begin{bmatrix} 5 & -2 & 1 \\\\ -2 & 1 & 0 \\\\ 1 & 0 & 1 \end{bmatrix}$$
   - 고유값: $\lambda_1 = 6, \; \lambda_2 = 1, \; \lambda_3 = 0$
   - 우특이벡터 ($V$ 의 열들):
     $$\mathbf{v}_1 = \frac{1}{\sqrt{30}}\begin{bmatrix} 5 \\\\ -2 \\\\ 1 \end{bmatrix}, \quad \mathbf{v}_2 = \frac{1}{\sqrt{5}}\begin{bmatrix} 0 \\\\ 1 \\\\ 2 \end{bmatrix}, \quad \mathbf{v}_3 = \frac{1}{\sqrt{6}}\begin{bmatrix} -1 \\\\ -2 \\\\ 1 \end{bmatrix}$$

2. 2단계: 특이값 행렬 $\Sigma$ 구축:
   - $\sigma_1 = \sqrt{\lambda_1} = \sqrt{6}, \quad \sigma_2 = \sqrt{\lambda_2} = 1$
   - $\Sigma = \begin{bmatrix} \sqrt{6} & 0 & 0 \\\\ 0 & 1 & 0 \end{bmatrix} \in \mathbb{R}^{2 \times 3}$

3. 3단계: 좌특이벡터 $U$ 계산 ($\mathbf{u}_i = \frac{1}{\sigma_i} A\mathbf{v}_i$):
   $$\mathbf{u}_1 = \frac{1}{\sqrt{6}} A \mathbf{v}_1 = \frac{1}{\sqrt{6}} \begin{bmatrix} 1 & 0 & 1 \\\\ -2 & 1 & 0 \end{bmatrix} \frac{1}{\sqrt{30}}\begin{bmatrix} 5 \\\\ -2 \\\\ 1 \end{bmatrix} = \frac{1}{\sqrt{5}}\begin{bmatrix} 1 \\\\ -2 \end{bmatrix}$$
   $$\mathbf{u}_2 = \frac{1}{1} A \mathbf{v}_2 = \frac{1}{1} \begin{bmatrix} 1 & 0 & 1 \\\\ -2 & 1 & 0 \end{bmatrix} \frac{1}{\sqrt{5}}\begin{bmatrix} 0 \\\\ 1 \\\\ 2 \end{bmatrix} = \frac{1}{\sqrt{5}}\begin{bmatrix} 2 \\\\ 1 \end{bmatrix}$$
   $$U = [\mathbf{u}_1, \mathbf{u}_2] = \frac{1}{\sqrt{5}}\begin{bmatrix} 1 & 2 \\\\ -2 & 1 \end{bmatrix} \in \mathbb{R}^{2 \times 2}$$


## 5. ⚔️ 고유값 분해 vs SVD 전면 비교 & SVD의 3가지 형태 (4.5.3절)


### 📌 1. Eigendecomposition vs SVD 6대 차이점 완벽 비교

| 비교 항목 | 고유값 분해 (Eigendecomposition $A = PDP^{-1}$) | 특이값 분해 (SVD $A = U\Sigma V^\top$) |
| :--- | :--- | :--- |
| 적용 행렬 형태 | 오직 정방행렬 ($n \times n$) 한정 | 모든 직사각형 행렬 ($m \times n$) 100% 가능 |
| 존재성 보장 | 비결함 행렬만 가능 (결함 행렬은 분해 불가) | 모든 행렬에 대해 항상 100% 존재 보장 |
| 기저 행렬의 성질 | $P$ 는 직교 행렬이 아닐 수 있음 ($P^{-1} \neq P^\top$) | $U, V$ 는 무조건 완벽한 정규직교 행렬(ONB) |
| 대각 성분의 값 | 고유값 $\lambda_i$ 는 음수, 복소수 가능 | 특이값 $\sigma_i$ 는 항상 0 이상의 실수 ($\sigma_i \ge 0$) |
| 작동 공간 | 동일한 벡터 공간 ($\mathbb{R}^n \to \mathbb{R}^n$) | 서로 다른 두 벡터 공간 ($\mathbb{R}^n \to \mathbb{R}^m$) |
| 대칭 행렬의 경우 | $A = PDP^\top$ (스펙트럴 정리) | $U = P = V, \Sigma = D$ 로 고유값 분해와 완전 일치 |


### 📌 2. SVD의 3대 표현 형태

1. Full SVD (Eq 4.64): $U \in \mathbb{R}^{m \times m}, \; \Sigma \in \mathbb{R}^{m \times n}, \; V \in \mathbb{R}^{n \times n}$.
2. Reduced SVD (Thin SVD: Eq 4.89): $m \ge n$ 일 때 0으로 채워진 불필요한 행들을 잘라내어 $U \in \mathbb{R}^{m \times n}, \; \Sigma \in \mathbb{R}^{n \times n}, \; V \in \mathbb{R}^{n \times n}$ 로 표현.
3. Truncated SVD (저계수 근사: Section 4.6): 상위 $k$개의 특이값만 남겨 $U_k \in \mathbb{R}^{m \times k}, \; \Sigma_k \in \mathbb{R}^{k \times k}, \; V_k^\top \in \mathbb{R}^{k \times n}$ 로 데이터 압축.


### 💡 [Example 4.14: 넷플릭스 영화 평점 데이터 SVD 전수 수치 분석]
3명의 사용자(Ali, Beatrix, Chandra)가 4편의 영화(Star Wars, Blade Runner, Amelie, Delicatessen)를 평가한 평점 행렬 $A \in \mathbb{R}^{4 \times 3}$:
$$A = U \Sigma V^\top$$
$$\begin{bmatrix} -0.6710 & 0.0236 & 0.4647 & -0.5774 \\\\ -0.7197 & 0.2054 & -0.4759 & 0.4619 \\\\ -0.0939 & -0.7705 & -0.5268 & -0.3464 \\\\ -0.1515 & -0.6030 & 0.5293 & -0.5774 \end{bmatrix} \begin{bmatrix} 9.6438 & 0 & 0 \\\\ 0 & 6.3639 & 0 \\\\ 0 & 0 & 0.7056 \\\\ 0 & 0 & 0 \end{bmatrix} \begin{bmatrix} -0.7367 & -0.6515 & -0.1811 \\\\ 0.0852 & 0.1762 & -0.9807 \\\\ 0.6708 & -0.7379 & -0.0743 \end{bmatrix}^\top$$

- 1번 잠재 테마 (SF 영화 테마: $\sigma_1 = 9.6438$):
  - 좌특이벡터 $\mathbf{u}_1$: Star Wars($-0.6710$)와 Blade Runner($-0.7197$)에 절대값이 집중됨 (SF 장르 축).
  - 우특이벡터 $\mathbf{v}_1$: SF 영화에 높은 평점을 준 Ali($-0.7367$)와 Beatrix($-0.6515$)에 절대값이 집중됨 (SF 매니아 축).
- 2번 잠재 테마 (프랑스 예술영화 테마: $\sigma_2 = 6.3639$):
  - 좌특이벡터 $\mathbf{u}_2$: Amelie($-0.7705$)와 Delicatessen($-0.6030$)에 집중됨.
  - 우특이벡터 $\mathbf{v}_2$: 해당 영화를 선호한 Chandra($-0.9807$)에 집중됨.
- 해석: 각 영화와 사용자는 이 직교하는 잠재 테마 축들의 선형결합으로 완벽하게 분해됩니다!


## 🧠 6. 4단계 정밀 개념 해설


### 1️⃣ [1단계 개념 정의]
- 특이값 분해 (SVD): 임의의 직사각형 행렬을 좌특이벡터 직교행렬 $U$, 특이값 대각행렬 $\Sigma$, 우특이벡터 직교행렬 $V^\top$ 의 세 행렬 곱으로 쪼개는 선형대수학의 기본 정리입니다.
- 특이값 ($\sigma_i$): 행렬 변환에 의해 각 직교 주축 방향으로 늘어나는 순수 팽창 스케일(크기)입니다.


### 2️⃣ [2단계 왜 쓰는가?]
- 모든 직사각형 데이터의 일반화된 분해: 정방행렬에 갇히지 않고 임의의 $m \times n$ 데이터 행렬 전체를 직교 축들로 해체하기 위해 사용합니다.
- 데이터의 최적 저계수(Low-Rank) 압축: 가장 에너지가 큰 특이값 몇 개만 남김으로써 노이즈를 제거하고 핵심 정보만 추출하기 위해 사용합니다.


### 3️⃣ [3단계 상황별 직관 & Trade-off]
- Full SVD vs Truncated SVD:
  - Full SVD: 원본 행렬을 100% 손실 없이 복원하지만 거대한 행렬 크기를 그대로 유지합니다.
  - Truncated SVD: 상위 $k$개 특이값만 유지하여 약간의 오차가 발생하지만, 데이터 용량을 $O(k(m+n))$ 으로 기하급수적으로 줄이고 핵심 잠재 의미(Latent Semantics)만 남깁니다.


### 4️⃣ [4단계 실전 AI 연결고리]
- 추천 시스템 (Netflix Matrix Factorization / Example 4.14):
  사용자-영화 평점 행렬 $A \in \mathbb{R}^{4 \times 3}$ 에 SVD를 적용하여 좌특이벡터 $\mathbf{u}_i$ 는 영화 장르 테마(SF vs 예술영화), 우특이벡터 $\mathbf{v}_j$ 는 사용자 취향 페르소나, 특이값 $\sigma_i$ 는 장르 선호 가중치로 분해하여 비어있는 평점을 정확히 예측합니다.
- 자연어 처리 잠재 의미 분석 (Latent Semantic Analysis - LSA):
  단어-문서 행렬(Term-Document Matrix)을 SVD로 분해하여 동의어와 다의어 속에서 핵심 주제(Topic) 축을 추출합니다.
- 의사역행렬 (Moore-Penrose Pseudoinverse $A^+$):
  비정방행렬의 역행렬을 $A^+ = V \Sigma^+ U^\top$ ($\Sigma^+$ 는 0이 아닌 특이값의 역수 $1/\sigma_i$) 로 단번에 계산하여 선형 회귀 최소제곱해를 구합니다.
- 이미지 압축 및 잡음 제거 (Denoising):
  이미지 픽셀 행렬에서 상위 $k$개의 특이값만 남기고 작은 특이값을 $0$ 으로 날려 고화질 압축 및 노이즈 제거를 수행합니다.
