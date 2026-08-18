# 📐 4.2 Eigenvalues and Eigenvectors (고유값과 고유벡터, 스펙트럴 정리)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 4.2 전수 분석 & 4단계 정밀 해설 노트


## 🌐 0. 4.1절과의 연결 및 자연스러운 빌드업: 왜 "고유값과 고유벡터"를 배우는가?

우리는 지난 4.1절에서 행렬의 부피 팽창 배율인 행렬식($\det(A)$)과 대각합($\text{tr}(A)$), 그리고 이들을 계수로 품은 특성 다항식 $p_A(\lambda) = \det(A - \lambda I)$ 을 정의했습니다.

일반적인 행렬 변환은 공간의 벡터들을 회전시키고 길이를 바꾸며 방향을 마구 뒤틀어 놓습니다.
하지만 놀랍게도 어떤 행렬이든 "변환을 가해도 방향이 전혀 꺾이지 않고 오직 순수하게 제자리에서 늘어나거나 줄어들기만 하는 고유한 불변의 주축"들이 존재합니다.

이 특별한 불변의 방향 축을 고유벡터(Eigenvector)라 부르고, 그 축 방향으로 늘어나는 팽창 배율을 고유값(Eigenvalue)이라 부릅니다.

고유값과 고유벡터를 파악하면, 복잡하게 뒤엉킨 다차원 행렬 변환을 서로 간섭하지 않는 독립적인 1차원 스케일링들의 결합으로 완벽하게 해체(분해)할 수 있습니다.


## 1. ⚔️ Section 4.2: Eigenvalues and Eigenvectors (고유값과 고유벡터의 정의)


### 📌 1. 고유값 방정식 (Eigenvalue Equation: Definition 4.6 & Eq 4.25)

정방행렬 $A \in \mathbb{R}^{n \times n}$ 에 대해, 0이 아닌 벡터 $\mathbf{x} \in \mathbb{R}^n \setminus \{\mathbf{0}\}$ 와 스칼라 $\lambda \in \mathbb{R}$ 가 다음 방정식을 만족할 때 $\lambda$ 를 $A$ 의 고유값(Eigenvalue), $\mathbf{x}$ 를 그에 대응하는 고유벡터(Eigenvector)라고 정의합니다:

$$A \mathbf{x} = \lambda \mathbf{x} \quad (\text{Eq 4.25})$$

- 고유값의 4대 동치 조건:
  1. $\lambda$ 는 행렬 $A$ 의 고유값이다.
  2. $(A - \lambda I)\mathbf{x} = \mathbf{0}$ 이 영벡터가 아닌 비자명해($\mathbf{x} \neq \mathbf{0}$)를 갖는다.
  3. 행렬 $A - \lambda I$ 의 계수가 완전계수 미만이다 ($\text{rk}(A - \lambda I) < n$).
  4. 행렬식 값이 0이다 ($\det(A - \lambda I) = 0$).

- 고유벡터의 비유일성과 공선성(Collinearity: Eq 4.26):
  만약 $\mathbf{x}$ 가 고유값 $\lambda$ 의 고유벡터라면, 0이 아닌 임의의 실수 $c \neq 0$ 에 대해 $c\mathbf{x}$ 역시 동일한 고유값의 고유벡터입니다 ($A(c\mathbf{x}) = c A\mathbf{x} = c\lambda\mathbf{x} = \lambda(c\mathbf{x})$). 즉, 고유벡터는 특정 벡터 하나가 아니라 하나의 방향선(Line) 전체를 의미합니다.


### 📌 2. 고유공간, 스펙트럼 및 중복도 (Definitions 4.9 ~ 4.11)

1. 고유공간 (Eigenspace $E_\lambda$: Definition 4.10 & Eq 4.27):
   특정 고유값 $\lambda$ 에 대응하는 모든 고유벡터들과 영벡터를 모아놓은 부분공간이며, $A - \lambda I$ 의 영공간(Null Space / Kernel)과 완벽히 일치합니다:
   $$E_\lambda = \ker(A - \lambda I) = \{\mathbf{x} \in \mathbb{R}^n \mid (A - \lambda I)\mathbf{x} = \mathbf{0}\}$$

2. 고유스펙트럼 (Eigenspectrum / Spectrum):
   행렬 $A$ 가 가지는 모든 고유값들의 집합입니다.

3. 대수적 중복도 vs 기하학적 중복도 (Algebraic vs Geometric Multiplicity):
   - 대수적 중복도(Algebraic Multiplicity: Definition 4.9): 특성 다항식 $p_A(\lambda) = 0$ 에서 해당 고유값 근이 몇 번 중복되어 나타나는가의 횟수.
   - 기하학적 중복도(Geometric Multiplicity: Definition 4.11): 고유값 $\lambda$ 에 대응하는 고유공간의 차원 ($\dim(E_\lambda)$), 즉 해당 고유값에 연결된 선형독립인 고유벡터의 최대 개수.
   - 대원칙 부등식:
     $$1 \le \text{기하학적 중복도} \le \text{대수적 중복도}$$


### 📌 3. 고유값과 고유벡터의 핵심 대수적 성질 및 역행렬 관계

1. 역행렬($A^{-1}$)의 고유값과 고유벡터 (가장 중요한 성질!):
   가역 행렬 $A$ 의 고유값이 $\lambda$ 이고 고유벡터가 $\mathbf{v}$ 라면 ($A\mathbf{v} = \lambda\mathbf{v}$):
   - $A^{-1}$ 의 고유벡터는 원래와 완전히 동일한 $\mathbf{v}$ 입니다.
   - $A^{-1}$ 의 고유값은 역수인 $\frac{1}{\lambda}$ 이 됩니다.
   $$A \mathbf{v} = \lambda \mathbf{v} \iff A^{-1} \mathbf{v} = \frac{1}{\lambda} \mathbf{v}$$
   - 기하학적 직관: 원래 행렬 $A$ 가 특정 축 $\mathbf{v}$ 방향으로 공간을 $\lambda$ 배 늘렸다면, 역행렬 $A^{-1}$ 은 완벽한 원상복구를 위해 똑같은 축 $\mathbf{v}$ 방향을 $\frac{1}{\lambda}$ 배로 줄여야 하므로 축(고유벡터)은 불변이고 고유값만 역수가 됩니다.

2. 행렬의 거듭제곱($A^k$)과 다항식 행렬 $f(A)$:
   임의의 정수 $k$ 에 대해 $A^k \mathbf{v} = \lambda^k \mathbf{v}$ 가 성립합니다 (역행렬은 $k = -1$ 인 특수한 경우).
   - 행렬 다항식: $f(A) = c_m A^m + \dots + c_0 I \implies f(A)\mathbf{v} = f(\lambda)\mathbf{v}$.

3. 전치 행렬과의 관계: 행렬 $A$ 와 전치 행렬 $A^\top$ 은 고유값이 100% 동일합니다 ($\det(A^\top - \lambda I) = \det((A - \lambda I)^\top) = \det(A - \lambda I)$). 단, 고유벡터는 서로 다를 수 있습니다.

4. 기저 변환 불변성: 유사 행렬 $B = S^{-1}AS$ 는 $A$ 와 완전히 동일한 고유값 집합을 가집니다.

5. 대칭 양의 정정 행렬(SPD): 대칭 양의 정정 행렬은 모든 고유값이 항상 엄밀한 양의 실수($\lambda_i > 0$)입니다.

6. 단위행렬(Identity Matrix: Example 4.4): $I \in \mathbb{R}^{n \times n}$ 은 고유값 $\lambda = 1$ 이 $n$번 중복되며, 모든 $n$개의 표준기저 벡터가 고유벡터가 되어 $E_1 = \mathbb{R}^n$ (대수적 중복도 $n$ = 기하학적 중복도 $n$)이 됩니다.


## 2. ⚔️ 고유값 및 고유공간 손풀기 계산 전수 분석 (Example 4.5)


### 📌 2x2 행렬 $A = \begin{bmatrix} 4 & 2 \\\\ 1 & 3 \end{bmatrix}$ 의 3단계 풀이

1. 1단계: 특성 다항식 구축 및 근(고유값) 계산 (Eq 4.29~4.30)
   $$p_A(\lambda) = \det(A - \lambda I) = \det\begin{bmatrix} 4-\lambda & 2 \\\\ 1 & 3-\lambda \end{bmatrix} = (4-\lambda)(3-\lambda) - 2 = \lambda^2 - 7\lambda + 10 = (\lambda - 2)(\lambda - 5) = 0$$
   - 고유값: $\lambda_1 = 5, \quad \lambda_2 = 2$

2. 2단계: 첫 번째 고유값 $\lambda_1 = 5$ 의 고유벡터 및 고유공간 $E_5$ 도출 (Eq 4.31~4.33)
   $$(A - 5I)\mathbf{x} = \begin{bmatrix} 4-5 & 2 \\\\ 1 & 3-5 \end{bmatrix} \begin{bmatrix} x_1 \\\\ x_2 \end{bmatrix} = \begin{bmatrix} -1 & 2 \\\\ 1 & -2 \end{bmatrix} \begin{bmatrix} x_1 \\\\ x_2 \end{bmatrix} = \begin{bmatrix} 0 \\\\ 0 \end{bmatrix}$$
   - $-x_1 + 2x_2 = 0 \implies x_1 = 2x_2$.
   - 고유벡터: $\mathbf{x}_1 = \begin{bmatrix} 2 \\\\ 1 \end{bmatrix}$, 고유공간: $E_5 = \text{span}\left(\begin{bmatrix} 2 \\\\ 1 \end{bmatrix}\right)$ ($\dim(E_5) = 1$).

3. 3단계: 두 번째 고유값 $\lambda_2 = 2$ 의 고유벡터 및 고유공간 $E_2$ 도출 (Eq 4.34~4.35)
   $$(A - 2I)\mathbf{x} = \begin{bmatrix} 4-2 & 2 \\\\ 1 & 3-2 \end{bmatrix} \begin{bmatrix} x_1 \\\\ x_2 \end{bmatrix} = \begin{bmatrix} 2 & 2 \\\\ 1 & 1 \end{bmatrix} \begin{bmatrix} x_1 \\\\ x_2 \end{bmatrix} = \begin{bmatrix} 0 \\\\ 0 \end{bmatrix}$$
   - $x_1 + x_2 = 0 \implies x_1 = -x_2$.
   - 고유벡터: $\mathbf{x}_2 = \begin{bmatrix} 1 \\\\ -1 \end{bmatrix}$, 고유공간: $E_2 = \text{span}\left(\begin{bmatrix} 1 \\\\ -1 \end{bmatrix}\right)$ ($\dim(E_2) = 1$).


## 3. ⚔️ 결함 행렬과 2차원 선형 사상 기하학적 직관 (Figure 4.4)


### 📌 1. 결함 행렬 (Defective Matrix: Definition 4.13 & Example 4.6)

정방행렬 $A \in \mathbb{R}^{n \times n}$ 이 $n$개의 선형독립인 고유벡터를 가지지 못할 때(즉 고유벡터들로 공간의 기저를 만들 수 없을 때) 이를 결함 행렬(Defective Matrix)이라 부릅니다.
결함 행렬은 적어도 하나의 고유값에서 기하학적 중복도 < 대수적 중복도 인 현상이 발생합니다.

- Example 4.6 수치 분석:
  $A = \begin{bmatrix} 2 & 1 \\\\ 0 & 2 \end{bmatrix}$ 은 특성 다항식이 $(\lambda - 2)^2 = 0$ 이므로 고유값 $\lambda = 2$ 의 대수적 중복도는 2입니다.
  그러나 $(A - 2I)\mathbf{x} = \begin{bmatrix} 0 & 1 \\\\ 0 & 0 \end{bmatrix} \begin{bmatrix} x_1 \\\\ x_2 \end{bmatrix} = \begin{bmatrix} 0 \\\\ 0 \end{bmatrix}$ 을 풀면 $x_2 = 0$ 이 되어 고유벡터는 오직 $\mathbf{x} = \begin{bmatrix} 1 \\\\ 0 \end{bmatrix}$ 1개뿐입니다.
  따라서 $\dim(E_2) = 1 < 2$ 이므로 $A$ 는 결함 행렬입니다.

- 서로 다른 고유값과 기저 (Theorem 4.12):
  행렬 $A \in \mathbb{R}^{n \times n}$ 이 서로 다른 $n$개의 고유값을 가지면, 그에 대응하는 $n$개의 고유벡터들은 무조건 선형독립이며 $\mathbb{R}^n$ 의 기저를 형성합니다.


### 📌 2. 2차원 5대 선형 사상의 기하학적 시각화 (Figure 4.4 전수 분석)

1. 축 스케일링 ($A_1 = \begin{bmatrix} 1/2 & 0 \\\\ 0 & 2 \end{bmatrix}$):
   - $y$축 방향으로 2배 팽창($\lambda_1 = 2$), $x$축 방향으로 $1/2$배 압축($\lambda_2 = 0.5$). 면적 보존 ($\det(A_1) = 1$).
2. 전단 변환 (Shearing: $A_2 = \begin{bmatrix} 1 & 1/2 \\\\ 0 & 1 \end{bmatrix}$):
   - $\lambda_1 = \lambda_2 = 1$ 중근을 가지며 수평축 방향으로만 기울임 작용. 면적 보존 ($\det(A_2) = 1$).
3. 회전 변환 (Rotation $30^\circ$: $A_3 = R(30^\circ)$):
   - 복소수 고유값($\cos 30^\circ \pm i\sin 30^\circ$)을 가지며, 방향이 유지되는 실수 고유벡터가 존재하지 않음. 부피 보존 ($\det(A_3) = 1$).
4. 1차원 붕괴 사영 ($A_4 = \begin{bmatrix} 1 & -1 \\\\ -1 & 1 \end{bmatrix}$):
   - $\lambda_1 = 0, \lambda_2 = 2$. 고유값 0 방향의 공간이 완전히 찌그러져 납작한 1차원 직선으로 붕괴. 면적 0 ($\det(A_4) = 0$).
5. 전단 및 확장 ($A_5 = \begin{bmatrix} 1 & 1/2 \\\\ 1/2 & 1 \end{bmatrix}$):
   - 대칭 행렬로서 직교하는 두 고유벡터 축을 따라 각각 1.5배 팽창($\lambda_2 = 1.5$), 0.5배 압축($\lambda_1 = 0.5$). 면적 $75\%$ 축소 ($\det(A_5) = 0.75$).


## 4. ⚔️ 스펙트럴 정리와 행렬식/대각합 연결 (Spectral Theorem)


### 📌 1. 스펙트럴 정리 (Spectral Theorem: Theorems 4.14 ~ 4.15)

1. $A^\top A$ 의 대칭 반양의 정정성 (Theorem 4.14 & Eq 4.36):
   임의의 행렬 $A \in \mathbb{R}^{m \times n}$ 에 대해 $S = A^\top A$ 는 항상 대칭 반양의 정정 행렬(SPSD)이며, $\text{rk}(A) = n$ 이면 대칭 양의 정정 행렬(SPD)이 됩니다.
   - 증명: $S^\top = (A^\top A)^\top = A^\top A = S$, 그리고 $\mathbf{x}^\top S \mathbf{x} = \mathbf{x}^\top A^\top A \mathbf{x} = \|A\mathbf{x}\|^2 \ge 0$.

2. 스펙트럴 정리 (Spectral Theorem: Theorem 4.15):
   실수 대칭 행렬 $A \in \mathbb{R}^{n \times n}$ ($A^\top = A$) 은:
   - 모든 고유값이 항상 실수(Real numbers)입니다.
   - 고유벡터들로 구성된 완벽한 정규직교기저(ONB)가 반드시 존재합니다!
   - 따라서 $A$ 는 직교 행렬 $P$ ($P^\top = P^{-1}$) 와 대각 행렬 $D$ 에 의해 $A = P D P^\top$ 로 완벽히 직교 대각화 분해됩니다.


#### 💡 [Example 4.8: 대칭 행렬의 직교 고유기저 그람-슈미트 구축]
$A = \begin{bmatrix} 3 & 2 & 2 \\\\ 2 & 3 & 2 \\\\ 2 & 2 & 3 \end{bmatrix}$
- 특성 다항식: $p_A(\lambda) = -(\lambda - 1)^2 (\lambda - 7) = 0 \implies \lambda_1 = 1$ (중복도 2), $\lambda_2 = 7$.
- 고유공간: $E_1 = \text{span}\left(\begin{bmatrix} -1 \\\\ 1 \\\\ 0 \end{bmatrix}, \begin{bmatrix} -1 \\\\ 0 \\\\ 1 \end{bmatrix}\right), \quad E_7 = \text{span}\left(\begin{bmatrix} 1 \\\\ 1 \\\\ 1 \end{bmatrix}\right)$
- $E_1$ 의 두 기저는 서로 직교하지 않으므로, 동일 고유공간 내 선형결합에 그람-슈미트를 적용하여 상호 수직인 직교 고유기저를 완성합니다:
  $$\mathbf{x}_1' = \begin{bmatrix} -1 \\\\ 1 \\\\ 0 \end{bmatrix}, \quad \mathbf{x}_2' = \frac{1}{2}\begin{bmatrix} -1 \\\\ -1 \\\\ 2 \end{bmatrix}$$


### 📌 2. 행렬식 & 대각합과 고유값의 기하학적 일치 (Theorems 4.16 ~ 4.17 & Figure 4.6)

1. 행렬식 = 고유값들의 총 곱 (Theorem 4.16 & Eq 4.42):
   $$\det(A) = \prod_{i=1}^n \lambda_i = \lambda_1 \lambda_2 \dots \lambda_n$$
   - 기하학적 의미: $n$차원 단위 초입방체가 변환될 때 변화하는 부피 팽창 배율 $|\lambda_1 \dots \lambda_n|$ 과 일치합니다.

2. 대각합 = 고유값들의 총 합 (Theorem 4.17 & Eq 4.43):
   $$\text{tr}(A) = \sum_{i=1}^n \lambda_i = \lambda_1 + \lambda_2 + \dots + \lambda_n$$
   - 기하학적 의미: 단위 도형이 변환될 때 변화하는 둘레(Perimeter) 길이의 변동 비율을 나타냅니다.


## 🧠 5. 4단계 정밀 개념 해설


### 1️⃣ [1단계 개념 정의]
- 고유벡터 ($\mathbf{x}$): 선형 변환을 가해도 회전하지 않고 방향을 유지하는 불변의 주축 벡터입니다.
- 고유값 ($\lambda$): 고유벡터 축 방향으로 공간이 늘어나거나 줄어드는 스케일링 팽창 배율입니다.
- 스펙트럴 정리 (Spectral Theorem): 대칭 행렬은 무조건 실수 고유값을 가지며 정규직교 고유벡터들로 분해($A = P D P^\top$)된다는 정리입니다.


### 2️⃣ [2단계 왜 쓰는가?]
- 다차원 연립방정식 및 행렬 거듭제곱의 분해: 복잡하게 얽힌 다변수 시스템을 독립된 1차원 축들의 단순 스케일링 문제로 대각화 디커플링(Decoupling)하기 위해 사용합니다.
- 데이터의 주요 변동성 축 추출: 데이터 공분산 행렬에서 가장 분산이 큰 방향을 찾아내기 위해 사용합니다.


### 3️⃣ [3단계 상황별 직관 & Trade-off]
- 대칭 행렬 vs 비대칭 행렬:
  - 대칭 행렬: 고유값이 100% 실수이고 고유벡터들이 서로 90도 직교하여 아름다운 회전-스케일링 분해가 가능합니다.
  - 비대칭 행렬: 복소수 고유값이 발생하거나 고유벡터가 부족한 결함 행렬(Defective Matrix)이 될 수 있어 분해가 불안정합니다.


### 4️⃣ [4단계 실전 AI 연결고리]
- 구글 페이지랭크 (Google PageRank: Example 4.9): 웹페이지 전이 확률 행렬 $A$ 의 최대 고유값 $\lambda = 1$ 에 대응하는 정상 상태 고유벡터 $\mathbf{x}^*$ ($A\mathbf{x}^* = \mathbf{x}^*$) 를 계산하여 전 세계 웹사이트의 중요도 순위를 매깁니다.
- PCA (주성분 분석 - Ch 10): 데이터 공분산 행렬 $\Sigma = \frac{1}{N}X^\top X$ 에 스펙트럴 정리를 적용하여 가장 큰 고유값에 대응하는 고유벡터 순으로 주성분 기저를 축출합니다.
- 생물학 신경망 및 그래프 신경망 (GNN: Example 4.7): 그래프 인접 행렬의 고유스펙트럼(Eigenspectrum)을 분석하여 신경망의 연결 구조와 커뮤니티 특성을 파악합니다.
- Hessian 행렬과 딥러닝 손실함수 곡률: 손실함수의 2차 미분 헤시안 행렬 $H$ 의 고유값이 모두 양수이면 국소 최솟값(Local Minimum)을 보장하며, 최대 고유값은 학습률(Learning Rate) 상한선을 결정합니다.
