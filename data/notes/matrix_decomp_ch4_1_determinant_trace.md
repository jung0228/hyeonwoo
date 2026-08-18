# 📐 4.0 & 4.1 Matrix Decompositions, Determinant and Trace (행렬 분해의 서막, 행렬식과 대각합)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Chapter 4 & Section 4.1 전수 분석 & 4단계 정밀 해설 노트


## 🌐 0. Chapter 4의 서막: 왜 "행렬 분해(Matrix Decomposition)"를 배우는가?

우리는 지난 Chapter 2와 3에서 벡터, 선형 사상, 내적, 정사영, 그리고 회전 변환을 공부했습니다.
데이터 과학과 머신러닝에서 대부분의 데이터는 행렬(Matrix) 형태로 표현됩니다 (예: 행은 사용자, 열은 키, 몸무게, 소득 등의 특성).

Chapter 4 행렬 분해(Matrix Decompositions / Matrix Factorization)는 마치 복잡한 자연수를 소인수분해($21 = 7 \times 3$)하듯, 거대하고 분석하기 어려운 행렬을 기하학적 의미가 투명하고 해석하기 쉬운 기본 행렬들의 곱으로 분해하는 선형대수학의 정점입니다.

4장은 다음과 같은 거대한 여정으로 구성됩니다:
1. 행렬을 단 몇 개의 숫자로 특성화하는 요약 도구: 행렬식(Determinant, 4.1절)과 고유값(Eigenvalues, 4.2절).
2. 대칭 양의 정정 행렬의 제곱근 분해: 숄레스키 분해(Cholesky Decomposition, 4.3절).
3. 직교 기저를 통한 완벽한 축 정렬: 행렬 대각화(Matrix Diagonalization, 4.4절).
4. 모든 임의의 직사각형 행렬을 분해하는 선형대수학의 꽃: 특이값 분해(SVD, 4.5절).
5. 행렬들의 속성과 계통을 한눈에 정리하는 분류 체계(Matrix Taxonomy, 4.7절).


## 1. ⚔️ Section 4.1: Determinant (행렬식)


### 📌 1. 행렬식의 수학적 정의와 가역성 판별 (Definition & Theorem 4.1)

행렬식(Determinant)은 정방행렬 $A \in \mathbb{R}^{n \times n}$ 을 입력받아 하나의 스칼라 실수 $\det(A)$ (또는 $|A|$) 로 매핑하는 함수입니다 (Eq 4.1).

- 가역성 판별의 대원칙 (Theorem 4.1):
  정방행렬 $A \in \mathbb{R}^{n \times n}$ 의 역행렬 $A^{-1}$ 이 존재할(Invertible) 필요충분조건은 행렬식 값이 0이 아닌 것입니다:
  $$A \text{ is invertible} \iff \det(A) \neq 0$$

- 소형 행렬의 닫힌 형식 행렬식 공식:
  1. $1 \times 1$ 행렬: $\det([a_{11}]) = a_{11}$ (Eq 4.5)
  2. $2 \times 2$ 행렬: $\det\left(\begin{bmatrix} a_{11} & a_{12} \\\\ a_{21} & a_{22} \end{bmatrix}\right) = a_{11}a_{22} - a_{12}a_{21}$ (Eq 4.4, 4.6)
  3. $3 \times 3$ 행렬 (사루스 법칙 Sarrus' rule: Eq 4.7):
     $$\det(A) = a_{11}a_{22}a_{33} + a_{21}a_{32}a_{13} + a_{31}a_{12}a_{23} - a_{31}a_{22}a_{13} - a_{11}a_{32}a_{23} - a_{21}a_{12}a_{33}$$
  4. 삼각행렬 (Upper/Lower Triangular Matrix: Eq 4.8): 대각선 아래 또는 위가 모두 0인 삼각행렬 $T$ 의 행렬식은 주대각선 성분들의 단순 곱과 같습니다:
     $$\det(T) = \prod_{i=1}^n T_{ii}$$


### 📌 2. 행렬식의 기하학적 의미: 부호 있는 부피 (Signed Volume: Example 4.2)

행렬 $A$ 의 행렬식 $\det(A)$ 의 절대값 $|\det(A)|$ 는 열벡터들이 $n$차원 공간에서 생성하는 평행육면체(Parallelepiped)의 $n$차원 초부피(Volume)를 의미합니다!

- 2차원 평면: 열벡터 $\mathbf{b}, \mathbf{g}$ 가 이루는 평행사변형의 넓이 $= |\det([\mathbf{b}, \mathbf{g}])|$ (Figure 4.2).
- 3차원 공간: 세 열벡터 $\mathbf{r}, \mathbf{g}, \mathbf{b}$ 가 이루는 평행육면체의 부피 $= |\det([\mathbf{r}, \mathbf{g}, \mathbf{b}])|$ (Figure 4.3).
- 선형종속성과의 관계: 열벡터들이 서로 선형종속이면 부피가 0으로 찌그러지며, 따라서 $\det(A) = 0$ 이 됩니다.
- 부호(Sign)의 의미: 표준기저 대비 공간 축이 뒤집혔는지(Flip / Orientation 반전)를 나타냅니다.

#### 💡 [Example 4.2 수치 연산]
$\mathbf{r} = \begin{bmatrix} 2 \\\\ 0 \\\\ -8 \end{bmatrix}, \mathbf{g} = \begin{bmatrix} 6 \\\\ 1 \\\\ 0 \end{bmatrix}, \mathbf{b} = \begin{bmatrix} 1 \\\\ 4 \\\\ -1 \end{bmatrix}$ 에 대해 행렬 $A = [\mathbf{r}, \mathbf{g}, \mathbf{b}]$ 의 부피:
$$V = |\det(A)| = \left| \det\begin{bmatrix} 2 & 6 & 1 \\\\ 0 & 1 & 4 \\\\ -8 & 0 & -1 \end{bmatrix} \right| = |2(-1 - 0) - 6(0 - (-32)) + 1(0 - (-8))| = |-2 - 192 + 8| = |-186| = 186$$


### 📌 3. 라플라스 전개 (Laplace Expansion: Theorem 4.2 & Example 4.3)

$n > 3$ 차원 행렬식은 특정 행이나 열을 따라 소행렬식(Minor $\det(A_{k,j})$)과 여인수(Cofactor $(-1)^{k+j}\det(A_{k,j})$)의 선형결합으로 쪼개어 재귀적으로 계산합니다:

1. $j$번째 열을 따른 라플라스 전개:
   $$\det(A) = \sum_{k=1}^n (-1)^{k+j} a_{kj} \det(A_{k,j}) \quad (\text{Eq 4.12})$$
2. $j$번째 행을 따른 라플라스 전개:
   $$\det(A) = \sum_{k=1}^n (-1)^{k+j} a_{jk} \det(A_{j,k}) \quad (\text{Eq 4.13})$$

#### 💡 [Example 4.3 수치 연산]
$A = \begin{bmatrix} 1 & 2 & 3 \\\\ 3 & 1 & 2 \\\\ 0 & 0 & 1 \end{bmatrix}$ 의 3번째 행(0이 많은 행)을 따른 라플라스 전개:
$$\det(A) = 0 \cdot C_{31} + 0 \cdot C_{32} + (-1)^{3+3} \cdot 1 \cdot \det\begin{bmatrix} 1 & 2 \\\\ 3 & 1 \end{bmatrix} = 1 \cdot (1 - 6) = -5$$


### 📌 4. 행렬식의 7대 핵심 대수적 성질

1. 행렬 곱의 행렬식: $\det(AB) = \det(A)\det(B)$
2. 전치 행렬의 행렬식: $\det(A^\top) = \det(A)$
3. 역행렬의 행렬식: $\det(A^{-1}) = \frac{1}{\det(A)}$
4. 유사 행렬 및 기저 변환 불변성: $\det(S^{-1}AS) = \det(A)$ (기저가 바뀌어도 선형 변환의 부피 팽창률은 불변!)
5. 행/열의 기본 행 연산 불변성: 한 행/열에 다른 행/열의 상수배를 더해도 행렬식은 변하지 않습니다 (가우스 소거법 활용 근거).
6. 행렬 스칼라 배 스케일링: 행/열 하나에 $\lambda$ 를 곱하면 $\det(A)$ 가 $\lambda$ 배 되며, $n \times n$ 전체에 곱하면 $\det(\lambda A) = \lambda^n \det(A)$ 가 됩니다.
7. 행/열 교환 시 부호 반전: 두 행 또는 두 열의 위치를 맞바꾸면 행렬식의 부호가 반전됩니다 ($\det(A) \to -\det(A)$).

- 계수(Rank)와의 동치 관계 (Theorem 4.3):
  $$\det(A) \neq 0 \iff \text{rk}(A) = n \iff A \text{ is Full Rank}$$


## 2. ⚔️ Section 4.1: Trace (대각합)


### 📌 1. 대각합의 정의 (Definition 4.4 & Eq 4.18)

정방행렬 $A \in \mathbb{R}^{n \times n}$ 의 주대각선 성분(Diagonal entries)들을 모두 더한 값을 대각합(Trace)이라 정의합니다:

$$\text{tr}(A) := \sum_{i=1}^n a_{ii} \quad (\text{Eq 4.18})$$


### 📌 2. 대각합의 핵심 성질과 순환 불변성 (Cyclic Permutation: Eq 4.19~4.21)

1. 선형성: $\text{tr}(A + B) = \text{tr}(A) + \text{tr}(B)$, $\text{tr}(\alpha A) = \alpha \text{tr}(A)$
2. 단위행렬의 대각합: $\text{tr}(I_n) = n$
3. 두 행렬 곱의 교환 가능성: $\text{tr}(AB) = \text{tr}(BA)$ (단, $A \in \mathbb{R}^{n \times k}, B \in \mathbb{R}^{k \times n}$)
4. 순환 불변성 (Cyclic Permutation Property: Eq 4.19):
   세 개 이상의 행렬 곱에 대해서도 순환 순서가 유지되면 대각합이 완벽히 일치합니다:
   $$\text{tr}(AKL) = \text{tr}(KLA) = \text{tr}(LAK)$$
5. 벡터 외적과 내적의 대각합 연결 (Eq 4.20):
   $$\text{tr}(\mathbf{x}\mathbf{y}^\top) = \text{tr}(\mathbf{y}^\top\mathbf{x}) = \mathbf{y}^\top\mathbf{x} \in \mathbb{R}$$
6. 기저 변환 불변성 (Basis Invariance: Eq 4.21):
   기저가 바뀌어 행렬이 $S^{-1}AS$ 로 변환되어도 대각합은 변하지 않습니다:
   $$\text{tr}(S^{-1}AS) = \text{tr}(ASS^{-1}) = \text{tr}(A)$$


## 3. ⚔️ Section 4.1: Characteristic Polynomial (특성 다항식)


### 📌 1. 특성 다항식의 정의 (Definition 4.5 & Eq 4.22~4.24)

정방행렬 $A \in \mathbb{R}^{n \times n}$ 과 스칼라 $\lambda \in \mathbb{R}$ 에 대해 다음 다항식을 행렬 $A$ 의 특성 다항식(Characteristic Polynomial)이라 부릅니다:

$$p_A(\lambda) := \det(A - \lambda I) = c_0 + c_1\lambda + c_2\lambda^2 + \dots + c_{n-1}\lambda^{n-1} + (-1)^n\lambda^n \quad (\text{Eq 4.22})$$

- 상수항 $c_0$ 의 정체 (Eq 4.23): $\lambda = 0$ 을 대입하면 바로 도출됩니다:
  $$c_0 = p_A(0) = \det(A)$$
- $c_{n-1}$ 계수의 정체 (Eq 4.24): 행렬의 대각합과 직결됩니다:
  $$c_{n-1} = (-1)^{n-1} \text{tr}(A)$$

이 특성 방정식 $p_A(\lambda) = 0$ 의 근이 바로 4.2절에서 다룰 행렬의 고유값(Eigenvalues)이 됩니다!


## 🧠 4. 4단계 정밀 개념 해설


### 1️⃣ [1단계 개념 정의]
- 행렬식 ($\det(A)$): 행렬 변환에 의해 공간의 초부피가 몇 배로 팽창/수축하는지를 나타내는 스칼라 부피 팽창 배율입니다.
- 대각합 ($\text{tr}(A)$): 주대각선 성분의 합이자 기저 변환에 불변인 고유값들의 총합입니다.
- 특성 다항식 ($p_A(\lambda) = \det(A - \lambda I)$): 행렬식과 대각합을 계수로 품고 행렬의 고유 스펙트럼을 결정하는 핵심 대수 다항식입니다.


### 2️⃣ [2단계 왜 쓰는가?]
- 가역성 및 영공간 존재 여부의 즉각적 판별: $\det(A) = 0$ 인 순간 연립방정식의 유일해가 파탄 나고 영공간이 존재함을 즉시 알 수 있습니다.
- 고차원 데이터의 기하학적 요약: 복잡한 $n \times n$ 행렬의 전체적인 스케일(부피 변동률은 $\det$, 총 분산 크기는 $\text{tr}$)을 단 하나의 숫자로 빠르게 요약하기 위해 사용합니다.


### 3️⃣ [3단계 상황별 직관 & Trade-off]
- 손계산(라플라스 전개 $O(n!)$) vs 컴퓨터 수치 계산(가우스 소거 / LU 분해 $O(n^3)$):
  - 라플라스 전개는 $n$이 커지면 계산량이 팩토리얼($n!$)로 폭발하므로 이론 증명용으로만 쓰입니다.
  - 실제 컴퓨터 ML 라이브러리는 가우스 소거법으로 상삼각행렬을 만든 후 주대각선 성분만 곱해 $O(n^3)$ 으로 행렬식을 고속 산출합니다.


### 4️⃣ [4단계 실전 AI 연결고리]
- 정규화 흐름 모델 (Normalizing Flows - Ch 11 생성 AI): 확률밀도함수를 비선형 변환할 때 변수 변환 공식(Change of Variables)에서 야코비안 행렬식 $|\det(J)|$ 을 계산하여 확률 보존을 수행합니다.
- 다변량 가우시안 분포 (Multivariate Gaussian - Ch 6): 다변량 정규분포의 확률밀도함수 정규화 상수에 공분산 행렬의 행렬식 $\sqrt{\det(\Sigma)}$ 이 분모로 들어갑니다.
- 딥러닝 손실함수의 Trace Trick: 데이터 행렬의 공분산 및 정규화 손실을 계산할 때 $\mathbf{x}^\top A \mathbf{x} = \text{tr}(A \mathbf{x}\mathbf{x}^\top)$ 성질을 활용하여 벡터화 병렬 연산을 수행합니다.
