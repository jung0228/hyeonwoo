# 📐 3.2 Inner Products (내적과 대칭 양의 정정 행렬)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 3.2 전수 분석 & 4단계 정밀 해설 노트

## 🌐 0. 3.1절(Norms)과의 연결 및 자연스러운 빌드업: 왜 "내적(Inner Product)"을 배우는가?

우리는 지난 3.1절(Norms)에서 벡터의 '크기'인 길이(Length)와 원점으로부터의 거리(Distance)를 다루는 수학적 검증 기준인 노름(Norm)을 공부했습니다. 
그러나 노름이라는 도구 하나만으로는 공간 안에서 "두 벡터가 서로 몇 도로 기울어져 있는지(각도 Angle)"나 "두 벡터가 서로 완전히 독립적으로 90도로 꺾여 있는지(직교성 Orthogonality)"를 직접 판단하거나 계산할 수 없습니다.

바로 이 지점에서 3.2절 내적(Inner Product)이 등장합니다! 

내적은 두 벡터를 입력받아 하나의 실수를 반환하는 이선형(Bilinear) 사상으로서, 벡터 공간에 각도(Angle)와 직교성(Orthogonality)이라는 기하학적 생명력을 불어넣는 훨씬 더 본질적이고 강력한 도구입니다.

### 🔗 노름(3.1절) ➡️ 내적(3.2절) ➡️ 후속 AI 모델로 이어지는 거대한 기하학적 흐름

1. 노름에서 내적으로의 자연스러운 확장:
   사실 노름과 내적은 완전히 별개의 개념이 아닙니다! 자기 자신과의 내적에 제곱근을 취하면 바로 벡터의 노름이 유도됩니다 ($\Vert\mathbf{x}\Vert = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle}$). 즉, 내적은 노름을 품고 있는 더 근본적인 상위 구조입니다.

2. 두 벡터 사이의 각도와 직교성 탄생:
   내적 $\langle \mathbf{x}, \mathbf{y} \rangle$ 을 정의함으로써 우리는 비로소 $\cos\theta = \frac{\langle \mathbf{x}, \mathbf{y} \rangle}{\Vert\mathbf{x}\Vert \Vert\mathbf{y}\Vert}$ 관계식을 통해 두 데이터 벡터 간의 각도 $\theta$ 를 측정할 수 있게 되며, 내적값이 0일 때 두 벡터가 완전히 직교(Orthogonal)함을 선언할 수 있게 됩니다.

3. 대칭 양의 정정 행렬(SPD)과 후속 AI 인공지능 알고리즘 연결:
   일반적인 내적 연산은 행렬 곱 형태인 $\langle \mathbf{x}, \mathbf{y} \rangle = \hat{\mathbf{x}}^\top A \hat{\mathbf{y}}$ 로 확장되며, 이때 등장하는 가중치 행렬 $A$ 가 바로 대칭 양의 정정 행렬(Symmetric Positive Definite Matrix, SPD)입니다. 
   이 개념은 다음과 같은 핵심 AI 모델들로 직접 연결됩니다:
   - Chapter 12 (Kernel Methods / SVM): 고차원 공간에서의 내적 연산을 직접 수행하지 않고 효율적으로 대체하는 커널 트릭(Kernel Trick)의 이론적 뼈대가 됩니다.
   - Chapter 10 (PCA - 주성분 분석): 데이터의 변동성이 가장 큰 직교하는 기저 축(Orthogonal Basis)으로 정사영(Projection)시키는 기준이 됩니다.
   - Chapter 4 (Matrix Decompositions): 공분산 행렬과 같은 SPD 행렬을 삼각형 행렬의 곱으로 고속 분해하는 Cholesky 분해 및 고유값 분해(Eigendecomposition)의 출발점이 됩니다.

## 1. ⚔️ Section 3.2.1 & 3.2.2: General Inner Products (일반 내적의 정의)

### 📌 1. 도트 곱 (Dot Product / Scalar Product: Eq 3.5)
우리가 유클리드 공간 $\mathbb{R}^n$ 에서 흔히 사용하는 표준적인 내적을 도트 곱(Dot Product)이라 부릅니다:

$$\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^n x_i y_i \quad (\text{Eq 3.5})$$

### 📌 2. 이선형 사상 (Bilinear Mapping: Eq 3.6~3.7)
인자 2개를 받는 사상 $\Omega : V \times V \to \mathbb{R}$ 이 각 인자에 대해 각각 선형성(Linearity)을 가질 때 이를 이선형 사상(Bilinear Mapping)이라 부릅니다. 
즉, 모든 $\mathbf{x}, \mathbf{y}, \mathbf{z} \in V$ 및 $\lambda, \psi \in \mathbb{R}$ 에 대해 다음이 성립합니다:

1. 첫 번째 인자에 대한 선형성:
   $$\Omega(\lambda \mathbf{x} + \psi \mathbf{y}, \mathbf{z}) = \lambda \Omega(\mathbf{x}, \mathbf{z}) + \psi \Omega(\mathbf{y}, \mathbf{z}) \quad (\text{Eq 3.6})$$
2. 두 번째 인자에 대한 선형성:
   $$\Omega(\mathbf{x}, \lambda \mathbf{y} + \psi \mathbf{z}) = \lambda \Omega(\mathbf{x}, \mathbf{y}) + \psi \Omega(\mathbf{x}, \mathbf{z}) \quad (\text{Eq 3.7})$$

### 📌 3. 내적과 내적 공간의 엄밀한 정의 (Definition 3.2 & 3.3)
벡터 공간 $V$ 위에서 정의된 이선형 사상 $\Omega : V \times V \to \mathbb{R}$ 이 다음 두 성질을 만족하면 이를 내적(Inner Product)이라 부르고 $\langle \mathbf{x}, \mathbf{y} \rangle$ 로 표기합니다:

1. 대칭성 (Symmetric: Def 3.2): 
   $$\forall \mathbf{x}, \mathbf{y} \in V : \langle \mathbf{x}, \mathbf{y} \rangle = \langle \mathbf{y}, \mathbf{x} \rangle$$
2. 양의 정정성 (Positive Definite: Def 3.2 & Eq 3.8): 
   $$\forall \mathbf{x} \in V \setminus \{\mathbf{0}\} : \langle \mathbf{x}, \mathbf{x} \rangle > 0 \quad \text{and} \quad \langle \mathbf{0}, \mathbf{0} \rangle = 0$$

- 내적 공간 (Inner Product Space: Def 3.3): 순서쌍 $(V, \langle \cdot, \cdot \rangle)$ 을 내적 공간이라 부릅니다. 표준 도트 곱을 사용하는 내적 공간은 유클리드 벡터 공간(Euclidean Vector Space)이라 부릅니다.

### 📌 4. 도트 곱이 아닌 일반 내적 예시 (Example 3.3 & Eq 3.9)
$\mathbb{R}^2$ 에서 다음과 같이 정의된 연산은 도트 곱($x_1 y_1 + x_2 y_2$)이 아니지만 내적의 공리를 완벽히 만족하는 내적입니다:

$$\langle \mathbf{x}, \mathbf{y} \rangle := x_1 y_1 - (x_1 y_2 + x_2 y_1) + 2 x_2 y_2 \quad (\text{Eq 3.9})$$

---

## 2. ⚔️ Section 3.2.3: Symmetric, Positive Definite Matrices (대칭 양의 정정 행렬)

### 📌 1. 행렬 표현 $A$ 를 통한 내적 도출 (Eq 3.10 & Theorem 3.5)
유한차원 벡터 공간 $V$ 의 순서기저 $\mathcal{B} = (\mathbf{b}_1, \dots, \mathbf{b}_n)$ 에 대해 임의의 두 벡터 $\mathbf{x} = \sum_{i=1}^n \hat{x}_i \mathbf{b}_i$, $\mathbf{y} = \sum_{j=1}^n \hat{y}_j \mathbf{b}_j$ 의 내적은 이선형성에 의해 다음과 같이 행렬 곱 형태로 유일하게 결정됩니다:

$$\langle \mathbf{x}, \mathbf{y} \rangle = \left\langle \sum_{i=1}^n \hat{x}_i \mathbf{b}_i, \sum_{j=1}^n \hat{y}_j \mathbf{b}_j \right\rangle = \sum_{i=1}^n \sum_{j=1}^n \hat{x}_i \langle \mathbf{b}_i, \mathbf{b}_j \rangle \hat{y}_j = \hat{\mathbf{x}}^\top A \hat{\mathbf{y}} \quad (\text{Eq 3.10})$$

여기서 성분 $A_{ij} := \langle \mathbf{b}_i, \mathbf{b}_j \rangle$ 은 기저 벡터들 사이의 내적을 모아놓은 행렬입니다.

- Theorem 3.5: 실수 유한차원 공간 $V$ 상의 연산 $\langle \cdot, \cdot \rangle$ 이 내적일 필요충분조건은, 대칭 양의 정정 행렬 $A \in \mathbb{R}^{n \times n}$ 이 존재하여 $\langle \mathbf{x}, \mathbf{y} \rangle = \hat{\mathbf{x}}^\top A \hat{\mathbf{y}}$ 로 표현되는 것입니다 (Eq 3.15).

#### 💡 [손으로 직접 해보는 구체적인 수치 예시]

가장 이해하기 쉬운 2차원 공간 $\mathbb{R}^2$ 에서 구체적인 수치로 행렬 $A$ 가 어떻게 만들어지고 $\hat{\mathbf{x}}^\top A \hat{\mathbf{y}}$ 로 계산되는지 직접 확인해 봅시다!

##### 1단계: 기저 벡터들 사이의 내적값으로 행렬 $A$ 의 성분 채우기
우리가 예시 3.3(Example 3.3)에서 보았던 특수 내적 연산 $\langle \mathbf{u}, \mathbf{v} \rangle = u_1 v_1 - (u_1 v_2 + u_2 v_1) + 2 u_2 v_2$ 를 사용해 봅시다.
표준기저 $\mathbf{b}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \mathbf{b}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$ 의 기저끼리 내적을 구합니다:

- $A_{11} = \langle \mathbf{b}_1, \mathbf{b}_1 \rangle = 1 \cdot 1 - (1 \cdot 0 + 0 \cdot 1) + 2(0 \cdot 0) = \mathbf{1}$
- $A_{12} = \langle \mathbf{b}_1, \mathbf{b}_2 \rangle = 1 \cdot 0 - (1 \cdot 1 + 0 \cdot 0) + 2(0 \cdot 1) = \mathbf{-1}$
- $A_{21} = \langle \mathbf{b}_2, \mathbf{b}_1 \rangle = \langle \mathbf{b}_1, \mathbf{b}_2 \rangle = \mathbf{-1}$ (대칭성에 의해 $A_{12}$ 와 동일)
- $A_{22} = \langle \mathbf{b}_2, \mathbf{b}_2 \rangle = 0 \cdot 0 - (0 \cdot 1 + 1 \cdot 0) + 2(1 \cdot 1) = \mathbf{2}$

기저 벡터들의 내적 결과표인 행렬 $A$ 완성:
$$A = \begin{bmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix} = \begin{bmatrix} 1 & -1 \\ -1 & 2 \end{bmatrix}$$

##### 2단계: 임의의 두 벡터 $\mathbf{x} = \begin{bmatrix} 2 \\ 3 \end{bmatrix}$, $\mathbf{y} = \begin{bmatrix} 4 \\ 1 \end{bmatrix}$ 에 대해 내적 직접 계산
- 방법 1 (내적 정의에 직접 대입):
  $$\langle \mathbf{x}, \mathbf{y} \rangle = (2 \cdot 4) - (2 \cdot 1 + 3 \cdot 4) + 2 (3 \cdot 1) = 8 - 14 + 6 = \mathbf{0}$$
- 방법 2 (행렬 곱 $\hat{\mathbf{x}}^\top A \hat{\mathbf{y}}$ 에 대입):
  $$\hat{\mathbf{x}}^\top A \hat{\mathbf{y}} = \begin{bmatrix} 2 & 3 \end{bmatrix} \begin{bmatrix} 1 & -1 \\ -1 & 2 \end{bmatrix} \begin{bmatrix} 4 \\ 1 \end{bmatrix}$$
  1. 먼저 $A \hat{\mathbf{y}}$ 연산: $\begin{bmatrix} 1 & -1 \\ -1 & 2 \end{bmatrix} \begin{bmatrix} 4 \\ 1 \end{bmatrix} = \begin{bmatrix} 4 - 1 \\ -4 + 2 \end{bmatrix} = \begin{bmatrix} 3 \\ -2 \end{bmatrix}$
  2. 마지막 $\hat{\mathbf{x}}^\top$ 곱하기: $\begin{bmatrix} 2 & 3 \end{bmatrix} \begin{bmatrix} 3 \\ -2 \end{bmatrix} = 2 \cdot 3 + 3 \cdot (-2) = 6 - 6 = \mathbf{0}$

##### 🎯 결론
내적 정의에 복잡하게 하나씩 대입해서 계산한 결과($0$)와, 기저 내적표로 만든 행렬 곱 $\hat{\mathbf{x}}^\top A \hat{\mathbf{y}}$ 으로 계산한 결과($0$)가 완벽하게 100% 일치합니다! 
즉, 행렬 $A$ 는 "기저 벡터들끼리의 내적 관계를 모아놓은 가중치 표" 역할을 합니다.

### 📌 2. 대칭 양의 정정 행렬의 정의 (Definition 3.4 & Eq 3.11)
실수 대칭 행렬 $A \in \mathbb{R}^{n \times n}$ ($A^\top = A$) 이 모든 0이 아닌 벡터 $\mathbf{x} \in V \setminus \{\mathbf{0}\}$ 에 대해 다음을 만족할 때 대칭 양의 정정 행렬 (Symmetric Positive Definite Matrix, SPD)이라 부릅니다:

$$\mathbf{x}^\top A \mathbf{x} > 0 \quad (\text{Eq 3.11})$$

- 등호($\ge 0$)만 성립하는 경우에는 대칭 반양의 정정 행렬 (Symmetric Positive Semidefinite Matrix, SPSD)이라 부릅니다.

### 📌 3. 원문 예제 판별 (Example 3.4 & Eq 3.12~3.13b)
- $A_1 = \begin{bmatrix} 9 & 6 \\ 6 & 5 \end{bmatrix}$:
  $$\mathbf{x}^\top A_1 \mathbf{x} = 9 x_1^2 + 12 x_1 x_2 + 5 x_2^2 = (3 x_1 + 2 x_2)^2 + x_2^2 > 0 \quad (\forall \mathbf{x} \neq \mathbf{0})$$
  완전제곱식의 합으로 변형되어 항상 0보다 크므로 양의 정정 행렬(SPD) 성립!
- $A_2 = \begin{bmatrix} 9 & 6 \\ 6 & 3 \end{bmatrix}$:
  $$\mathbf{x}^\top A_2 \mathbf{x} = 9 x_1^2 + 12 x_1 x_2 + 3 x_2^2 = (3 x_1 + 2 x_2)^2 - x_2^2$$
  $\mathbf{x} = [2, -3]^\top$ 일 때 $(6 - 6)^2 - (-3)^2 = -9 < 0$ 이 되므로 양의 정정 행렬 불성립!

### 📌 4. 대칭 양의 정정 행렬(SPD)의 2대 핵심 정리 성질
1. Null Space (Kernel)의 단일성: $A$ 의 영공간(Kernel)은 오직 영벡터 $\{\mathbf{0}\}$ 뿐입니다 ($\text{ker}(A) = \{\mathbf{0}\}$). 따라서 $A$ 는 무조건 가역 행렬(Invertible / Non-singular)입니다.
2. 주대각 성분의 양수성: $A$ 의 모든 대각 성분 $a_{ii}$ 는 무조건 양수입니다 ($a_{ii} = \mathbf{e}_i^\top A \mathbf{e}_i > 0$).

---

## 🧠 3. 4단계 정밀 개념 해설

### 1️⃣ [1단계 개념 정의]
- 내적(Inner Product): 두 벡터를 받아 하나의 실수를 반환하는 이선형, 대칭, 양의 정정 사상이며, 공간에 각도(Angle)와 직교성(Orthogonality)을 부여하는 기하학적 기준입니다.
- SPD 행렬: $\mathbf{x}^\top A \mathbf{x} > 0$ 을 만족하는 대칭 행렬로, 일반적인 내적 연산 $\langle \mathbf{x}, \mathbf{y} \rangle = \hat{\mathbf{x}}^\top A \hat{\mathbf{y}}$ 을 생성하는 핵심 가중치 행렬입니다.

### 2️⃣ [2단계 왜 쓰는가?]
- 표준 도트 곱($\mathbf{x}^\top \mathbf{y}$)을 넘어, 공간의 축이 비틀어지거나 가중치가 다른 임의의 차원 공간에서도 거리와 각도를 왜곡 없이 엄밀하게 정의하기 위해 사용합니다.

### 3️⃣ [3단계 상황별 직관 & Trade-off]
- SPD 행렬 $A$ 의 이차형식(Quadratic Form) 직관:
  - $A = I$ 이면 평범한 동그란 형태의 유클리드 공간 도트 곱이 됩니다.
  - $A$ 가 일반적인 SPD 행렬이면 타원(Ellipsoid) 형태의 변형된 거리/각도 측정 공간이 만들어집니다.
  - $A$ 의 고유값이 모두 양수($\lambda_i > 0$)인 것과 양의 정정성이 완전 동치입니다!

### 4️⃣ [4단계 실전 AI 연결고리]
- 커널 기법 (Kernel Methods - Ch 12): 머신러닝의 SVM이나 커널 PCA에서 고차원 특징 공간에서의 내적 $\langle \Phi(\mathbf{x}), \Phi(\mathbf{y}) \rangle = k(\mathbf{x}, \mathbf{y})$ 을 직접 계산하지 않고 커널 함수 $k$ 로 대체하는 Kernel Trick의 이론적 기반이 됩니다.
- Cholesky 분해 (Cholesky Decomposition - Ch 4.3): 공분산 행렬(Covariance Matrix)과 같은 SPD 행렬은 삼각행렬의 곱 $A = L L^\top$ 으로 고속 분해되어 가우시안 샘플링 및 최적화 계산에 핵심 활용됩니다.
