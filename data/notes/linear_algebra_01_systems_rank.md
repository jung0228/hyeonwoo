# 📐 MML Chapter 2: Linear Algebra (선형대수학 전수 완전 해부 바이블)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Chapter 2 전수 정복
> 
> 본 문서는 MML 교재 Chapter 2 (2.1절부터 2.9절까지 단 하나의 정의, 정리, 예시, 맹점도 빠짐없이) 완전히 파고들어 전수 정리한 바이블 노트입니다.


## 📌 Chapter 2 세부 목차 (Full Section Map)

- 2.1 Systems of Linear Equations (선형방정식계): Row vs Column Picture
- 2.2 Matrices (행렬 대수): 덧셈, 곱셈, 역행렬, 전치행렬, 결합성
- 2.3 Solving Systems of Linear Equations (선형계의 풀이):
  - 2.3.1 Particular and General Solution ($x = x_p + x_h$)
  - 2.3.2 Elementary Transformations (3가지 ERO, REF, RREF, Minus-1 Trick)
- 2.4 Vector Spaces (벡터 공간):
  - Vector Space 공리 (8가지 연산 닫힘 성질)
  - Vector Subspaces (부분공간) & Example 2.12 (부분공간 판별 및 모순 사례 A, B, C, D 심층 해부)
- 2.5 Linear Independence (선형 독립):
  - 선형 결합(Linear Combination)과 Span
  - 선형 독립(Linear Independence) vs 선형 종속(Linear Dependence) 판별
- 2.6 Basis and Rank (기저와 계수):
  - 기저(Basis)의 정의 및 유일 좌표 표현
  - Rank (Row Rank = Column Rank 증명), 기하학적 유효 차원
- 2.7 Linear Mappings (선형 사상 & 표현 행렬):
  - 선형 변환 $T(x+y)=T(x)+T(y)$, Kernel(Nullspace) & Image(Column Space)
  - Rank-Nullity 정리 ($\dim(V) = \text{Rank}(T) + \text{Nullity}(T)$)
  - 기저변환(Change of Basis) & 닮음 변환($P^{-1}AP$)
- 2.8 Affine Spaces (아핀 공간):
  - 아핀 공간 $L = x_0 + U$, 아핀 변환 $f(x) = Ax + b$
- 2.9 Summary & AI Connection (실전 AI 연결고리)


## 💡 1. Section 2.1 & 2.2: 선형방정식계 & 행렬 대수

### 1️⃣ $Ax = b$를 바라보는 2가지 기하학적 시각
- Row Picture (행 시각): 초평면(Hyperplane)들의 교점 찾기.
- Column Picture (열 시각 - AI 핵심!): 열벡터들의 선형 결합 $$a_1 x_1 + a_2 x_2 + \dots + a_n x_n = b$$
- 핵심 인사이트: $Ax=b$의 해 존재 $\iff b \in \text{Col}(A)$ ($b$가 열공간 내에 존재).

### 2️⃣ 행렬 곱셈의 본질
- $C = AB \iff c_{ij} = \sum_k a_{ik}b_{kj}$
- 기하학적 본질: 변환 $B$ 적용 후 변환 $A$를 연속 적용하는 선형 사상의 합성(Composition of Linear Mappings).


## 💡 2. Section 2.3: 선형방정식계의 풀이 (Solving Systems)

### 1️⃣ Section 2.3.1: Particular and General Solution
- 해의 일반 구조: $\mathbf{x = x_p + x_h}$
  - 특수해 $x_p$: $A x_p = b$ 만족.
  - 동차해 $x_h$: $A x_h = 0$ 만족 (행렬 $A$의 Nullspace / Kernel).
- 아핀 공간 구조: 원점을 지나는 선형 부분공간 $x_h$가 특수해 점 $x_p$만큼 평행 이동된 공간.

### 2️⃣ Section 2.3.2: Elementary Transformations (기본 행 연산)
- 3가지 ERO: 1) Exchange ($R_i \leftrightarrow R_j$), 2) Scaling ($R_i \leftarrow c R_i$), 3) Addition ($R_i \leftarrow R_i + c R_j$)
- 성질: ERO를 수행해도 선형계의 해 집합 및 Nullspace는 100% 보존됨 (Row Equivalent).
- REF vs RREF: RREF는 피벗이 $1$이고, 피벗 열의 다른 모든 성분이 $0$.
- The Minus-1 Trick: RREF 자유 변수 열의 대각 위치에 $-1$을 채워 넣고 Nullspace 기저를 연산 없이 즉시 추출하는 MML 고유 스킬.


## 💡 3. Section 2.4: Vector Spaces & Vector Subspaces (벡터 공간과 부분공간)

### 1️⃣ Vector Space 공리
집합 $V$와 체 $\mathbb{R}$에 대해 덧셈과 스칼라배 연산이 아래 8가지 공리(닫힘성, 결합법칙, 교환법칙, 항등원 $0$, 역원 $-v$, 분배법칙 등)를 만족하는 공간.

### 2️⃣ Vector Subspaces (부분공간) 조건
$V$의 부분집합 $U \subseteq V$가 그 자체로 벡터 공간이 되기 위한 3가지 필수 조건:
1. 비어있지 않음 (Non-empty): 원점(Zero Vector)을 반드시 포함함 ($\mathbf{0} \in U$).
2. 덧셈에 대해 닫혀있음 (Closed under Addition): $\forall u, v \in U \implies u + v \in U$.
3. 스칼라배에 대해 닫혀있음 (Closed under Scalar Multiplication): $\forall u \in U, c \in \mathbb{R} \implies c u \in U$.


### 🔍 ★ MML 교재 원문 심층 해부: Example 2.12 (Vector Subspaces)

MML 교재에서 $\mathbb{R}^2$ 평면 상의 4가지 부분집합 $A, B, C, D$의 예시를 통해 부분공간 판별 조건을 직관적으로 증명함.

```
       [Subset A]                   [Subset B]                   [Subset C]                   [Subset D]
   (원점을 지나지 않는 직선)       (1사분면 전체 영역)           (두 원점을 지나는 십자선)       (원점을 지나는 원점을 지나는 직선)
        |    /                          |                             |   |                        \
        |   / (x2 = x1 + 1)             |  Q1 Only                    |   |                         \ (x2 = -2x1)
   -----+--/----->                 -----+--------->              -----+---+----->             -----+-----\---->
        | /                             |                             |   |                        \
        |/                              |                             |   |                         \
```

- [Case A] $U_A = \{(x_1, x_2) \in \mathbb{R}^2 \mid x_2 = x_1 + 1\}$ (원점을 지나지 않는 직선)
  - ❌ 부분공간 탈락!
  - 이유: $(0, 0)$을 대입하면 $0 = 0 + 1$ 모순 ➡️ 원점(Zero Vector)을 포함하지 않음 ($\mathbf{0} \notin U_A$).
- [Case B] $U_B = \{(x_1, x_2) \in \mathbb{R}^2 \mid x_1 \ge 0, x_2 \ge 0\}$ (1사분면 전체)
  - ❌ 부분공간 탈락!
  - 이유: 스칼라배에 대해 닫혀있지 않음. $(1, 1) \in U_B$이지만 $c = -1$을 곱하면 $(-1, -1) \notin U_B$.
- [Case C] $U_C = \{(x_1, x_2) \in \mathbb{R}^2 \mid x_1 x_2 = 0\}$ ($x_1$축과 $x_2$축의 합집합)
  - ❌ 부분공간 탈락!
  - 이유: 덧셈에 대해 닫혀있지 않음. $(1, 0) \in U_C$이고 $(0, 1) \in U_C$이지만 둘을 더한 $(1, 1) \notin U_C$ ($1 \cdot 1 = 1 \neq 0$).
- [Case D] $U_D = \{(x_1, x_2) \in \mathbb{R}^2 \mid x_2 = -2 x_1\}$ (원점을 지나는 직선)
  - ✅ 완벽한 벡터 부분공간 (Subspace)!
  - 이유: 
    1) 원점 포함: $(0, 0) \in U_D$.
    2) 덧셈 닫힘: $(a, -2a) + (b, -2b) = (a+b, -2(a+b)) \in U_D$.
    3) 스칼라배 닫힘: $c(a, -2a) = (ca, -2(ca)) \in U_D$.


## 💡 4. Section 2.5 & 2.6: 선형 독립, 기저(Basis), 그리고 Rank

### 1️⃣ Linear Independence (선형 독립)
- 벡터 집합 $\{v_1, \dots, v_k\}$의 선형 결합 $\sum c_i v_i = 0$을 만족하는 계수가 오직 $c_1 = c_2 = \dots = c_k = 0$ 뿐일 때 선형 독립(Linearly Independent)임.

### 2️⃣ Basis (기저)
- 벡터 공간 $V$를 생성(Span)하면서 동시에 선형 독립인 최소 벡터 집합.
- 성질: 공간 내의 모든 벡터는 기저 벡터들의 유일한 선형 결합(Unique Linear Combination)으로만 표현됨.

### 3️⃣ Rank (계수)
- 행렬 $A$의 독립적인 행/열의 개수. ($\text{Row Rank} = \text{Column Rank} = \text{Rank}(A)$).
- Rouché-Capelli Theorem: $\text{Rank}(A) = \text{Rank}([A \mid b]) = n \iff$ 유일해.


## 💡 5. Section 2.7 & 2.8: 선형 사상, Rank-Nullity 정리, 어파인 공간

### 1️⃣ Linear Mappings & Matrix Representation
- 사상 $T: V \to W$가 $T(x+y) = T(x)+T(y)$ 및 $T(cx) = cT(x)$를 만족할 때 선형 사상.
- Kernel (Nullspace): $T(x) = 0$으로 소실되는 $V$의 부분공간 $\ker(T)$.
- Image (Column Space): $T$에 의해 도달하는 $W$의 부분공간 $\text{Im}(T)$.

### 2️⃣ Rank-Nullity Theorem (차원 보존 정리)
- $\dim(V) = \dim(\ker(T)) + \dim(\text{Im}(T)) \iff n = \text{Nullity}(T) + \text{Rank}(T)$
- AI 연결: Autoencoder에서 라텐트 차원으로 압축되어 소실되는 정보가 $\text{Nullity}$, 보존되어 생성되는 정보가 $\text{Rank}$.

### 3️⃣ Change of Basis (기저변환)
- 구기저에서 신기저로의 변환 행렬 $P$. 표현 행렬의 닮음 변환: $\mathbf{A_{new} = P^{-1} A_{old} P}$.

### 4️⃣ Affine Spaces (아핀 공간)
- 원점을 지나지 않는 부분공간의 평행이동 $L = x_0 + U$.
- 신경망 레이어: $y = \sigma(W x + b)$에서 편향(Bias) $b$가 아핀 공간 평행 이동을 담당.


## 📝 6. MML Chapter 2 전수 연습문제 풀이집

### 📌 [Problem 1] Ex 2.1 - 3차원 선형계 가우스 소거 (유일해)
- $[A \mid b] \to \text{RREF} \implies \mathbf{x = [3, -1, 0]^T}$.

### 📌 [Problem 2] Ex 2.2 - Inconsistent System (해 없음)
- $0 = 1$ 모순, $\text{Rank}(A)=1 < \text{Rank}([A \mid b])=2 \implies b \notin \text{Col}(A)$.

### 📌 [Problem 3] Ex 2.12 - Subspace 4가지 Case 백지 판별 (MML 핵심 예제)
- Case A(원점 미포함), Case B(스칼라배 음수 파탄), Case C(덧셈 파탄) ➡️ Case D만 원점을 지나는 직선으로 완벽한 Subspace!
