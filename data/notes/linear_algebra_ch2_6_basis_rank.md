# 📐 2.6 Basis and Rank (기저와 계수)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.6 원문 완전 대조 스토리텔링 노트

---

## 1. 🌐 서론: 왜 "기저(Basis)"와 "계수(Rank)"를 배우는가?

2.5절에서 우리는 벡터들의 중복성을 판단하는 선형독립(Linear Independence)과 공간을 펼쳐내는 스팬(Span)을 공부했습니다.
이제 2.6절의 핵심 주제는 "그렇다면 어떠한 벡터 모음이 공간 전체를 낭비 없이 정확하게 지탱하는 뼈대(기저)가 되는가?" 그리고 "행렬이 가진 실질적인 정보의 차원 수(Rank)는 얼마인가?" 입니다.

---

## 2. ⚔️ Section 2.6.1: Generating Set and Basis (생성집합과 기저)

### 📌 1. 기저(Basis)의 동치 조건 (Definition 2.14 & Theorem 2.15)
공집합이 아닌 벡터 집합 $B \subseteq V$ 에 대해 다음 명제들은 모두 완전히 동치(Equivalent)입니다:

1. $B$ 는 벡터 공간 $V$ 의 기저(Basis)입니다.
2. $B$ 는 공간 $V$ 의 최소 생성집합(Minimal Generating Set)입니다. (원소를 하나라도 빼면 더 이상 공간 전체를 생성할 수 없음)
3. $B$ 는 공간 $V$ 의 최대 선형독립 집합(Maximal Linearly Independent Set)입니다. (원소를 하나라도 더 추가하면 무조건 선형종속이 됨)
4. 공간 $V$ 안의 모든 벡터 $\mathbf{x} \in V$ 는 $B$ 의 원소들의 선형결합으로 "유일하게(Uniquely)" 표현됩니다 (Eq 2.77).

$$\mathbf{x} = \sum_{i=1}^k \lambda_i \mathbf{b}_i = \sum_{i=1}^k \psi_i \mathbf{b}_i \implies \lambda_i = \psi_i$$

---

### 📌 2. 차원(Dimension: Definition 2.16)
- 차원 $\text{dim}(V)$: 벡터 공간 $V$ 의 기저 벡터의 개수를 의미합니다.
- 직관적 의미: 그 공간 안에서 서로 독립적으로 움직일 수 있는 독립된 방향의 개수입니다.
- Remark: 차원이 벡터 내부 원소의 개수를 의미하는 것은 아닙니다. 예를 들어 $V = \text{span}\left(\begin{bmatrix} 0 \\ 1 \end{bmatrix}\right) \subseteq \mathbb{R}^2$ 는 원소가 2개이지만 독립 방향이 1개이므로 1차원 부분공간입니다.

---

### 📌 3. 실전 기저 구하기 4단계 알고리즘 (Remark p.46)
부분공간 $U = \text{span}[\mathbf{x}_1, \dots, \mathbf{x}_m] \subseteq \mathbb{R}^n$ 의 기저를 구하는 법:
1. 생성 벡터들을 행렬의 열벡터(Column Vectors)로 써서 행렬 $A = [\mathbf{x}_1 \mid \dots \mid \mathbf{x}_m]$ 을 구성합니다.
2. 가우스 소거법을 수행하여 행 사다리꼴(REF)로 변환합니다.
3. 피벗 열(Pivot Columns)에 해당하는 원래 행렬 $A$ 의 열벡터들을 선택합니다.
4. 이 선택된 열벡터 모음이 바로 부분공간 $U$ 의 기저(Basis)가 됩니다!

---

## 3. ⚔️ Section 2.6.2: Rank (행렬의 계수)

### 📌 1. Rank(계수)의 정의와 대칭성 (Definition 2.17 & Remark p.47)
행렬 $A \in \mathbb{R}^{m \times n}$ 의 선형독립인 열(Column)의 개수를 행렬의 계수(Rank)라 부르며 $\text{rk}(A)$ 로 표기합니다.

- 놀라운 대칭성 (Fundamental Theorem of Rank):
  $$\text{rk}(A) = \text{rk}(A^\top)$$
  즉, 선형독립인 열의 개수(Column Rank)와 선형독립인 행의 개수(Row Rank)는 언제나 정확하게 일치합니다!

---

### 📌 2. Rank의 핵심 주요 성질
- $\text{rk}(A) \le \min(m, n)$: 행렬의 Rank는 행의 개수와 열의 개수 중 작은 값을 넘을 수 없습니다.
- Full Rank (만적 계수): $\text{rk}(A) = \min(m, n)$ 일 때 행렬이 손실 없이 꽉 차있다고 말합니다.
- Matrix Multiplication Rank Bound: $\text{rk}(AB) \le \min(\text{rk}(A), \text{rk}(B))$
- Subadditivity: $\text{rk}(A + B) \le \text{rk}(A) + \text{rk}(B)$

---

### 📌 3. MML 원문 수치 계산 예제 (Eq 2.84)
$$A = \begin{bmatrix} 1 & 2 & 1 \\ -2 & -3 & 1 \\ 3 & 5 & 0 \end{bmatrix} \xrightarrow{\text{REF}} \begin{bmatrix} \mathbf{1} & 2 & 1 \\ 0 & \mathbf{1} & 3 \\ 0 & 0 & 0 \end{bmatrix}$$

피벗의 개수가 2개이므로, 이 행렬의 실질적 정보 차원 수인 $\text{rk}(A) = 2$ 가 됩니다!
