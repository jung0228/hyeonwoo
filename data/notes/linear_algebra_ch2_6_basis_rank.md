# 📐 2.6 Basis and Rank (기저와 계수)

> **POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.6 전수 분석 & 4단계 정밀 해설 노트**

## 🌐 0. 지난 노트(2.5절)와의 연결 및 빌드업: 왜 "기저"와 "계수"를 배우는가?

우리는 지난 **2.5절 (Linear Independence & Span)**에서 벡터들의 중복성을 체크하는 **선형독립(Linear Independence)**과, 벡터들이 조합되어 뻗어나갈 수 있는 범위인 **스팬(Span)**을 공부했습니다.

2.5절을 마치며 자연스럽게 도달하는 다음 질문은 이것입니다:
**"어떤 공간 전체를 덮으면서(Span), 중복(종속)은 단 1%도 없는 가장 완벽하고 다이어트된 최소한의 뼈대 벡터 모음은 무엇인가?"**

이 질문의 답이 바로 **기저(Basis)**이며, 그 기저 벡터의 개수가 공간의 크기인 **차원(Dimension)**이 됩니다. 
또한 이 개념을 행렬로 가져와 **"행렬 안에 들어있는 진짜 알짜배기 정보의 차원 수"**를 측정한 것이 바로 **계수(Rank)**입니다!

## 1. ⚔️ Section 2.6.1: Generating Set and Basis (생성집합과 기저)

### 📌 1. 생성집합(Generating Set)과 스팬(Span)의 정확한 정의 (Definition 2.13)
벡터 공간 $V = (V, +, \cdot)$ 와 부분집합 $A = \{\mathbf{x}_1, \dots, \mathbf{x}_k\} \subseteq V$ 에 대해:
- **생성집합 (Generating Set)**: 공간 $V$ 안의 모든 벡터 $\mathbf{v} \in V$ 를 $A$ 에 속한 벡터들의 선형결합으로 빠짐없이 표현할 수 있을 때, $A$ 를 $V$ 의 생성집합이라 부릅니다.
- **스팬 (Span)**: $A$ 속 원소들로 만들 수 있는 모든 가능한 선형결합들의 집합을 $A$ 의 스팬이라 하며 $V = \text{span}[A]$ 로 표기합니다.

### 📌 2. 기저(Basis)의 4대 동치 조건 (Definition 2.14 & Theorem p.45)
공집합이 아닌 벡터 집합 $B \subseteq V$ 가 다음 4가지 중 어느 하나라도 만족하면, 완전히 동치(Equivalent)로서 **$B$ 는 공간 $V$ 의 기저(Basis)**가 됩니다:

1. **$B$ 는 공간 $V$ 의 기저(Basis)입니다.**
2. **$B$ 는 공간 $V$ 의 최소 생성집합(Minimal Generating Set)입니다.** (원소를 하나라도 빼면 더 이상 공간 전체를 덮지 못함).
3. **$B$ 는 공간 $V$ 의 최대 선형독립 집합(Maximal Linearly Independent Set)입니다.** (원소를 하나라도 더 추가하면 무조건 선형종속으로 무너짐).
4. **공간 $V$ 안의 모든 벡터 $\mathbf{x} \in V$ 는 $B$ 의 원소들의 선형결합으로 "오직 유일하게(Uniquely)" 표현됩니다 (Eq 2.77).**

$$\mathbf{x} = \sum_{i=1}^k \lambda_i \mathbf{b}_i = \sum_{i=1}^k \psi_i \mathbf{b}_i \implies \lambda_i = \psi_i \quad (\text{계수가 유일함})$$

### 📌 3. 표준 기저와 비표준 기저 (Example 2.16)
- **$\mathbb{R}^3$ 의 표준기저 (Canonical Basis)**:
  $$B = \left\{ \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} \right\}$$
- **$\mathbb{R}^3$ 의 비표준 기저 예시**:
  $$B_1 = \left\{ \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}, \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix} \right\}$$
- **기저가 아닌 예시**: $A = \{[1,2,3,4]^\top, [2,-1,0,2]^\top, [1,1,0,-4]^\top\}$ 는 4차원 공간 $\mathbb{R}^4$ 에서 선형독립이지만 원소가 3개뿐이어서 $\mathbb{R}^4$ 전체를 생성하지 못하므로 기저가 아닙니다!

### 📌 4. 차원(Dimension: Definition 2.16 & Remark)
- **차원 $\text{dim}(V)$**: 벡터 공간 $V$ 의 **기저 벡터의 개수**를 의미합니다.
- **직관적 맹점**: 차원이 벡터 내부 원소의 개수(길이)를 의미하지는 않습니다!
  - 예: $V = \text{span}\left(\begin{bmatrix} 0 \\ 1 \end{bmatrix}\right)$ 은 벡터 원소가 2개지만 기저가 1개이므로 **1차원 부분공간**입니다.

### 📌 5. 실전 기저 구하기 4단계 알고리즘 & 원문 예제 (Example 2.17)
부분공간 $U = \text{span}[\mathbf{x}_1, \dots, \mathbf{x}_m] \subseteq \mathbb{R}^n$ 의 기저를 찾는 법:
1. 생성 벡터들을 행렬의 열벡터로 나열하여 행렬 $A = [\mathbf{x}_1 \mid \dots \mid \mathbf{x}_m]$ 을 세웁니다.
2. 가우스 소거법을 수행하여 **행 사다리꼴(REF)**로 변환합니다.
3. **피벗 열(Pivot Columns)**에 해당하는 원래 행렬 $A$ 의 열벡터를 추출합니다.

- **Example 2.17 백지 분석 (Eq 2.81~2.83)**:
  $$\mathbf{x}_1 = \begin{bmatrix} 1 \\ 2 \\ -1 \\ -1 \\ -1 \end{bmatrix}, \mathbf{x}_2 = \begin{bmatrix} 2 \\ -1 \\ 1 \\ 2 \\ -2 \end{bmatrix}, \mathbf{x}_3 = \begin{bmatrix} 3 \\ -4 \\ 3 \\ 5 \\ -3 \end{bmatrix}, \mathbf{x}_4 = \begin{bmatrix} -1 \\ 8 \\ -5 \\ -6 \\ 1 \end{bmatrix}$$
  - 계수 행렬 소거 결과 **1번째, 2번째, 4번째 열이 피벗 열**이 되므로, $\{\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_4\}$ 가 부분공간 $U$ 의 **기저(Basis)**가 됩니다!

## 2. ⚔️ Section 2.6.2: Rank (행렬의 계수)

### 📌 1. Rank(계수)의 정의와 대칭성 (Definition 2.17 & Remark p.47)
행렬 $A \in \mathbb{R}^{m \times n}$ 의 **선형독립인 열(Column)의 개수**를 행렬의 계수(Rank)라 부르며 $\text{rk}(A)$ 로 표기합니다.

- **대칭성 (Fundamental Theorem of Rank)**:
  $$\text{rk}(A) = \text{rk}(A^\top)$$
  **독립된 열의 개수(Column Rank)와 독립된 행의 개수(Row Rank)는 무조건 정확하게 일치**합니다!

### 📌 2. Rank의 핵심 주요 성질
- $\text{rk}(A) \le \min(m, n)$: Rank는 행 또는 열의 최소 크기를 초과할 수 없습니다.
- **Full Rank (만적 계수)**: $\text{rk}(A) = \min(m, n)$ 일 때 행렬이 손실 없이 정보로 꽉 차있음을 뜻합니다.
- **Rank Deficient (계수 결손)**: $\text{rk}(A) < \min(m, n)$ 일 때 정보 중복 및 손실이 발생했음을 뜻합니다.
- **가역성 조건**: 정방행렬 $A \in \mathbb{R}^{n \times n}$ 이 가역(Invertible)일 필요충분조건은 **$\text{rk}(A) = n$ (Full Rank)** 입니다.
- **해공간 차원**: $A\mathbf{x} = \mathbf{0}$ 동차계의 해공간(Kernel)의 차원은 **$n - \text{rk}(A)$** 가 됩니다.

### 📌 3. MML 원문 수치 계산 예제 (Example 2.18 & Eq 2.84)
$$A = \begin{bmatrix} 1 & 2 & 1 \\ -2 & -3 & 1 \\ 3 & 5 & 0 \end{bmatrix} \xrightarrow{\text{REF}} \begin{bmatrix} \mathbf{1} & 2 & 1 \\ 0 & \mathbf{1} & 3 \\ 0 & 0 & 0 \end{bmatrix}$$

피벗의 개수가 2개이므로, 이 행렬의 알짜배기 독립 차원 수는 **$\text{rk}(A) = 2$** 가 됩니다!

## 🚀 3. 4단계 실전 AI / 머신러닝 연결고리
- **SVD (이상값 분해) & Low-Rank Approximation (LoRA)**:
  - LLM 초거대 모델 파인튜닝 시 전체 가중치 업데이트 행렬 $W$ 대신 Low-rank 행렬 $A \times B$ ($\text{rank} \ll d$) 로 분해하여 파라미터 메모리를 99% 절약하는 기술의 핵심 근거가 됩니다.
