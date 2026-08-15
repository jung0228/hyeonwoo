# 📐 2.2 Matrices (행렬 대수)

> **POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.2 원문 완전 대조 노트**

---

## 1. 🌐 Definition & Basic Concepts (행렬의 정의와 기본 개념)

### 📌 Definition 2.1 (Matrix 행렬)
$m, n \in \mathbb{R}$ 에 대해 실숫값 $(m, n)$-행렬 $A$는 $m$개의 행(Row)과 $n$개의 열(Column)로 구성된 직사각형 배열입니다 (Equation 2.11):

$$A = \begin{bmatrix} a_{11} & a_{12} & \dots & a_{1n} \\\\ a_{21} & a_{22} & \dots & a_{2n} \\\\ \vdots & \vdots & \ddots & \vdots \\\\ a_{m1} & a_{m2} & \dots & a_{mn} \end{bmatrix}, \quad a_{ij} \in \mathbb{R}$$

- **Row Vector / Column Vector**: $(1, n)$-행렬을 행 벡터(Row Vector), $(m, 1)$-행렬을 열 벡터(Column Vector)라 부릅니다.
- **Vector Stacking (Reshape)**: 모든 실수 $(m, n)$-행렬의 집합을 $\mathbb{R}^{m \times n}$ 이라 표기하며, 행렬 $A \in \mathbb{R}^{m \times n}$ 의 $n$개 열을 수직으로 쌓아 긴 벡터 $\mathbf{a} \in \mathbb{R}^{mn}$ 으로 동등하게 표현할 수 있습니다 (Figure 2.4).

---

## 2. ⚔️ Section 2.2.1: Matrix Addition and Multiplication (덧셈과 곱셈 연산)

### 📌 행렬의 덧셈 (Matrix Addition: Eq 2.12)
동일한 크기의 두 행렬 $A, B \in \mathbb{R}^{m \times n}$ 의 합 $A + B$는 성별 성분(Element-wise)의 합으로 정의됩니다:

$$A + B := \begin{bmatrix} a_{11} + b_{11} & \dots & a_{1n} + b_{1n} \\\\ \vdots & \ddots & \vdots \\\\ a_{m1} + b_{m1} & \dots & a_{mn} + b_{mn} \end{bmatrix} \in \mathbb{R}^{m \times n}$$

### 📌 행렬의 곱셈 (Matrix Multiplication: Eq 2.13 & 2.14)
$A \in \mathbb{R}^{m \times n}, B \in \mathbb{R}^{n \times k}$ 일 때, 곱행렬 $C = AB \in \mathbb{R}^{m \times k}$ 의 성분 $c_{ij}$는 $A$의 $i$번째 행과 $B$의 $j$번째 열의 내적(Dot Product)으로 계산됩니다:

$$c_{ij} = \sum_{l=1}^n a_{il} b_{lj}, \quad i = 1, \dots, m, \quad j = 1, \dots, k$$

- **차원 매칭 조건 (Remark)**: 행렬 곱셈은 인접한 차원(Neighboring dimensions)이 일치할 때만 가능합니다:
  $$A_{n \times k} \cdot B_{k \times m} = C_{n \times m}$$
- **Hadamard Product (하다마르 곱)과의 구분**: 행렬 곱셈은 성분별 곱이 아닙니다. 프로그래밍 언어에서 배열끼리의 성분별 곱은 하다마르 곱(Hadamard Product, $A \odot B$)이라 부릅니다.

---

### 📌 Example 2.3 (MML 원문: 행렬 곱셈 교환법칙 불성립 증명)

$A = \begin{bmatrix} 1 & 2 & 3 \\\\ 3 & 2 & 1 \end{bmatrix} \in \mathbb{R}^{2 \times 3}$, $B = \begin{bmatrix} 0 & 2 \\\\ 1 & -1 \\\\ 0 & 1 \end{bmatrix} \in \mathbb{R}^{3 \times 2}$ 일 때:

$$AB = \begin{bmatrix} 1 & 2 & 3 \\\\ 3 & 2 & 1 \end{bmatrix} \begin{bmatrix} 0 & 2 \\\\ 1 & -1 \\\\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 2 & 3 \\\\ 2 & 5 \end{bmatrix} \in \mathbb{R}^{2 \times 2} \quad (2.15)$$

$$BA = \begin{bmatrix} 0 & 2 \\\\ 1 & -1 \\\\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 2 & 3 \\\\ 3 & 2 & 1 \end{bmatrix} = \begin{bmatrix} 6 & 4 & 2 \\\\ -2 & 0 & 2 \\\\ 3 & 2 & 1 \end{bmatrix} \in \mathbb{R}^{3 \times 3} \quad (2.16)$$

- **인사이트**: 행렬 곱셈은 교환법칙이 성립하지 않습니다 ($AB \neq BA$). 심지어 $AB$와 $BA$의 연산 결과 차원($2 \times 2$ vs $3 \times 3$) 자체가 다릅니다 (Figure 2.5).

---

### 📌 Definition 2.2 (Identity Matrix 단위행렬) 및 대수적 성질
주대각선 성분이 모두 1이고 나머지가 0인 $n \times n$ 행렬을 단위행렬 $I_n$ 이라 정의합니다.

- **결합법칙 (Associativity)**: $(AB)C = A(BC) \quad (2.18)$
- **분배법칙 (Distributivity)**: $(A+B)C = AC + BC$, $A(C+D) = AC + AD \quad (2.19a, 2.19b)$
- **단위행렬 곱셈**: $I_m A = A I_n = A \quad (2.20)$

---

## 3. ⚔️ Section 2.2.2: Inverse and Transpose (역행렬과 전치행렬)

### 📌 Definition 2.3 (Inverse 역행렬)
정방행렬 $A \in \mathbb{R}^{n \times n}$ 에 대해 $AB = I_n = BA$ 를 만족하는 $B \in \mathbb{R}^{n \times n}$ 가 존재할 때, $B$를 $A$의 역행렬(Inverse)이라 부르고 $A^{-1}$ 로 표기합니다.
- 역행렬이 존재하면 가역(Invertible/Regular/Nonsingular)이라 부르고, 존재하지 않으면 특이(Singular/Noninvertible)라 부릅니다. 역행렬이 존재할 때 이는 유일합니다.

---

### 📌 Remark: $2 \times 2$ 행렬의 역행렬 공식 (Equation 2.21 ~ 2.24)

$A = \begin{bmatrix} a_{11} & a_{12} \\\\ a_{21} & a_{22} \end{bmatrix} \in \mathbb{R}^{2 \times 2}$ 에 대해 $A' = \begin{bmatrix} a_{22} & -a_{12} \\\\ -a_{21} & a_{11} \end{bmatrix}$ 를 곱하면:

$$AA' = (a_{11}a_{22} - a_{12}a_{21}) I$$

따라서 $a_{11}a_{22} - a_{12}a_{21} \neq 0$ 일 때만 역행렬이 존재합니다:

$$A^{-1} = \frac{1}{a_{11}a_{22} - a_{12}a_{21}} \begin{bmatrix} a_{22} & -a_{12} \\\\ -a_{21} & a_{11} \end{bmatrix} \quad (2.24)$$

---

### 📌 Example 2.4 (Inverse Matrix 수치 예시: Eq 2.25)

$$A = \begin{bmatrix} 1 & 2 & 1 \\\\ 4 & 4 & 5 \\\\ 6 & 7 & 7 \end{bmatrix}, \quad B = \begin{bmatrix} -7 & -7 & 6 \\\\ 2 & 1 & -1 \\\\ 4 & 5 & -4 \end{bmatrix}$$

두 행렬은 $AB = I_3 = BA$ 를 만족하므로 서로의 역행렬 관계입니다 ($B = A^{-1}$).

---

### 📌 Definition 2.4 (Transpose 전치행렬) & 주요 성질 (Eq 2.26 ~ 2.31)
$A \in \mathbb{R}^{m \times n}$ 의 행과 열을 뒤집은 $B \in \mathbb{R}^{n \times m} \ (b_{ij} = a_{ji})$ 를 전치행렬이라 부르며 $A^\top$ 로 표기합니다.

- $A A^{-1} = I = A^{-1} A \quad (2.26)$
- $(AB)^{-1} = B^{-1} A^{-1} \quad (2.27)$
- $(A + B)^{-1} \neq A^{-1} + B^{-1} \quad (2.28)$
- $(A^\top)^\top = A \quad (2.29)$
- $(AB)^\top = B^\top A^\top \quad (2.30)$
- $(A + B)^\top = A^\top + B^\top \quad (2.31)$

---

### 📌 Definition 2.5 (Symmetric Matrix 대칭행렬)
$A = A^\top$ 인 정방행렬을 대칭행렬(Symmetric Matrix)이라 부릅니다.
- 대칭행렬의 합 $A + B$는 항상 대칭행렬이지만, **곱 $AB$는 일반적으로 대칭행렬이 아닙니다** (Eq 2.32 Remark).

---

## 4. ⚔️ Section 2.2.3 & 2.2.4: Scalar Multiplication & Compact Representation

### 📌 Example 2.5 (MML 원문: 스칼라 배의 분배법칙 증명 - Eq 2.33 & 2.34)
$C = \begin{bmatrix} 1 & 2 \\\\ 3 & 4 \end{bmatrix}$ 일 때, 임의의 스칼라 $\lambda, \psi \in \mathbb{R}$ 에 대해:

$$(\lambda + \psi) C = \begin{bmatrix} (\lambda + \psi)1 & (\lambda + \psi)2 \\\\ (\lambda + \psi)3 & (\lambda + \psi)4 \end{bmatrix} = \begin{bmatrix} \lambda & 2\lambda \\\\ 3\lambda & 4\lambda \end{bmatrix} + \begin{bmatrix} \psi & 2\psi \\\\ 3\psi & 4\psi \end{bmatrix} = \lambda C + \psi C$$

---

### 📌 Section 2.2.4: 선형방정식계의 컴팩트 행렬 표현 (Equation 2.35 & 2.36)

다음 연립방정식계:
$$\begin{aligned}
2x_1 + 3x_2 + 5x_3 &= 1 \\\\
4x_1 - 2x_2 - 7x_3 &= 8 \\\\
9x_1 + 5x_2 - 3x_3 &= 2
\end{aligned}$$

행렬 곱셈 규칙을 적용하여 컴팩트한 행렬 형태 $Ax = b$ 로 표기합니다:

$$\begin{bmatrix} 2 & 3 & 5 \\\\ 4 & -2 & -7 \\\\ 9 & 5 & -3 \end{bmatrix} \begin{bmatrix} x_1 \\\\ x_2 \\\\ x_3 \end{bmatrix} = \begin{bmatrix} 1 \\\\ 8 \\\\ 2 \end{bmatrix} \quad (2.36)$$

여기서 $x_1$은 1번째 열, $x_2$는 2번째 열, $x_3$는 3번째 열을 스칼라 배하여 더하는 열벡터들의 선형 결합(Linear Combination)을 형성합니다.
