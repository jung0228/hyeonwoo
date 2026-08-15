# 📐 2.3 Solving Systems of Linear Equations (선형방정식계의 풀이)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.3 완전 해부

---

## 1. ⚔️ Section 2.3.1: Particular and General Solution (특수해와 일반해)
- 해 공간 공식: $\mathbf{x = x_p + x_h}$
  - 특수해 $x_p$: $A x_p = b$를 만족하는 특정 구체적 점.
  - 동차해 $x_h$: $A x_h = 0$을 만족하는 행렬 $A$의 Nullspace (Kernel) 부분공간.
- 기하학적 아핀 공간 구조: 원점을 지나는 부분공간 $x_h$가 특수해 $x_p$ 점만큼 평행 이동된 아핀 공간(Affine Subspace)이 해 공간임.

---

## 2. ⚔️ Section 2.3.2: Elementary Transformations (기본 행 연산 & 가우스 소거법)
- 3가지 ERO (Elementary Row Operations):
  1. Exchange (행 교환): $R_i \leftrightarrow R_j$
  2. Scaling (스칼라 배): $R_i \leftarrow c R_i \ (c \neq 0)$
  3. Addition (행 더하기): $R_i \leftarrow R_i + c R_j$
- REF vs RREF: RREF는 피벗이 $1$이고 피벗 열의 다른 성분이 모두 $0$.
- MML 특수 기법 (The Minus-1 Trick):
  - RREF에서 자유 변수 위치에 $-1$을 채워 넣고 Nullspace 기저를 암산으로 읽어내는 MML 독점 팁.

---

## 3. 🔍 MML 교재 원문 예시 해부 (Section 2.3 Examples)

### 📌 Example 2.5 (MML 원문: 무수히 많은 해를 가지는 3차원 선형계)
MML 교재 2.3절 원문 Example 2.5:
> *"Find the solution space of the system of linear equations:"*
$$\begin{aligned}
x_1 + x_2 + x_3 + x_4 &= 3 \\\\
2x_1 + 3x_2 + x_3 + 2x_4 &= 7 \\\\
x_1 - x_2 + 3x_3 + x_4 &= -1
\end{aligned}$$
- RREF 변환 결과:
  $$\begin{bmatrix} 1 & 0 & 2 & 1 & \mid & 2 \\\\ 0 & 1 & -1 & 0 & \mid & 1 \\\\ 0 & 0 & 0 & 0 & \mid & 0 \end{bmatrix}$$
- 일반해 (General Solution):
  $$\mathbf{x} = \begin{bmatrix} 2 \\\\ 1 \\\\ 0 \\\\ 0 \end{bmatrix} + x_3 \begin{bmatrix} -2 \\\\ 1 \\\\ 1 \\\\ 0 \end{bmatrix} + x_4 \begin{bmatrix} -1 \\\\ 0 \\\\ 0 \\\\ 1 \end{bmatrix}, \quad x_3, x_4 \in \mathbb{R}$$
- MML 교재 구조 분석: 특수해 $\mathbf{x_p} = [2, 1, 0, 0]^T$ 점에 동차해 부분공간 $\mathbf{x_h} = \text{span}([-2, 1, 1, 0]^T, [-1, 0, 0, 1]^T)$ 이 더해진 평면 모양의 아핀 공간(Affine Subspace)!

---

### 📌 Example 2.6 (MML 원문: 가우스 소거법 단계별 전개)
MML 교재 2.3절 원문 Example 2.6:
> *"Transform the matrix $A = \begin{bmatrix} 1 & 2 & 1 \\\\ 2 & 3 & 4 \\\\ 1 & 4 & -2 \end{bmatrix}$ into RREF."*
- RREF 가우스-조던 소거 결과:
  $$\begin{bmatrix} 1 & 2 & 1 \\\\ 2 & 3 & 4 \\\\ 1 & 4 & -2 \end{bmatrix} \xrightarrow{\text{ERO}} \begin{bmatrix} 1 & 0 & 0 \\\\ 0 & 1 & 0 \\\\ 0 & 0 & 1 \end{bmatrix} = I_3$$
- 결론: 행렬 $A$의 랭크 $\text{Rank}(A) = 3$ (Full Rank) 이며 완벽히 단위행렬 $I_3$ 로 소거됨!

---

### 📌 Example 2.7 (MML 원문: The Minus-1 Trick으로 Nullspace 암산 추출)
MML 교재 2.3절 원문 Example 2.7:
> *"Find the Kernel (Nullspace) of the matrix $A = \begin{bmatrix} 1 & 2 & 0 & 1 \\\\ 0 & 0 & 1 & 2 \end{bmatrix}$ using the minus-1 trick."*

- 자유 변수열 위치 확인: 2열, 4열이 피벗이 없는 자유 변수열.
- $-1$ Trick 행 삽입 행렬 $\tilde{A}$:
  $$\tilde{A} = \begin{bmatrix} 1 & 2 & 0 & 1 \\\\ 0 & \mathbf{-1} & 0 & 0 \\\\ 0 & 0 & 1 & 2 \\\\ 0 & 0 & 0 & \mathbf{-1} \end{bmatrix}$$
- Nullspace 기저 (Kernel Basis): $-1$이 삽입된 2열과 4열의 벡터:
  $$\text{Nullspace}(A) = \text{span}\left( \begin{bmatrix} 2 \\\\ -1 \\\\ 0 \\\\ 0 \end{bmatrix}, \begin{bmatrix} 1 \\\\ 0 \\\\ 2 \\\\ -1 \end{bmatrix} \right)$$

---

### 📌 Example 2.8 (MML 원문: 가우스-조던 $[A \mid I] \to [I \mid A^{-1}]$ 역행렬 계산)
MML 교재 2.3절 원문 Example 2.8:
> *"Compute the inverse of $A = \begin{bmatrix} 1 & 2 & 1 \\\\ 2 & 3 & 4 \\\\ 1 & 4 & -2 \end{bmatrix}$ using Gauss-Jordan elimination."*

- 증대행렬 소거 전개:
  $$[A \mid I_3] = \begin{bmatrix} 1 & 2 & 1 & \mid & 1 & 0 & 0 \\\\ 2 & 3 & 4 & \mid & 0 & 1 & 0 \\\\ 1 & 4 & -2 & \mid & 0 & 0 & 1 \end{bmatrix} \xrightarrow{\text{ERO}} \begin{bmatrix} 1 & 0 & 0 & \mid & -22 & 8 & 5 \\\\ 0 & 1 & 0 & \mid & 8 & -3 & -2 \\\\ 0 & 0 & 1 & \mid & 5 & -2 & -1 \end{bmatrix}$$
- 최종 역행렬:
  $$A^{-1} = \begin{bmatrix} -22 & 8 & 5 \\\\ 8 & -3 & -2 \\\\ 5 & -2 & -1 \end{bmatrix}$$
