# 📐 2.3 Solving Systems of Linear Equations (선형방정식계의 풀이)

> **POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.3 완전 해부**

---

## 1. ⚔️ Section 2.3.1: Particular and General Solution (특수해와 일반해)
- **해 공간 공식**: $\mathbf{x = x_p + x_h}$
  - **특수해 $x_p$**: $A x_p = b$를 만족하는 특정 구체적 점.
  - **동차해 $x_h$**: $A x_h = 0$을 만족하는 행렬 $A$의 **Nullspace (Kernel)** 부분공간.
- **기하학적 아핀 공간 구조**: 원점을 지나는 부분공간 $x_h$가 특수해 $x_p$ 점만큼 평행 이동된 **아핀 공간(Affine Subspace)**이 해 공간임.

---

## 2. ⚔️ Section 2.3.2: Elementary Transformations (기본 행 연산 & 가우스 소거법)
- **3가지 ERO (Elementary Row Operations)**:
  1. **Exchange (행 교환)**: $R_i \leftrightarrow R_j$
  2. **Scaling (스칼라 배)**: $R_i \leftarrow c R_i \ (c \neq 0)$
  3. **Addition (행 더하기)**: $R_i \leftarrow R_i + c R_j$
- **REF vs RREF**: RREF는 피벗이 $1$이고 피벗 열의 다른 성분이 모두 $0$.
- **MML 특수 기법 (The Minus-1 Trick)**:
  - RREF에서 자유 변수 위치에 $-1$을 채워 넣고 Nullspace 기저를 암산으로 읽어내는 MML 독점 팁.

---

## 3. 📝 MML 교재 연습문제 전수 풀이

### 📌 [Problem 1] Ex 2.1 - 가우스 소거 유일해
- $[A \mid b] \to \text{RREF} \implies \mathbf{x = [3, -1, 0]^T}$.

### 📌 [Problem 2] Ex 2.2 - Inconsistent System (해 없음)
- $0 = 1$ 모순, $\text{Rank}(A)=1 < \text{Rank}([A \mid b])=2 \implies b \notin \text{Col}(A)$.
