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

## 3. 🔍 MML 교재 원문 예시 해부 (Examples 2.5 ~ 2.8)

### 📌 Example 2.5 (무수히 많은 해를 가지는 3차원 선형계)
MML 교재 2.3절의 **Example 2.5**에서는 피벗이 부족해 자유 변수가 생기는 경우를 전개함:
- 계수 행렬이 피벗 2개만 가져 자유 변수 $x_3$ 발생.
- **일반해 (General Solution)**:
  $$\mathbf{x} = \begin{bmatrix} 1 \\\\ 2 \\\\ 0 \end{bmatrix} + x_3 \begin{bmatrix} -2 \\\\ 1 \\\\ 1 \end{bmatrix}, \quad x_3 \in \mathbb{R}$$
- **구조**: 특수해 $\mathbf{x_p} = [1, 2, 0]^T$ 점에 동차해 부분공간 $\mathbf{x_h} = \text{span}([-2, 1, 1]^T)$ 이 더해진 직선 모양의 **아핀 공간(Affine Subspace)**!

---

### 📌 Example 2.6 (가우스-조던 소거법과 RREF 변환)
MML 교재 2.3절의 **Example 2.6**에서는 행렬 $A$를 RREF(Reduced Row Echelon Form)로 만드는 가우스-조던 과정 전개:
- **피벗열 성분 0화**: 피벗 위치 위아래 성분을 모두 0으로 만들어 최종 식을 $x_1 = c_1, x_2 = c_2$ 형태의 단위 피벗열로 완전 정리.

---

### 📌 Example 2.7 (The Minus-1 Trick으로 Nullspace 암산 구하기)
MML 교재 2.3절의 **Example 2.7**에서는 RREF 형태에서 $-1$을 채워 채우는 MML 독점 기법 전개:
- **RREF 행렬**: $\begin{bmatrix} 1 & 3 & 0 & 2 \\\\ 0 & 0 & 1 & 4 \end{bmatrix}$ (자유 변수열: 2열, 4열)
- **$-1$ Trick 적용**: 2행과 4행 자리에 $-1$ 피벗 행을 삽입하여 $4 \times 4$ 행렬 구성:
  $$\tilde{A} = \begin{bmatrix} 1 & 3 & 0 & 2 \\\\ 0 & \mathbf{-1} & 0 & 0 \\\\ 0 & 0 & 1 & 4 \\\\ 0 & 0 & 0 & \mathbf{-1} \end{bmatrix}$$
- **Nullspace 기저**: $-1$이 들어간 2열과 4열 벡터인 $\begin{bmatrix} 3 \\\\ -1 \\\\ 0 \\\\ 0 \end{bmatrix}$ 과 $\begin{bmatrix} 2 \\\\ 0 \\\\ 4 \\\\ -1 \end{bmatrix}$ 이 곧바로 $Ax = 0$ 의 Nullspace 기저가 됨!

---

### 📌 Example 2.8 (역행렬 구하기: 가우스-조던 소거법 $[A \mid I] \to [I \mid A^{-1}]$)
MML 교재 2.3절의 **Example 2.8**에서는 행렬 $A$ 옆에 단위행렬 $I$를 붙인 증대행렬 연산 전개:
- $[A \mid I_n] \xrightarrow{\text{ERO}} [I_n \mid A^{-1}]$
- **본질**: $A x_i = e_i$ 시스템 $n$개를 동시에 소거하여 원상복구 역행렬 $A^{-1}$을 일괄 추정!
