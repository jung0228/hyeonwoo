# 📐 2.2 Matrices (행렬 대수)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.2 완전 해부

---

## 1. ⚔️ 4단계 개념 구조화

### 1️⃣ [1단계 명확한 개념 정의]
- 행렬 (Matrix): $m \times n$ 개 실수를 격자 형태로 나열한 실수 표 $A \in \mathbb{R}^{m \times n}$.
- 행렬 곱셈 (Matrix Multiplication): $C = AB \iff c_{ij} = \sum_{k=1}^n a_{ik} b_{kj}$

---

### 2️⃣ [2단계 존재 이유 & 기하학적 본질]
- 선형 사상의 합성 (Composition of Linear Mappings):
  - 행렬 곱셈 $AB$는 단순 숫자 연산이 아니라, 먼저 변환 $B$를 공간에 적용한 뒤 연속해서 변환 $A$를 적용하는 공간 변환의 연쇄 합성임.
- 역행렬 (Inverse Matrix $A^{-1}$):
  - 변환 $A$를 거쳐 변형된 공간을 정확히 반대로 돌리는 원상복구 역변환 ($A A^{-1} = I_n$).

---

### 3️⃣ [3단계 상황별 직관 & 대수적 성질]
- 결합법칙 성립: $(AB)C = A(BC)$ (연속 공간 변환의 순서는 보존됨).
- 교환법칙 성립 안 함: $AB \neq BA$ (회전/수축 변환의 순서를 바꾸면 전혀 다른 공간 상태가 됨).
- 전치행렬 (Transpose $A^T$): $(AB)^T = B^T A^T$ (변환 순서가 반대로 뒤집힘).

---

### 4️⃣ [4단계 실전 AI 연결고리]
- 신경망의 딥 레이어 (Deep Neural Net Layers): $y = W_3 \sigma(W_2 \sigma(W_1 x + b_1) + b_2) + b_3$ 연쇄 행렬 곱셈 연산의 근간.

---

## 🔍 2. ★ MML 교재 원문 예시 해부 (Section 2.2 Examples)

### 📌 Example 2.2 (MML 원문: 3차원 선형계 가우스 소거법)
MML 교재 2.2절 원문 Example 2.2:
> *"If we want to solve the system of linear equations:"*
$$\begin{aligned}
x_1 + 2x_2 + x_3 &= 1 \\\\
2x_1 + 3x_2 + 4x_3 &= 3 \\\\
x_1 + 4x_2 - 2x_3 &= -1
\end{aligned}$$
- 증대 행렬 (Augmented Matrix):
  $$[A \mid b] = \begin{bmatrix} 1 & 2 & 1 & \mid & 1 \\\\ 2 & 3 & 4 & \mid & 3 \\\\ 1 & 4 & -2 & \mid & -1 \end{bmatrix}$$
- 가우스 소거법 (Gauss Elimination):
  1. $R_2 \leftarrow R_2 - 2R_1, R_3 \leftarrow R_3 - R_1 \implies \begin{bmatrix} 1 & 2 & 1 & \mid & 1 \\\\ 0 & -1 & 2 & \mid & 1 \\\\ 0 & 2 & -3 & \mid & -2 \end{bmatrix}$
  2. $R_3 \leftarrow R_3 + 2R_2 \implies \begin{bmatrix} 1 & 2 & 1 & \mid & 1 \\\\ 0 & -1 & 2 & \mid & 1 \\\\ 0 & 0 & 1 & \mid & 0 \end{bmatrix}$ (REF 상삼각 행렬 완성)
- 후방 대입법 (Back-Substitution):
  - $x_3 = 0, x_2 = -1, x_1 = 3 \implies \mathbf{x = [3, -1, 0]^T}$ (3개 초평면의 유일한 교점).

---

### 📌 Example 2.3 (MML 원문: 행렬 곱셈 연산 $A \in \mathbb{R}^{3 \times 2}, B \in \mathbb{R}^{2 \times 3}$)
MML 교재 2.2절 원문 Example 2.3:
> *"For the matrices $A = \begin{bmatrix} 1 & 2 \\\\ 0 & 1 \\\\ 3 & 0 \end{bmatrix}$ and $B = \begin{bmatrix} 2 & 1 & 0 \\\\ 1 & 3 & 4 \end{bmatrix}$, calculate $AB$ and $BA$."*

- $AB \in \mathbb{R}^{3 \times 3}$ 계산:
  $$AB = \begin{bmatrix} 1(2)+2(1) & 1(1)+2(3) & 1(0)+2(4) \\\\ 0(2)+1(1) & 0(1)+1(3) & 0(0)+1(4) \\\\ 3(2)+0(1) & 3(1)+0(3) & 3(0)+0(4) \end{bmatrix} = \begin{bmatrix} 4 & 7 & 8 \\\\ 1 & 3 & 4 \\\\ 6 & 3 & 0 \end{bmatrix}$$
- $BA \in \mathbb{R}^{2 \times 2}$ 계산:
  $$BA = \begin{bmatrix} 2(1)+1(0)+0(3) & 2(2)+1(1)+0(0) \\\\ 1(1)+3(0)+4(3) & 1(2)+3(1)+4(0) \end{bmatrix} = \begin{bmatrix} 2 & 5 \\\\ 13 & 5 \end{bmatrix}$$
- MML 교재 인사이트: $AB \neq BA$ 일 뿐만 아니라, 차원 자체가 $3 \times 3$ 과 $2 \times 2$ 로 완전히 다름! (행렬 곱셈 교환법칙 절대 불성립 증명).

---

### 📌 Example 2.4 (MML 원문: $2 \times 2$ 행렬의 역행렬 유도)
MML 교재 2.2절 원문 Example 2.4:
> *"For a matrix $A = \begin{bmatrix} a & b \\\\ c & d \end{bmatrix} \in \mathbb{R}^{2 \times 2}$, find its inverse $A^{-1}$."*

- 역행렬 수식 유도:
  $$A^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\\\ -c & a \end{bmatrix}$$
- $\det(A) = ad - bc = 0$ 이면 분모가 0이 되어 역행렬 존재 안 함 (특이 행렬).
