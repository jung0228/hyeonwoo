# 📐 2.2 Matrices (행렬 대수)

> **POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.2 완전 해부**

---

## 1. ⚔️ 4단계 개념 구조화

### 1️⃣ [1단계 명확한 개념 정의]
- **행렬 (Matrix)**: $m \times n$ 개 실수를 격자 형태로 나열한 실수 표 $A \in \mathbb{R}^{m \times n}$.
- **행렬 곱셈 (Matrix Multiplication)**: $C = AB \iff c_{ij} = \sum_{k=1}^n a_{ik} b_{kj}$

---

### 2️⃣ [2단계 존재 이유 & 기하학적 본질]
- **선형 사상의 합성 (Composition of Linear Mappings)**:
  - 행렬 곱셈 $AB$는 단순 숫자 연산이 아니라, 먼저 변환 $B$를 공간에 적용한 뒤 연속해서 변환 $A$를 적용하는 **공간 변환의 연쇄 합성**임.
- **역행렬 (Inverse Matrix $A^{-1}$)**:
  - 변환 $A$를 거쳐 변형된 공간을 정확히 반대로 돌리는 원상복구 역변환 ($A A^{-1} = I_n$).

---

### 3️⃣ [3단계 상황별 직관 & 대수적 성질]
- **결합법칙 성립**: $(AB)C = A(BC)$ (연속 공간 변환의 순서는 보존됨).
- **교환법칙 성립 안 함**: $AB \neq BA$ (회전/수축 변환의 순서를 바꾸면 전혀 다른 공간 상태가 됨).
- **전치행렬 (Transpose $A^T$)**: $(AB)^T = B^T A^T$ (변환 순서가 반대로 뒤집힘).

---

### 4️⃣ [4단계 실전 AI 연결고리]
- **신경망의 딥 레이어 (Deep Neural Net Layers)**: $y = W_3 \sigma(W_2 \sigma(W_1 x + b_1) + b_2) + b_3$ 연쇄 행렬 곱셈 연산의 근간.

---

## 🔍 2. ★ MML 교재 원문 예시 해부

### 📌 Example 2.3 (행렬 곱셈과 선형 사상의 합성)
MML 교재 2.2절의 **Example 2.3**에서는 행렬 곱셈 $C = AB$의 각 성분이 공간 변환의 합성이 됨을 직접 유도함:
- 두 변환 $B \in \mathbb{R}^{2 \times 3}, A \in \mathbb{R}^{3 \times 2}$:
  $$A = \begin{bmatrix} 1 & 2 \\\\ 0 & 1 \\\\ 3 & 0 \end{bmatrix}, \quad B = \begin{bmatrix} 2 & 1 & 0 \\\\ 1 & 3 & 4 \end{bmatrix}$$
- 곱행렬 $AB \in \mathbb{R}^{3 \times 3}$:
  $$AB = \begin{bmatrix} 1(2)+2(1) & 1(1)+2(3) & 1(0)+2(4) \\\\ 0(2)+1(1) & 0(1)+1(3) & 0(0)+1(4) \\\\ 3(2)+0(1) & 3(1)+0(3) & 3(0)+0(4) \end{bmatrix} = \begin{bmatrix} 4 & 7 & 8 \\\\ 1 & 3 & 4 \\\\ 6 & 3 & 0 \end{bmatrix}$$
- **직관**: 입력 3차원 벡터를 2차원으로 축소($B$)한 뒤 다시 3차원으로 팽창($A$)시키는 공간 연속 변환의 수식화!

---

### 📌 Example 2.4 (2x2 행렬의 역행렬과 가역성 판별)
MML 교재 2.2절의 **Example 2.4**에서는 $2 \times 2$ 행렬 $A = \begin{bmatrix} a & b \\\\ c & d \end{bmatrix}$ 의 역행렬 공식을 유도함:
- **역행렬 존재 조건**: $\det(A) = ad - bc \neq 0$
- **역행렬 공식**:
  $$A^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\\\ -c & a \end{bmatrix}$$
- 만약 $ad - bc = 0$ 이면 행렬 $A$는 **특이 행렬(Singular Matrix)**이 되어 공간이 1차원 선 이하로 찌그러져 원상복구 역변환이 불가능해짐!
