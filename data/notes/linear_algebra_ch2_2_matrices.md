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
