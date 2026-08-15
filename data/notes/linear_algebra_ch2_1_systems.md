# 📐 2.1 Systems of Linear Equations (선형방정식계)

> **POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.1 완전 해부**

---

## 1. ⚔️ 4단계 개념 구조화

### 1️⃣ [1단계 명확한 개념 정의]
- **선형방정식계 (Systems of Linear Equations)**: $m$개의 방정식과 $n$개의 미지수 $x_1, \dots, x_n$으로 구성된 연립 1차 방정식계.
  $$\sum_{j=1}^n a_{ij} x_j = b_i \quad (i = 1, \dots, m) \iff A x = b$$

---

### 2️⃣ [2단계 기하학적 2가지 시각 (Row vs Column Picture)]
- **Row Picture (행 시각)**:
  - 각 행(Row)을 $n$차원 공간상의 **초평면(Hyperplane)** 방정식으로 해석함.
  - $Ax = b$를 푼다는 것은 **"모든 초평면들이 동시에 만나는 단 하나의 교점(Intersection)을 찾는 일"**.
- **Column Picture (열 시각 - AI/ML 핵심!)**:
  - 행렬 $A = \begin{bmatrix} a_1 & a_2 & \dots & a_n \end{bmatrix}$ 의 각 열벡터 $a_j$들에 가중치 $x_j$를 곱해 더하는 **선형 결합(Linear Combination)**.
  - $$a_1 x_1 + a_2 x_2 + \dots + a_n x_n = b$$
  - **★ 핵심 인사이트**: $Ax = b$에 해가 존재한다는 것은 **"결과 벡터 $b$가 행렬 $A$의 열벡터들이 펼치는 공간(Column Space)에 정확히 속한다($b \in \text{Col}(A)$)"**는 뜻임!

---

### 3️⃣ [3단계 상황별 직관 & 해의 3가지 가능성]
1. **유일해 (Unique Solution)**: 열공간 내에 $b$가 정확히 1개의 계수 조합으로 표현됨.
2. **무수히 많은 해 (Infinite Solutions)**: 열벡터 간 중복(Redundancy)이 존재하여 계수 자유 변수(Free Variable)가 발생함.
3. **해 없음 (Inconsistent System)**: 결과 벡터 $b$가 열공간 밖으로 튕겨 나감 ($b \notin \text{Col}(A)$).

---

### 4️⃣ [4단계 실전 AI 연결고리]
- **선형 회귀 (Linear Regression)**: $y = Xw$에서 노이즈로 해가 없을 때($y \notin \text{Col}(X)$), 정사영을 내려 최적 웨이트 **$w = (X^T X)^{-1} X^T y$** 추정.
