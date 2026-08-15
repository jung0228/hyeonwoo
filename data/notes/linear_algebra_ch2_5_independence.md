# 📐 2.5 Linear Independence (선형 독립)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.5 완전 해부

---

## 1. ⚔️ 4단계 개념 구조화

### 1️⃣ [1단계 명확한 개념 정의]
- 선형 결합 (Linear Combination): 벡터 집합 $v_1, \dots, v_k$와 스칼라 $c_1, \dots, c_k$에 대해 $v = \sum c_i v_i$.
- Span (생성): 벡터들의 선형 결합으로 생성 가능한 전체 부분공간 $\text{span}(v_1, \dots, v_k)$.
- 선형 독립 (Linear Independence): 
  - $$\sum_{i=1}^k c_i v_i = 0 \iff c_1 = c_2 = \dots = c_k = 0$$
  - 어떤 벡터도 다른 벡터들의 선형 결합으로 표현할 수 없는 상태.

---

### 2️⃣ [2단계 존재 이유]
- 데이터 중복 제거: 백터 간 종속(Dependence) 관계가 있으면 정보가 중복되어 행렬 랭크가 떨어지고 역행렬이 파탄 남.

---

### 3️⃣ [3단계 상황별 직관 & 맹점]
- 선형 종속 (Linear Dependence): 한 벡터가 다른 벡터들의 평면/직선 상에 얹혀 있어 새로운 차원을 제공하지 못함.

---

### 4️⃣ [4단계 실전 AI 연결고리]
- 다중공선성 (Multicollinearity): AI 특징(Feature) 간 선형 종속 시 모델의 가중치 추정이 불가능해지므로 L2 규제(Ridge)를 적용함.

---

## 🔍 2. ★ MML 교재 원문 예시 해부 (Examples 2.9 ~ 2.11)

### 📌 Example 2.9 (3개 2차원 벡터의 선형 종속성 판별)
MML 교재 2.5절의 Example 2.9에서는 $2$차원 공간 $\mathbb{R}^2$ 상의 3개 벡터 $x_1 = \begin{bmatrix} 1 \\\\ 2 \end{bmatrix}, x_2 = \begin{bmatrix} 3 \\\\ 4 \end{bmatrix}, x_3 = \begin{bmatrix} 5 \\\\ 6 \end{bmatrix}$ 의 독립성 판별합니다:
- 미지수 개수 $3$ > 공간 차원 $2 \implies$ 무조건 선형 종속(Linearly Dependent)!
- 계수 행렬 $\begin{bmatrix} 1 & 3 & 5 \\\\ 2 & 4 & 6 \end{bmatrix}$ 소거 시 피벗 2개, 자유 변수 1개 발생으로 $x_3$를 $x_1, x_2$의 선형 결합으로 표현 가능 ($x_3 = -x_1 + 2x_2$).

---

### 📌 Example 2.10 (3개 3차원 벡터의 선형 독립성 증명)
MML 교재 2.5절의 Example 2.10에서는 $3$차원 벡터 $v_1 = \begin{bmatrix} 1 \\\\ 0 \\\\ 1 \end{bmatrix}, v_2 = \begin{bmatrix} 0 & 1 & 0 \end{bmatrix}^T, v_3 = \begin{bmatrix} 0 \\\\ 0 \\\\ 1 \end{bmatrix}$ 판별합니다:
- $\sum c_i v_i = 0 \implies \begin{bmatrix} c_1 \\\\ c_2 \\\\ c_1 + c_3 \end{bmatrix} = \begin{bmatrix} 0 \\\\ 0 \\\\ 0 \end{bmatrix} \implies c_1=0, c_2=0, c_3=0$
- 유일해 $c_i=0$ 만 존재 ➡️ 완벽한 선형 독립(Linearly Independent)!

---

### 📌 Example 2.11 (행렬 랭크를 통한 선형독립 자동 판별법)
MML 교재 2.5절의 Example 2.11에서는 행렬 $V = [v_1, \dots, v_k]$ 의 피벗 개수(Rank)를 이용한 판별 규칙 정립:
- $\text{Rank}(V) = k \implies$ 선형 독립 (Full Column Rank)
- $\text{Rank}(V) < k \implies$ 선형 종속 (Rank Deficient)
