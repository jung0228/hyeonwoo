# 📐 2.5 Linear Independence (선형 독립)

> **POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.5 완전 해부**

---

## 1. ⚔️ 4단계 개념 구조화

### 1️⃣ [1단계 명확한 개념 정의]
- **선형 결합 (Linear Combination)**: 벡터 집합 $v_1, \dots, v_k$와 스칼라 $c_1, \dots, c_k$에 대해 $v = \sum c_i v_i$.
- **Span (생성)**: 벡터들의 선형 결합으로 생성 가능한 전체 부분공간 $\text{span}(v_1, \dots, v_k)$.
- **선형 독립 (Linear Independence)**: 
  - $$\sum_{i=1}^k c_i v_i = 0 \iff c_1 = c_2 = \dots = c_k = 0$$
  - 어떤 벡터도 다른 벡터들의 선형 결합으로 표현할 수 없는 상태.

---

### 2️⃣ [2단계 존재 이유]
- **데이터 중복 제거**: 백터 간 종속(Dependence) 관계가 있으면 정보가 중복되어 행렬 랭크가 떨어지고 역행렬이 파탄 남.

---

### 3️⃣ [3단계 상황별 직관 & 맹점]
- **선형 종속 (Linear Dependence)**: 한 벡터가 다른 벡터들의 평면/직선 상에 얹혀 있어 새로운 차원을 제공하지 못함.

---

### 4️⃣ [4단계 실전 AI 연결고리]
- **다중공선성 (Multicollinearity)**: AI 특징(Feature) 간 선형 종속 시 모델의 가중치 추정이 불가능해지므로 L2 규제(Ridge)를 적용함.
