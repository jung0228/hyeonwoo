# 📐 2.5 & 2.6 Linear Independence, Basis and Rank (선형 독립, 기저, 계수)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.5 & 2.6 완전 해부


## 1. ⚔️ Section 2.5: Linear Independence & Span
- 선형 결합 (Linear Combination): $v = \sum c_i v_i$
- Span: 벡터들의 선형 결합으로 생성 가능한 전체 공간 $\text{span}(v_1, \dots, v_k)$.
- 선형 독립 (Linear Independence): $\sum c_i v_i = 0 \iff c_1 = \dots = c_k = 0$ (어떤 벡터도 다른 벡터들의 조합으로 표현 불가).


## 2. ⚔️ Section 2.6: Basis and Rank (기저와 계수)
- 기저 (Basis): 공간 $V$를 생성(Span)하면서 동시에 선형 독립인 최소 벡터 집합.
- Rank (계수): RREF 변환 후 피벗의 개수 = 행렬의 독립 행/열의 개수 = 데이터의 유효 정보 차원(Effective Dimensionality).
