# 📐 2.6 Basis and Rank (기저와 계수)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.6 완전 해부

---

## 1. ⚔️ 4단계 개념 구조화

### 1️⃣ [1단계 명확한 개념 정의]
- 기저 (Basis): 공간 $V$를 생성(Span)하면서 동시에 선형 독립인 최소 벡터 집합.
- Rank (계수): RREF 변환 후 피벗의 개수 = 행렬의 독립 행/열의 개수 = 데이터의 유효 정보 차원(Effective Dimensionality).

---

### 2️⃣ [2단계 존재 이유]
- 유일 좌표 표현: 기저가 주어지면 공간 내의 모든 벡터는 오직 단 하나의 계수 조합으로 유일하게 표현됨.

---

### 3️⃣ [3단계 상황별 직관 & 맹점]
- Row Rank = Column Rank: 행렬의 독립된 행의 개수와 열의 개수는 무조건 정확히 같다.

---

### 4️⃣ [4단계 실전 AI 연결고리]
- 차원 축소 (Dimensionality Reduction): 데이터의 진짜 Rank만큼 핵심 축(Basis)을 추출하는 PCA / SVD의 이론적 토대.
