# 📐 2.6 Basis and Rank (기저와 계수)

> **POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.6 원문 완전 복사 & 심층 가공 노트**

---

## 📖 Part 1. MML 교재 원문 (Textbook Original Text)

### 1. Definition 2.13 (Generating Set and Span) & Definition 2.14 (Basis)
```text
Definition 2.13 (Generating Set and Span).
Consider a vector space V = (V, +, \cdot) and set of vectors A = {x1, ..., xk} in V. If every vector v in V can be expressed as a linear combination of x1, ..., xk, A is called a generating set of V. The set of all linear combinations of vectors in A is called the span of A. If A spans the vector space V, we write V = span[A] or V = span[x1, ..., xk].

Definition 2.14 (Basis).
Consider a vector space V = (V, +, \cdot) and A in V. A generating set A of V is called minimal if there exists no smaller set A_tilde \subsetneq A in V that spans V. Every linearly independent generating set of V is minimal and is called a basis of V.
```

### 2. Theorem (Equivalent Statements for Basis) & Example 2.16
```text
Let V = (V, +, \cdot) be a vector space and B in V, B != \emptyset. Then, the following statements are equivalent:
1. B is a basis of V.
2. B is a minimal generating set.
3. B is a maximal linearly independent set of vectors in V, i.e., adding any other vector to this set will make it linearly dependent.
4. Every vector x in V is a linear combination of vectors from B, and every linear combination is unique, i.e., with
x = \sum_{i=1}^k \lambda_i b_i = \sum_{i=1}^k \psi_i b_i  (Eq 2.77)
and \lambda_i, \psi_i in R, b_i in B it follows that \lambda_i = \psi_i, i = 1, ..., k.

Example 2.16 (Standard & Non-standard Bases):
In R^3, canonical basis B = {[1,0,0]^T, [0,1,0]^T, [0,0,1]^T}.
Different bases B1 = {[1,0,0]^T, [1,1,0]^T, [1,1,1]^T}.
Set A = {[1,2,3,4]^T, [2,-1,0,2]^T, [1,1,0,-4]^T} is linearly independent, but not a generating set (and no basis) of R^4.
```

### 3. Remark & Example 2.17 (Determining a Basis via Pivot Columns)
```text
Remark (Finding a Basis of U = span[x1, ..., xm] in R^n):
1. Write the spanning vectors as columns of a matrix A.
2. Determine the row-echelon form of A.
3. The spanning vectors associated with the pivot columns are a basis of U.

Example 2.17:
Spanning vectors x1, x2, x3, x4 in R^5. Matrix A = [x1, x2, x3, x4]:
[  1  2  3 -1 ]         [ 1  2  3 -1 ]
[  2 -1 -4  8 ]  --> ... --> [ 0  1  2 -2 ]
[ -1  1  3 -5 ]         [ 0  0  0  1 ]
[ -1  2  5 -6 ]         [ 0  0  0  0 ]
[ -1 -2 -3  1 ]         [ 0  0  0  0 ]
Pivot columns are 1, 2, 4. Therefore, {x1, x2, x4} is a basis of U.
```

### 4. Section 2.6.2 Rank (Definition & Properties) & Example 2.18
```text
Definition (Rank).
The number of linearly independent columns of a matrix A in R^{m \times n} equals the number of linearly independent rows and is called the rank of A and is denoted by rk(A).

Remark (Properties of Rank):
- rk(A) = rk(A^T), i.e., column rank equals row rank.
- The columns of A span a subspace U in R^m with dim(U) = rk(A).
- A in R^{n \times n} is regular (invertible) if and only if rk(A) = n.
- Full rank matrix: rk(A) = min(m, n). Rank deficient if rk(A) < min(m, n).

Example 2.18:
A = [1 2 1; -2 -3 1; 3 5 0] --> REF [1 2 1; 0 1 3; 0 0 0] (Eq 2.84).
Two pivot rows/columns, so rk(A) = 2.
```

---

## 🧠 Part 2. 한국어 정밀 가공 & 개념 설명 (Deep Interpretation)

### 📌 1. [개념 정의] 기저(Basis)와 계수(Rank)란 무엇인가?
- **기저 (Basis)**: 어떤 공간 전체를 덮으면서(Span) 중복된 원소가 단 하나도 없는 **"최소 뼈대 벡터 모음"**입니다.
- **계수 (Rank)**: 행렬이 가진 **실질적인 독립 정보의 차원 수**를 의미합니다.

### 📌 2. [존재 이유 & 직관] 왜 최소 생성집합이자 최대 독립집합인가?
- 기저(Basis)는 두 가지 팽창과 축소의 조화점입니다:
  - 공간 전체를 표현하려 벡터를 자꾸 추가하다 보면(Generating Set) 팽창하지만,
  - 군더더기를 싹 제거해서 가장 작게 다이어트시킨 상태(Minimal)가 됩니다.
  - 동시에 독립성을 유지하며 가장 많이 모을 수 있는 최대 개수(Maximal Linearly Independent)가 됩니다.

### 📌 3. [상황별 Trade-off & 맹점] Row Rank = Column Rank 의 경이로움
- 행렬의 세로 길이(행)와 가로 길이(열)가 완전히 달라도, **독립된 행의 개수와 독립된 열의 개수는 무조건 100% 일치**합니다 ($\text{rk}(A) = \text{rk}(A^\top)$).
- **Rank Deficient (계수 결손)**: 만약 100차원 데이터 행렬의 Rank가 5라면, 불필요한 노이즈와 중복 정보가 95개나 섞여 있음을 뜻합니다.

### 📌 4. [실전 AI 연결고리]
- **SVD(이상값 분해) & Low-Rank Approximation (LoRA)**:
  - LLM 초거대 모델 파인튜닝 시 전체 가중치 업데이트 행렬 $W$ 대신 Low-rank 행렬 $A \times B$ ($\text{rank} \ll d$) 로 분해하여 파라미터 메모리를 99% 절약하는 기술의 핵심 근거가 됩니다.
