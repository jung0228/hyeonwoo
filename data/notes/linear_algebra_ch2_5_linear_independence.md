# 📐 2.5 Linear Independence (선형독립과 생성)

> **POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.5 원문 완전 복사 & 심층 가공 노트**

---

## 📖 Part 1. MML 교재 원문 (Textbook Original Text)

### 1. Intro & Definition 2.11 (Linear Combination)
```text
In the following, we will have a close look at what we can do with vectors (elements of the vector space). In particular, we can add vectors together and multiply them with scalars. The closure property guarantees that we end up with another vector in the same vector space.

Definition 2.11 (Linear Combination).
Consider a vector space V and a finite number of vectors x1, ..., xk in V. Then, every v in V of the form
v = \lambda_1 x_1 + ... + \lambda_k x_k = \sum_{i=1}^k \lambda_i x_i  (Eq 2.65)
with \lambda_1, ..., \lambda_k in R is a linear combination of the vectors x1, ..., xk.
```

### 2. Definition 2.12 (Linear (In)dependence) & Example 2.13
```text
Definition 2.12 (Linear (In)dependence).
Let us consider a vector space V with k in N and x1, ..., xk in V. If \sum_{i=1}^k \lambda_i x_i = 0 possesses only the trivial solution \lambda_1 = 0, ..., \lambda_k = 0, then the vectors x1, ..., xk are linearly independent. If there is at least one non-trivial solution, the vectors are linearly dependent.

Example 2.13 (Geographic Example)
A person in Nairobi (Kenya) describing where Kigali (Rwanda) is might say: "You can get to Kigali by first going 506 km Northwest to Kampala (Uganda) and then 374 km Southwest." This is sufficient information. The person may add: "It is about 751 km West of here."
In this example, "506 km Northwest" and "374 km Southwest" are linearly independent. However, the third "751 km West" vector is a linear combination of the other two, making the set linearly dependent.
```

### 3. Remark & Gaussian Elimination for Independence
```text
Remark (Properties of Linear Independence):
- k vectors are either linearly dependent or linearly independent.
- If at least one vector is 0 or two vectors are identical, they are linearly dependent.
- Practical checking via Gaussian elimination: Write vectors as columns of matrix A and perform Gaussian elimination to Row Echelon Form (REF).
  * Pivot columns: Linearly independent of vectors on the left.
  * Non-pivot columns: Can be expressed as linear combinations of pivot columns on their left.
  * All columns are linearly independent if and only if all columns are pivot columns.
```

### 4. Example 2.14 & Example 2.15 (Original Textbook Calculations)
```text
Example 2.14:
x1 = [1, 2, -3, 4]^T,  x2 = [1, 1, 0, 2]^T,  x3 = [-1, -2, 1, 1]^T in R^4.
Solving \lambda_1 x_1 + \lambda_2 x_2 + \lambda_3 x_3 = 0 leads to matrix:
[ 1  1 -1 ]         [ 1  1 -1 ]
[ 2  1 -2 ]  --> ... --> [ 0  1  0 ] (Eq 2.69)
[-3  0  1 ]         [ 0  0  1 ]
[ 4  2  1 ]         [ 0  0  0 ]
Every column is a pivot column. Thus \lambda_1 = \lambda_2 = \lambda_3 = 0 is the unique solution. Linearly independent.

Example 2.15:
Given linearly independent vectors b1, b2, b3, b4 in R^n and combinations x1, x2, x3, x4.
Coefficient matrix A:
[  1 -4  2  17 ]         [ 1  0  0  -7 ]
[ -2 -2  3 -10 ]  --> ... --> [ 0  1  0 -15 ] (Eq 2.76)
[  1  0 -1  11 ]         [ 0  0  1 -18 ]
[ -1  4 -3   1 ]         [ 0  0  0   0 ]
The 4th column is a non-pivot column: x4 = -7 x1 - 15 x2 - 18 x3. Linearly dependent.
```

---

## 🧠 Part 2. 한국어 정밀 가공 & 개념 설명 (Deep Interpretation)

### 📌 1. [개념 정의] 선형결합이란 무엇이며 왜 선형독립을 다루는가?
- **선형결합 (Linear Combination)**: 기존 벡터들에 숫자를 곱하고(스칼라배) 더해서 새로운 벡터를 만드는 가장 기본적인 수학적 합성 조작입니다.
- **선형독립 (Linear Independence)**: "서로 다른 벡터 모음 중에 완전히 똑같은 방향을 가리키거나 다른 원소들의 조합으로 만들어지는 '군더더기(중복 정보)'가 단 하나도 없는 상태"를 의미합니다.

### 📌 2. [존재 이유 & 직관] 왜 가우스 소거법으로 선형독립을 검증하는가?
- **피벗 열(Pivot Column)의 본질**: 행렬의 열벡터들을 가우스 소거법으로 바꿨을 때, **피벗이 있는 열**은 왼쪽에 있는 벡터들로 결코 만들어낼 수 없는 **새로운 독립적인 차원 방향**을 뜻합니다.
- **비피벗 열(Non-pivot Column)의 본질**: 피벗이 생기지 않는 열은 앞선 피벗 열들의 선형결합으로 100% 표현되는 **중복 정보(종속 원소)**를 의미합니다.

### 📌 3. [상황별 Trade-off & 맹점]
- **선형종속일 때 일어나는 파탄**: 데이터의 피처(Feature) 간에 선형종속이 발생하면 행렬의 Rank가 떨어지고, 역행렬이 존재하지 않게 되어(Singular Matrix) 선형 회귀 등 최적화 방정식의 유일해 도출이 파탄 납니다.

### 📌 4. [실전 AI 연결고리]
- **PCA(주성분 분석) & 다중공선성(Multicollinearity) 해결**: 딥러닝 입력 데이터에서 선형종속 관계에 있는 중복 피처를 가우스 소거법/SVD로 찾아내어 선형독립인 주요 성분 기저축만 남기는 방식으로 차원 축소를 수행합니다.
