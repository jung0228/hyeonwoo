# 📐 2.5 Linear Independence (선형독립과 생성)

> **POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.5 원문 완전 복사 & 심층 가공 노트**


## 📖 Part 1. MML 교재 2.5절 전체 원문 (Textbook Full Original Text)

```text
2.5 Linear Independence
In the following, we will have a close look at what we can do with vectors
(elements of the vector space). In particular, we can add vectors together
and multiply them with scalars. The closure property guarantees that we
end up with another vector in the same vector space. It is possible to find
a set of vectors with which we can represent every vector in the vector
space by adding them together and scaling them. This set of vectors is
a basis, and we will discuss them in Section 2.6.1. Before we get there,
we will need to introduce the concepts of linear combinations and linear
independence.

Definition 2.11 (Linear Combination). Consider a vector space V and a
finite number of vectors x1, . . . , xk \in V . Then, every v \in V of the form
v = \lambda_1 x_1 + \cdot\cdot\cdot + \lambda_k x_k = \sum_{i=1}^k \lambda_i x_i \in V (2.65)
with \lambda_1, . . . , \lambda_k \in R is a linear combination of the vectors x1, . . . , xk.

The 0-vector can always be written as the linear combination of k vectors x1, . . . , xk because 0 = \sum_{i=1}^k 0 x_i is always true. In the following, we are interested in non-trivial linear combinations of a set of vectors to represent 0, i.e., linear combinations of vectors x1, . . . , xk, where not all coefficients \lambda_i in (2.65) are 0.

Definition 2.12 (Linear (In)dependence). Let us consider a vector space V with k \in N and x1, . . . , xk \in V . If there is a non-trivial linear combination, such that 0 = \sum_{i=1}^k \lambda_i x_i with at least one \lambda_i \neq 0, the vectors x1, . . . , xk are linearly dependent. If only the trivial solution exists, i.e., \lambda_1 = . . . = \lambda_k = 0 the vectors x1, . . . , xk are linearly independent.

Linear independence is one of the most important concepts in linear algebra. Intuitively, a set of linearly independent vectors consists of vectors that have no redundancy, i.e., if we remove any of those vectors from the set, we will lose something. Throughout the next sections, we will formalize this intuition more.

Example 2.13 (Linearly Dependent Vectors)
A geographic example may help to clarify the concept of linear independence. A person in Nairobi (Kenya) describing where Kigali (Rwanda) is might say ,“You can get to Kigali by first going 506 km Northwest to Kampala (Uganda) and then 374 km Southwest.”. This is sufficient information to describe the location of Kigali because the geographic coordinate system may be considered a two-dimensional vector space (ignoring altitude and the Earth’s curved surface). The person may add, “It is about 751 km West of here.” Although this last statement is true, it is not necessary to find Kigali given the previous information (see Figure 2.7 for an illustration). In this example, the “506 km Northwest” vector (blue) and the “374 km Southwest” vector (purple) are linearly independent. This means the Southwest vector cannot be described in terms of the Northwest vector, and vice versa. However, the third “751 km West” vector (black) is a linear combination of the other two vectors, and it makes the set of vectors linearly dependent. Equivalently, given “751 km West” and “374 km Southwest” can be linearly combined to obtain “506 km Northwest”.

Remark. The following properties are useful to find out whether vectors are linearly independent:
- k vectors are either linearly dependent or linearly independent. There is no third option.
- If at least one of the vectors x1, . . . , xk is 0 then they are linearly dependent. The same holds if two vectors are identical.
- The vectors {x1, . . . , xk : xi \neq 0, i = 1, . . . , k}, k \ge 2, are linearly dependent if and only if (at least) one of them is a linear combination of the others. In particular, if one vector is a multiple of another vector, i.e., xi = \lambda xj , \lambda \in R then the set {x1, . . . , xk : xi \neq 0, i = 1, . . . , k} is linearly dependent.

A practical way of checking whether vectors x1, . . . , xk \in V are linearly independent is to use Gaussian elimination: Write all vectors as columns of a matrix A and perform Gaussian elimination until the matrix is in row echelon form (the reduced row-echelon form is unnecessary here):
– The pivot columns indicate the vectors, which are linearly independent of the vectors on the left. Note that there is an ordering of vectors when the matrix is built.
– The non-pivot columns can be expressed as linear combinations of the pivot columns on their left. For instance, the row-echelon form
[ 1 3 0 ]
[ 0 0 2 ] (2.66)
tells us that the first and third columns are pivot columns. The second column is a non-pivot column because it is three times the first column.
All column vectors are linearly independent if and only if all columns are pivot columns. If there is at least one non-pivot column, the columns (and, therefore, the corresponding vectors) are linearly dependent.

Example 2.14
Consider R4 with
x1 = [1, 2, -3, 4]^T, x2 = [1, 1, 0, 2]^T, x3 = [-1, -2, 1, 1]^T. (2.67)
To check whether they are linearly dependent, we follow the general approach and solve
\lambda_1 x_1 + \lambda_2 x_2 + \lambda_3 x_3 = 0 (2.68)
for \lambda_1, . . . , \lambda_3. We write the vectors xi, i = 1, 2, 3, as the columns of a matrix and apply elementary row operations until we identify the pivot columns:
[  1  1 -1 ]         [ 1  1 -1 ]
[  2  1 -2 ]  --> ... --> [ 0  1  0 ] (2.69)
[ -3  0  1 ]         [ 0  0  1 ]
[  4  2  1 ]         [ 0  0  0 ]
Here, every column of the matrix is a pivot column. Therefore, there is no non-trivial solution, and we require \lambda_1 = 0, \lambda_2 = 0, \lambda_3 = 0 to solve the equation system. Hence, the vectors x1, x2, x3 are linearly independent.

Remark. Consider a vector space V with k linearly independent vectors b1, . . . , bk and m linear combinations
x1 = \sum_{i=1}^k \lambda_{i1} b_i ,  ... ,  xm = \sum_{i=1}^k \lambda_{im} b_i. (2.70)
Defining B = [b1, . . . , bk] as the matrix whose columns are the linearly independent vectors b1, . . . , bk, we can write
xj = B \lambda_j , \lambda_j = [\lambda_{1j}, ..., \lambda_{kj}]^T , j = 1, . . . , m , (2.71)
in a more compact form.
We want to test whether x1, . . . , xm are linearly independent. For this purpose, we follow the general approach of testing when \sum_{j=1}^m \psi_j x_j = 0. With (2.71), we obtain
\sum_{j=1}^m \psi_j x_j = \sum_{j=1}^m \psi_j B \lambda_j = B \sum_{j=1}^m \psi_j \lambda_j . (2.72)
This means that {x1, . . . , xm} are linearly independent if and only if the column vectors {\lambda_1, . . . , \lambda_m} are linearly independent.

Remark. In a vector space V , m linear combinations of k vectors x1, . . . , xk are linearly dependent if m > k.

Example 2.15
Consider a set of linearly independent vectors b1, b2, b3, b4 \in R^n and
x1 = b1 - 2b2 + b3 - b4
x2 = -4b1 - 2b2 + 4b4
x3 = 2b1 + 3b2 - b3 - 3b4
x4 = 17b1 - 10b2 + 11b3 + b4 . (2.73)
Are the vectors x1, . . . , x4 \in R^n linearly independent? To answer this question, we investigate whether the column vectors
[ 1, -2, 1, -1 ]^T, [ -4, -2, 0, 4 ]^T, [ 2, 3, -1, -3 ]^T, [ 17, -10, 11, 1 ]^T (2.74)
are linearly independent. The reduced row-echelon form of the corresponding linear equation system with coefficient matrix
A =
[  1 -4  2  17 ]
[ -2 -2  3 -10 ] (2.75)
[  1  0 -1  11 ]
[ -1  4 -3   1 ]
is given as
[ 1 0 0  -7 ]
[ 0 1 0 -15 ] (2.76)
[ 0 0 1 -18 ]
[ 0 0 0   0 ] .
We see that the corresponding linear equation system is non-trivially solvable: The last column is not a pivot column, and x4 = -7x1 - 15x2 - 18x3. Therefore, x1, . . . , x4 are linearly dependent as x4 can be expressed as a linear combination of x1, . . . , x3.
```


## 🧠 Part 2. 한국어 정밀 가공 & 개념 설명 (Deep Interpretation)

### 📌 1. [개념 정의] 선형결합이란 무엇이며 왜 선형독립을 다루는가?
- **선형결합 (Linear Combination: Eq 2.65)**: 기존 벡터들에 숫자를 곱하고(스칼라배) 더해서 새로운 벡터를 만드는 가장 기본적인 수학적 합성 조작입니다.
- **선형독립 (Linear Independence: Def 2.12)**: "서로 다른 벡터 모음 중에 완전히 똑같은 방향을 가리키거나 다른 원소들의 조합으로 만들어지는 '군더더기(중복 정보)'가 단 하나도 없는 상태"를 의미합니다.

### 📌 2. [존재 이유 & 직관] 왜 가우스 소거법으로 선형독립을 검증하는가?
- **피벗 열(Pivot Column)의 본질**: 행렬의 열벡터들을 가우스 소거법으로 바꿨을 때, **피벗이 있는 열**은 왼쪽에 있는 벡터들로 결코 만들어낼 수 없는 **새로운 독립적인 차원 방향**을 뜻합니다.
- **비피벗 열(Non-pivot Column)의 본질**: 피벗이 생기지 않는 열은 앞선 피벗 열들의 선형결합으로 100% 표현되는 **중복 정보(종속 원소)**를 의미합니다.

### 📌 3. [상황별 Trade-off & 맹점]
- **선형종속일 때 일어나는 파탄**: 데이터의 피처(Feature) 간에 선형종속이 발생하면 행렬의 Rank가 떨어지고, 역행렬이 존재하지 않게 되어(Singular Matrix) 선형 회귀 등 최적화 방정식의 유일해 도출이 파탄 납니다.

### 📌 4. [실전 AI 연결고리]
- **PCA(주성분 분석) & 다중공선성(Multicollinearity) 해결**: 딥러닝 입력 데이터에서 선형종속 관계에 있는 중복 피처를 가우스 소거법/SVD로 찾아내어 선형독립인 주요 성분 기저축만 남기는 방식으로 차원 축소를 수행합니다.
