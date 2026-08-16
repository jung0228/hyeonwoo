# 📐 2.5 Linear Independence (선형독립과 생성)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.5 심층 분석 노트

## 🌐 0. 지난 노트(2.4절)와의 연결 및 빌드업: 왜 "선형독립"이 필요한가?

우리는 지난 2.4절 (Vector Spaces & Subspaces)에서 벡터들이 안전하게 살 수 있는 집이자 무대인 '벡터 공간(Vector Space)'을 정의했습니다. 
벡터 공간이란 덧셈과 스칼라 곱에 대해 닫혀있는 거대한 공간이었습니다.

하지만 2.4절까지 만든 벡터 공간에는 한 가지 커다란 문제가 있습니다.
공간 안에는 무수히 많은 무한개의 벡터들이 빽빽하게 퍼져 있기 때문에, "대체 이 넓은 공간 전체를 덮으려면(Span) 어떤 핵심 벡터들만 뽑아놓아야 하는가?" 에 대한 기준이 없다는 점입니다.

예를 들어, 3차원 공간 $\mathbb{R}^3$ 의 어떤 방향을 설명하기 위해 100개의 벡터를 가져왔다고 해봅시다. 
그 100개 중 대부분은 다른 벡터들을 더하고 숫자를 곱해서 똑같이 만들어낼 수 있는 '군더더기(중복 정보)'일 것입니다. 

수학자들은 생각했습니다:
1. "다른 원소들을 가지고 흉내 내거나 만들어낼 수 없는 진짜 '독립된 순수 방향'들만 남길 수는 없을까?"
2. "공간 전체를 지탱하는 최소한의 뼈대 모음(기저 Basis)을 어떻게 골라낼 것인가?"

이 질문에 답하기 위해 등장하는 개념이 바로 선형결합(Linear Combination)과 선형독립(Linear Independence)입니다! 

- 선형결합: 기존 벡터들을 더하고 숫자를 곱해 새로운 벡터를 만들어내는 기본 조작법
- 선형독립: 어떤 벡터도 다른 벡터들의 선형결합으로 만들어지지 않는, '중복 0%의 순수한 새 방향'들만 모여있는 상태

## 🧠 1. [1단계 개념 정의] 선형결합과 선형독립이란 무엇인가?

### 📌 1. 선형결합 (Linear Combination: Definition 2.11 & Eq 2.65)
벡터 집합 $\mathbf{v}_1, \dots, \mathbf{v}_k$ 와 스칼라 $c_1, \dots, c_k$ 에 대해 다음과 같이 표현되는 벡터:

$$\mathbf{v} = \sum_{i=1}^k c_i \mathbf{v}_i = c_1 \mathbf{v}_1 + \dots + c_k \mathbf{v}_k \quad (2.65)$$

- Span (생성: Definition 2.13): 벡터들의 선형결합으로 생성 가능한 전체 부분공간 $\text{span}(\mathbf{v}_1, \dots, \mathbf{v}_k)$.

### 📌 2. 선형독립과 선형종속 (Linear Independence: Definition 2.12)
$$\sum_{i=1}^k c_i \mathbf{v}_i = \mathbf{0} \iff c_1 = c_2 = \dots = c_k = 0$$

어떤 벡터도 다른 벡터들의 선형결합으로 표현할 수 없는 상태를 선형독립(Linearly Independent)이라 부르며, 0이 아닌 계수로 $\mathbf{0}$을 만들 수 있으면 선형종속(Linearly Dependent)이라 부릅니다.

## 💡 2. [2단계 존재 이유] 데이터 중복 제거 및 역행렬 파탄 방지
백터 간 종속(Dependence) 관계가 있으면 정보가 중복되어 행렬 랭크가 떨어지고 역행렬이 파탄 납니다.

## ⚖️ 3. [3단계 상황별 직관 & 맹점] 지리학적 예시 및 행렬 랭크 판별

### 📌 1. 선형종속의 직관적 기하학 (Example 2.13 & Figure 2.7)
한 벡터가 다른 벡터들의 평면/직선 상에 얹혀 있어 새로운 차원을 제공하지 못하는 상태입니다.
- 나이로비 ➡️ 키갈리 이동 시 "북서쪽 506km"와 "남서쪽 374km" 2개 벡터면 충분하지만, 여기에 "서쪽 751km" 벡터를 추가하면 앞선 두 벡터의 합으로 표현되는 중복 정보(선형종속)가 됩니다.

### 📌 2. MML 교재 원문 예시 해부 (Examples 2.14, 2.15 & Rank 판별)

#### 🎯 Example 2.14 (3개 2차원/4차원 벡터의 선형 독립성 판별)
$$\mathbf{x}_1 = \begin{bmatrix} 1 \\ 2 \\ -3 \\ 4 \end{bmatrix}, \quad \mathbf{x}_2 = \begin{bmatrix} 1 \\ 1 \\ 0 \\ 2 \end{bmatrix}, \quad \mathbf{x}_3 = \begin{bmatrix} -1 \\ -2 \\ 1 \\ 1 \end{bmatrix}$$

계수 행렬 $[\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3]$ 소거 결과 모든 열이 피벗 열(Pivot Column)이므로 유일해 $c_1=c_2=c_3=0$ 만 존재하여 완벽한 선형 독립(Linearly Independent)입니다!

#### 🎯 Example 2.15 (4개 벡터의 선형 종속성 판별: Eq 2.73~2.76)
$$\begin{aligned}
\mathbf{x}_1 &= \mathbf{b}_1 - 2\mathbf{b}_2 + \mathbf{b}_3 - \mathbf{b}_4 \\\\
\mathbf{x}_2 &= -4\mathbf{b}_1 - 2\mathbf{b}_2 + 4\mathbf{b}_4 \\\\
\mathbf{x}_3 &= 2\mathbf{b}_1 + 3\mathbf{b}_2 - \mathbf{b}_3 - 3\mathbf{b}_4 \\\\
\mathbf{x}_4 &= 17\mathbf{b}_1 - 10\mathbf{b}_2 + 11\mathbf{b}_3 + \mathbf{b}_4
\end{aligned}$$

계수 행렬 $A$ 소거 시 4번째 열이 비피벗 열이 되어 $\mathbf{x}_4 = -7\mathbf{x}_1 - 15\mathbf{x}_2 - 18\mathbf{x}_3$ 선형 결합 표현 가능 ➡️ 선형 종속(Linearly Dependent)!

#### 📌 행렬 랭크를 통한 선형독립 자동 판별 규칙
행렬 $V = [\mathbf{v}_1, \dots, \mathbf{v}_k]$ 의 피벗 개수(Rank)를 이용한 판별:
- $\text{Rank}(V) = k \implies$ 선형 독립 (Full Column Rank)
- $\text{Rank}(V) < k \implies$ 선형 종속 (Rank Deficient)

## 🚀 4. [4단계 실전 AI 연결고리] 다중공선성 (Multicollinearity) & Ridge 규제
AI 특징(Feature) 간 선형 종속 시 모델의 가중치 추정이 불가능해지므로 $X^\top X$ 가 비가역(Singular)이 됩니다. 이를 막기 위해 L2 규제(Ridge: $X^\top X + \lambda I$)를 적용하여 강제 독립성을 확보합니다.
