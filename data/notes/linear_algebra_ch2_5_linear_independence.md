# 📐 2.5 Linear Independence (선형독립과 생성)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.5 원문 완전 대조 스토리텔링 노트

---

## 1. 🌐 서론: 왜 "선형결합"과 "선형독립"을 다루는가?

2.4절에서 우리는 벡터들이 살고 있는 무대인 벡터 공간(Vector Space)과 부분공간(Subspace)을 정의했습니다.
이제 수학자들의 관심사는 "이 공간 안에 존재하는 무수히 많은 벡터들을 가장 효율적이고 낭비 없이 만들어내는 최소한의 대표 벡터 모음은 무엇인가?" 로 옮겨갑니다.

벡터들을 더하고 스칼라배를 곱해서 새로운 벡터를 만드는 과정을 선형결합(Linear Combination)이라 부르며, 이 벡터들이 서로 중복된 정보 없이 순수하게 새로운 방향을 가리키고 있는지를 판별하는 개념이 바로 선형독립(Linear Independence)입니다.

---

## 2. ⚔️ Section 2.5.1: Linear Combination & (In)dependence (선형결합과 선형독립)

### 📌 1. 선형결합 (Linear Combination: Definition 2.11 & Eq 2.65)
벡터 공간 $V$ 의 유한개의 벡터 $\mathbf{x}_1, \dots, \mathbf{x}_k \in V$ 와 실수 스칼라 $\lambda_1, \dots, \lambda_k \in \mathbb{R}$ 에 대해 다음과 같이 표현되는 모든 벡터 $\mathbf{v} \in V$ 를 선형결합이라 부릅니다:

$$\mathbf{v} = \lambda_1 \mathbf{x}_1 + \dots + \lambda_k \mathbf{x}_k = \sum_{i=1}^k \lambda_i \mathbf{x}_i \in V \quad (2.65)$$

- 영벡터의 자명한 선형결합: 영벡터 $\mathbf{0}$ 은 모든 계수를 0으로 둔 자명한 결합 $\mathbf{0} = \sum 0 \mathbf{x}_i$ 로 언제나 표현 가능합니다.
- 핵심 질문: 계수 $\lambda_i$ 중 0이 아닌 스칼라가 적어도 하나 이상 존재하는 비자명한(non-trivial) 조합으로 $\mathbf{0}$을 만들 수 있는가?

---

### 📌 2. 선형독립과 선형종속 (Linear Independence: Definition 2.12)
벡터 집합 $\{\mathbf{x}_1, \dots, \mathbf{x}_k\} \subseteq V$ 에 대해 선형방정식:

$$\sum_{i=1}^k \lambda_i \mathbf{x}_i = \mathbf{0} \quad (\lambda_1 \mathbf{x}_1 + \dots + \lambda_k \mathbf{x}_k = \mathbf{0})$$

을 만족하는 계수 $\lambda_1, \dots, \lambda_k$ 가 오직 $\lambda_1 = \dots = \lambda_k = 0$ (자명한 해) 일 때만 성립하면, 이 벡터들은 선형독립(Linearly Independent)이라 부릅니다.

반면, 0이 아닌 계수 $\lambda_i \neq 0$ 가 하나라도 존재하여 $\mathbf{0}$을 만들 수 있다면, 이 벡터들은 선형종속(Linearly Dependent)이라 부릅니다.

---

### 📌 3. 지리학적 직관으로 이해하는 선형독립 (Example 2.13 & Figure 2.7)
우리가 나이로비(Nairobi)에서 키갈리(Kigali)로 가는 위치 벡터를 설명할 때:
- 안내 1: "북서쪽으로 506km 이동 후, 남서쪽으로 374km 이동하세요."
  - 북서쪽 벡터(파란색)와 남서쪽 벡터(보라색)는 서로가 서로를 표현할 수 없는 완전히 독립된 차원 2개(선형독립)입니다.
- 안내 2: 여기에 추가로 "서쪽으로 751km 이동한 셈입니다." 라고 서쪽 벡터(검은색)를 덧붙입니다.
  - 서쪽 벡터는 앞선 두 벡터의 합(선형결합)으로 이미 완벽히 표현되는 군더더기 중복 정보입니다!
  - 이 3번째 벡터가 추가되는 순간 3개의 벡터 모음은 선형종속(Linearly Dependent)이 됩니다.

---

### 📌 4. 선형독립성 판별 4대 주요 성질 (Remark & Gaussian Elimination)

1. $k$개의 벡터는 선형독립 아니면 선형종속 둘 중 하나만 존재합니다 (제3의 선택지는 없음).
2. 벡터들 중 영벡터 $\mathbf{0}$이 하나라도 껴있거나, 동일한 벡터가 2개 이상 포함되면 무조건 선형종속입니다.
3. 2개 이상의 벡터가 선형종속일 필요충분조건은 적어도 하나의 벡터가 다른 벡터들의 선형결합(배수)으로 표현 가능하다는 점입니다.
4. 가우스 소거법을 통한 실전 선형독립 판별법:
   - 벡터 $\mathbf{x}_1, \dots, \mathbf{x}_k$ 를 행렬 $A$ 의 열벡터로 나열하고 가우스 소거법을 수행하여 행 사다리꼴(REF)을 만듭니다 (Eq 2.66).
   - 모든 열이 피벗 열(Pivot Column)이면 ➡️ 선형독립(Linearly Independent)!
   - 비피벗 열(Non-pivot Column)이 1개라도 존재하면 ➡️ 선형종속(Linearly Dependent)!

---

### 📌 5. MML 원문 실전 예제 (Example 2.14 & 2.15 백지 대조)

#### 🎯 Example 2.14 ($\mathbb{R}^4$ 공간 3개 벡터 선형독립 검증: Eq 2.67 ~ 2.69)
$$\mathbf{x}_1 = \begin{bmatrix} 1 \\ 2 \\ -3 \\ 4 \end{bmatrix}, \quad \mathbf{x}_2 = \begin{bmatrix} 1 \\ 1 \\ 0 \\ 2 \end{bmatrix}, \quad \mathbf{x}_3 = \begin{bmatrix} -1 \\ -2 \\ 1 \\ 1 \end{bmatrix}$$

열행렬 $A = [\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3]$ 에 기본 행 연산을 적용:

$$\begin{bmatrix} 1 & 1 & -1 \\ 2 & 1 & -2 \\ -3 & 0 & 1 \\ 4 & 2 & 1 \end{bmatrix} \xrightarrow{\text{REF}} \begin{bmatrix} \mathbf{1} & 1 & -1 \\ 0 & \mathbf{1} & 0 \\ 0 & 0 & \mathbf{1} \\ 0 & 0 & 0 \end{bmatrix} \quad (2.69)$$

3개 열 모두 피벗을 가지고 있으므로 비자명해 가 존재하지 않아 $\lambda_1 = \lambda_2 = \lambda_3 = 0$ 이며, 세 벡터는 완벽하게 선형독립입니다!

---

#### 🎯 Example 2.15 (기저 변환 벡터 4개의 선형종속 증명: Eq 2.70 ~ 2.76)
독립기저 $\mathbf{b}_1, \dots, \mathbf{b}_4$ 에 의해 생성된 4개 벡터:
$$\begin{aligned}
\mathbf{x}_1 &= \mathbf{b}_1 - 2\mathbf{b}_2 + \mathbf{b}_3 - \mathbf{b}_4 \\
\mathbf{x}_2 &= -4\mathbf{b}_1 - 2\mathbf{b}_2 + 4\mathbf{b}_4 \\
\mathbf{x}_3 &= 2\mathbf{b}_1 + 3\mathbf{b}_2 - \mathbf{b}_3 - 3\mathbf{b}_4 \\
\mathbf{x}_4 &= 17\mathbf{b}_1 - 10\mathbf{b}_2 + 11\mathbf{b}_3 + \mathbf{b}_4
\end{aligned}$$

계수 행렬 $A$ 의 RREF 구하기 (Eq 2.75 & 2.76):

$$A = \begin{bmatrix} 1 & -4 & 2 & 17 \\ -2 & -2 & 3 & -10 \\ 1 & 0 & -1 & 11 \\ -1 & 4 & -3 & 1 \end{bmatrix} \xrightarrow{\text{RREF}} \begin{bmatrix} \mathbf{1} & 0 & 0 & -7 \\ 0 & \mathbf{1} & 0 & -15 \\ 0 & 0 & \mathbf{1} & -18 \\ 0 & 0 & 0 & 0 \end{bmatrix}$$

- 결과: 4번째 열이 비피벗 열입니다! ($\mathbf{x}_4 = -7\mathbf{x}_1 - 15\mathbf{x}_2 - 18\mathbf{x}_3$)
- 4번째 벡터가 앞선 3개 벡터의 선형결합으로 표현되므로 네 벡터는 선형종속(Linearly Dependent)입니다!

- 핵심 정리 (Remark): $k$개의 독립 벡터로 표현되는 공간에서 $m > k$ 이면, 즉 벡터 개수가 차원보다 많으면 무조건 선형종속이 됩니다.

---

## 3. ⚔️ Section 2.5.2: Generating Set and Span (생성집합과 스팬)

### 📌 1. 생성집합(Generating Set)과 스팬(Span)의 정의 (Definition 2.13)
벡터 공간 $V$ 와 부분집합 $A = \{\mathbf{x}_1, \dots, \mathbf{x}_k\} \subseteq V$ 에 대해:
- Span (스팬): 벡터 $A$ 의 모든 가능한 선형결합들의 집합을 $A$ 의 스팬이라 부르며 $\text{span}[A]$ 또는 $\text{span}[\mathbf{x}_1, \dots, \mathbf{x}_k]$ 로 표기합니다.
- Generating Set (생성집합): 벡터 공간 $V$ 안의 모든 벡터가 $A$ 의 선형결합으로 빠짐없이 만들어질 때 ($V = \text{span}[A]$), $A$ 를 $V$ 의 생성집합이라 부릅니다.

---

### 📌 2. 기저(Basis)로의 최종 연결 (Definition 2.14)
생성집합 $A$ 는 공간 $V$ 전체를 덮을 수 있지만, 중복된 군더더기 벡터가 들어있을 수 있습니다.

- Minimal Generating Set (최소 생성집합): 공간 $V$ 를 생성하는 생성집합 중에서 더 이상 원소를 뺄 수 없는 가장 작은 크기의 집합을 의미합니다.
- Basis (기저): 선형독립인 생성집합을 공간 $V$ 의 기저(Basis)라 부릅니다!

---

### 4단계 실전 AI / 머신러닝 연결고리
- 특징 공간의 차원 축소 (PCA & Feature Independence):
  - 딥러닝 입력 데이터의 컬럼(Feature)들이 선형종속이라는 것은 정보의 중복(Multicollinearity)이 발생함을 의미합니다.
  - 가우스 소거법으로 비피벗 열을 제거하듯, PCA(주성분 분석)나 Autoencoder는 선형독립인 기저 벡터 축만 남겨 핵심 정보의 차원을 축소합니다.
