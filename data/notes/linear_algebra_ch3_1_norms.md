# 📐 3.1 Norms (노름과 길이기하학)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Chapter 3.1 전수 분석 & 4단계 정밀 해설 노트

## 🌐 0. Chapter 2와 Chapter 3의 연결: 왜 "분석 기하학(Analytic Geometry)"으로 나아가는가?

우리는 Chapter 2 (Linear Algebra)에서 벡터, 벡터 공간, 기저, 선형사상 등 벡터들의 대수적(Algebraic) 구조를 다루었습니다. 
그러나 Chapter 2까지의 개념에는 아직 "길이(Length)", "거리(Distance)", "각도(Angle)"라는 기하학적 개념이 존재하지 않았습니다.

Chapter 3 (Analytic Geometry)에서는 벡터 공간에 내적(Inner Product)과 노름(Norm)을 부여함으로써, 벡터에 구체적인 기하학적 직관(Geometric Intuition)을 완성합니다!

```text
[Chapter 2: 추상 대수 구조]       [Chapter 3: 분석 기하학]             [후속 AI 모델 연결]
- 벡터 공간 (Vector Space)  ───>  - 노름 (Norm: 길이/거리)    ───>  - Ch 12: Support Vector Machine (Margin 최적화)
- 선형사상 (Linear Mapping) ───>  - 내적 (Inner Product: 각도)───>  - Ch 10: PCA (주성분 직교 정사영)
- 기저 (Basis)              ───>  - 직교 정사영 (Projection)  ───>  - Ch 9: Linear Regression (최소제곱 회귀)
```

## 1. ⚔️ Section 3.1: Norms (노름의 엄밀한 수학적 정의)

### 📌 1. 노름(Norm)의 3대 공리 (Definition 3.1 & Eq 3.1~3.2)
벡터 공간 $V$ 위의 노름(Norm)이란 모든 벡터 $\mathbf{x} \in V$ 에 대해 그 길이 $\Vert\mathbf{x}\Vert \in \mathbb{R}$ 를 할당하는 함수 $\Vert\cdot\Vert : V \to \mathbb{R}$ 이며, 모든 스칼라 $\lambda \in \mathbb{R}$ 및 벡터 $\mathbf{x}, \mathbf{y} \in V$ 에 대해 다음 3가지 공리를 반드시 만족해야 합니다:

1. 절대 동차성 (Absolutely Homogeneous):
   $$\Vert\lambda \mathbf{x}\Vert = |\lambda| \Vert\mathbf{x}\Vert$$
   - *직관*: 벡터의 길이를 $\lambda$배 스케일링하면, 그 노름(길이)도 정확히 $|\lambda|$ 절대값배만큼 비율대로 늘어나거나 줄어듭니다.
2. 삼각 부등식 (Triangle Inequality: Figure 3.2):
   $$\Vert\mathbf{x} + \mathbf{y}\Vert \le \Vert\mathbf{x}\Vert + \Vert\mathbf{y}\Vert$$
   - *직관*: 삼각형에서 두 변의 길이의 합은 나머지 한 변의 길이보다 무조건 크거나 같습니다. (돌아가는 길이 직선거리보다 짧을 수 없다).
3. 양의 정정성 (Positive Definite):
   $$\Vert\mathbf{x}\Vert \ge 0 \quad \text{and} \quad \Vert\mathbf{x}\Vert = 0 \iff \mathbf{x} = \mathbf{0}$$
   - *직관*: 모든 벡터의 길이는 0 이상이며, 길이가 0인 벡터는 오직 영벡터 $\mathbf{0}$ 뿐입니다.

## 2. ⚔️ 대표적인 노름 예시와 단위 원(Unit Circle) 기하학

### 📌 1. 맨해튼 노름 (Manhattan Norm / $\ell_1$ Norm: Example 3.1 & Eq 3.3)
벡터 $\mathbf{x} \in \mathbb{R}^n$ 의 각 성분의 절대값을 모두 더한 노름입니다:

$$\Vert\mathbf{x}\Vert_1 := \sum_{i=1}^n |x_i| \quad (\text{Eq 3.3})$$

- 기하학적 형태 (Figure 3.3 좌측): $\mathbb{R}^2$ 에서 $\Vert\mathbf{x}\Vert_1 = 1$ 인 단위 원(Unit Circle)을 그리면 원점이 아닌 마름모(Diamond) 형태가 됩니다.
- AI/ML 연결: Lasso 회귀 및 희소성(Sparsity) 유도에 사용됩니다.

### 📌 2. 유클리드 노름 (Euclidean Norm / $\ell_2$ Norm: Example 3.2 & Eq 3.4)
우리가 흔히 말하는 피타고라스 정리에 의한 원점으로부터의 거리를 나타내는 노름입니다:

$$\Vert\mathbf{x}\Vert_2 := \sqrt{\sum_{i=1}^n x_i^2} = \sqrt{\mathbf{x}^\top \mathbf{x}} \quad (\text{Eq 3.4})$$

- 기하학적 형태 (Figure 3.3 우측): $\mathbb{R}^2$ 에서 $\Vert\mathbf{x}\Vert_2 = 1$ 인 단위 원은 우리가 아는 매끄러운 동그라미(Circle) 형태입니다.
- 교재 기본 설정 (Remark): MML 교재 전체에서 별도의 명시가 없으면 기본적으로 유클리드 노름(Euclidean Norm)을 사용합니다.

## 🧠 3. 4단계 정밀 개념 해설

### 1️⃣ [1단계 개념 정의]
- 노름(Norm): 추상적인 벡터 공간 위의 원소(벡터)에 "길이(Length)"라는 실수값을 부여하는 검증 기준 함수입니다.

### 2️⃣ [2단계 왜 쓰는가?]
- 벡터 간의 거리(Distance)와 유사도(Similarity)를 정량적 수치로 계산하기 위해 사용합니다. 노름이 있어야 데이터 간의 가까움과 먼 정도를 측정할 수 있습니다.

### 3️⃣ [3단계 상황별 직관 & Trade-off]
- $\ell_1$ vs $\ell_2$ 단위 원의 모서리(Corner) 차이:
  - $\ell_1$ 노름은 축 상의 점($(1,0), (0,1)$ 등)에 날카로운 꺾임(Corner)이 존재하여, 최적화 시 성분값을 정확히 0으로 만들어 주는 Feature Selection (Sparsity) 효과가 발생합니다.
  - $\ell_2$ 노름은 미분 가능하고 매끄러운 형태를 가져 무난한 구형(Spherical) 제약을 줍니다.

### 4️⃣ [4단계 실전 AI 연결고리]
- Support Vector Machine (SVM - Ch 12): 두 클래스 간의 마진(Margin)을 최대화하는 과정에서 마진의 크기가 $\frac{2}{\Vert\mathbf{w}\Vert_2}$ 로 정의되므로, 유클리드 노름을 최소화하는 QP 문제로 귀결됩니다.
- Ridge ($\ell_2$) & Lasso ($\ell_1$) 규제: 과적합(Overfitting)을 방지하기 위해 가중치 벡터의 노름 $\Vert\mathbf{w}\Vert$ 을 손실 함수에 페널티 항으로 추가합니다.
