# 📐 6.7 & 6.8 Change of Variables / Inverse Transform and Further Reading (변수 변환 기법, 야코비안 행렬식과 노말라이징 플로우)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 6.7 & 6.8 전수 분석 & 4단계 정밀 해설 노트 (Chapter 6 대단원 완성)


## 🌐 0. 공간의 왜곡과 확률의 보존: 왜 "변수 변환 기법(Change of Variables)"인가?

우리가 잘 아는 기본 확률분포(표준정규분포 $\mathcal{N}(\mathbf{0}, I)$, 균등분포 $\mathcal{U}[0, 1]$)에 비선형/선형 변환 함수 $\mathbf{y} = U(\mathbf{x})$ 를 가하면, 변환된 변수 $\mathbf{y}$ 는 완전히 새로운 확률분포를 형성합니다.

- 이산 확률변수의 변환: 단순히 이벤트의 상태값만 역함수 $P(Y=y) = P(X=U^{-1}(y))$ 로 이동합니다.
- 연속 확률변수의 변환: 연속 공간에서는 점 확률이 0이므로, 변환 함수 $U$ 가 공간의 부피(Volume)를 얼마나 확대하거나 축소(왜곡)시키는지 보정해 주어야 합니다.
- 야코비안 행렬식 (Jacobian Determinant $|\det J|$): 공간 변환 시 부피의 팽창/축소 비율을 나타내는 수학적 보정 인자로, 현대 생성 모델인 노말라이징 플로우(Normalizing Flows) 의 수식적 근간이 됩니다.


## 1. ⚔️ Section 6.7.1: Distribution Function Technique (분포함수법과 확률적분변환)


### 📌 1. 분포함수법 2단계 절차 (Eq 6.126~6.127)

연속 확률변수 $X$ 와 가역 변환 $Y = U(X)$ 에 대해 $Y$ 의 확률밀도함수(PDF) $f(y)$ 를 구하는 가장 근본적인 2단계 절차:

1. 누적분포함수 (CDF) 구하기:
   $$F_Y(y) = P(Y \le y) = P(U(X) \le y) = P(X \le U^{-1}(y)) = F_X(U^{-1}(y)) \quad (\text{Eq 6.126})$$
2. CDF 미분으로 PDF 구하기:
   $$f(y) = \frac{d}{dy} F_Y(y) \quad (\text{Eq 6.127})$$

#### 💡 [Example 6.16: 분포함수법 수치 유도]
$f(x) = 3x^2 \; (0 \le x \le 1)$ 일 때 $Y = X^2$ 의 PDF 구하기:
1. CDF: $F_Y(y) = P(X^2 \le y) = P(X \le y^{1/2}) = \int_0^{y^{1/2}} 3t^2 dt = [t^3]_0^{y^{1/2}} = y^{3/2} \quad (\text{Eq 6.129})$
2. 미분 PDF: $f(y) = \frac{d}{dy} (y^{3/2}) = \frac{3}{2} y^{1/2} \; (0 \le y \le 1) \quad (\text{Eq 6.131})$


### 📌 2. 확률적분변환 정리 (Probability Integral Transform: Theorem 6.15 ★ 역변환 샘플링!)

엄격히 단조증가하는 CDF $F_X(x)$ 를 가지는 연속 확률변수 $X$ 에 대해, 변환된 확률변수 $Y := F_X(X)$ 는 무조건 단위 균등분포 $\mathcal{U}[0, 1]$ 를 따릅니다!

$$X \sim p(x) \implies Y = F_X(X) \sim \mathcal{U}[0, 1] \quad (\text{Theorem 6.15})$$

- 역변환 샘플링 (Inverse Transform Sampling):
  위 정리를 거꾸로 이용하면, 컴퓨터에서 균등분포 난수 $u \sim \mathcal{U}[0, 1]$ 를 뽑은 후 역CDF 변환 $x = F_X^{-1}(u)$ 를 가하기만 하면 원하는 임의의 복잡한 분포 $X$ 의 난수를 즉시 생성할 수 있습니다!


## 2. ⚔️ Section 6.7.2: Change of Variables (다변량 변수 변환과 야코비안 행렬식)


### 📌 1. 단변량 변수 변환 공식 (Eq 6.143)

미적분학의 치환적분(Substitution Rule: Eq 6.133)과 미적분학 기본 정리에 의해 유도되는 단변량 변수 변환 공식:

$$f(y) = f_x(U^{-1}(y)) \cdot \left| \frac{d}{dy} U^{-1}(y) \right| \quad (\text{Eq 6.143})$$

- $\left| \frac{d}{dy} U^{-1}(y) \right|$ 는 변환 $U$ 에 의해 1차원 미소 미분 구간(길이)이 얼마나 확대/축소되는지를 보정하는 미분 미율 인자입니다.


### 👑 2. 다변량 변수 변환 정리 (Multivariate Change of Variables: Theorem 6.16 & Eq 6.144 ★ Normalizing Flows 핵심!)

다변량 연속 확률변수 $\mathbf{x} \in \mathbb{R}^D$ 와 가역 미분가능 변환 $\mathbf{y} = U(\mathbf{x})$ 에 대해, 변환된 확률변수 $\mathbf{y}$ 의 확률밀도함수 $f(\mathbf{y})$ 는 다음과 같습니다:

$$f(\mathbf{y}) = f_x(U^{-1}(\mathbf{y})) \cdot \left| \det \left( \frac{\partial}{\partial \mathbf{y}} U^{-1}(\mathbf{y}) \right) \right| \quad (\text{Theorem 6.16 / Eq 6.144})$$

$$\text{또는 역함수 정리 적용 시: } f(\mathbf{y}) = f_x(\mathbf{x}) \cdot \left| \det J_U(\mathbf{x}) \right|^{-1}$$

- 야코비안 행렬식 ($\det J$): $D$ 차원 미소 초입체(Hyper-cube)의 부피가 변환 $U$ 에 의해 평행초체(Parallelepiped)로 변형될 때의 부피 변화 비율(Volume Change Factor)을 뜻합니다.


### 💡 [Example 6.17: 선형 변환과 다변량 가우시안의 수식적 유도]

표준 정규분포 $\mathbf{x} \sim \mathcal{N}(\mathbf{0}, I)$ ($f(\mathbf{x}) = \frac{1}{2\pi} \exp(-\frac{1}{2} \mathbf{x}^\top \mathbf{x})$) 에 2D 선형 변환 $\mathbf{y} = A\mathbf{x}$ ($A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$) 를 가할 때:

1. 역변환: $\mathbf{x} = A^{-1}\mathbf{y} = \frac{1}{ad-bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix} \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} \quad (\text{Eq 6.147})$
2. 야코비안 행렬식: $\left| \det \left( \frac{\partial}{\partial \mathbf{y}} A^{-1}\mathbf{y} \right) \right| = |\det(A^{-1})| = \frac{1}{|ad-bc|} \quad (\text{Eq 6.150})$
3. 밀도 변환 적용:
   $$f(\mathbf{y}) = f_x(A^{-1}\mathbf{y}) \cdot |\det(A^{-1})| = \frac{1}{2\pi} \exp\left( -\frac{1}{2} \mathbf{y}^\top A^{-\top} A^{-1} \mathbf{y} \right) |ad-bc|^{-1} \quad (\text{Eq 6.151})$$

- 결과: 변환된 밀도 $f(\mathbf{y})$ 는 평균이 $\mathbf{0}$ 이고 공분산 행렬이 $\Sigma = AA^\top$ 인 완벽한 다변량 가우시안 분포 $\mathcal{N}(\mathbf{0}, AA^\top)$ 로 수식 유도됩니다!


## 🧠 3. 4단계 정밀 개념 해설


### 1️⃣ [1단계 개념 정의]
- 확률적분변환 (Probability Integral Transform $Y = F_X(X) \sim \mathcal{U}[0, 1]$): 임의의 연속 분포의 CDF 값은 무조건 단위 균등분포가 된다는 정리로, 역변환 난수 샘플링의 기초입니다.
- 다변량 변수 변환 ($f(\mathbf{y}) = f_x(U^{-1}(\mathbf{y})) \cdot |\det J_{U^{-1}}(\mathbf{y})|$): 공간 변환에 따른 확률 밀도의 부피 왜곡률을 야코비안 행렬식(Jacobian Determinant)으로 보정하는 밀도 변환 정리입니다.


### 2️⃣ [2단계 왜 쓰는가?]
- 복잡한 복합 생성 모델의 정확한 우도(Likelihood) 산출: 딥러닝에서 단순한 표준정규분포 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I)$ 를 복잡한 신경망 변환 $\mathbf{x} = g_\theta(\mathbf{z})$ 에 통과시켰을 때, 생성된 이미지/음성의 정확한 로그 우도 $\ln p(\mathbf{x})$ 를 야코비안 행렬식으로 정확히 트래킹하기 위해 사용합니다.
- 임의의 난수 생성 엔진 구축: 균등 난수 $u \sim \mathcal{U}[0, 1]$ 하나만 있으면 역CDF $F_X^{-1}(u)$ 연산을 통해 가우시안, 지수, 지프 분포 등의 난수를 컴퓨터로 즉시 생성하기 위해 사용합니다.


### 3️⃣ [3단계 상황별 직관 & Trade-off]
- VAE/GAN vs 노말라이징 플로우 (Normalizing Flows):
  - VAE/GAN: 잠재 공간 변환 시 야코비안 계산을 하지 않으므로 샘플링이 자유롭고 계산이 빠르지만, 생성 데이터의 정확한 확률 밀도 $p(\mathbf{x})$ 를 알지 못함.
  - 노말라이징 플로우: 가역 신경망(Invertible Neural Network)과 야코비안 행렬식 $\det J$ 를 직접 계산하므로 차원이 유지되어야(Same Dimension) 하고 설계에 제한이 있지만, exact likelihood $p(\mathbf{x})$ 계산이 가능함.


### 4️⃣ [4단계 실전 AI 연결고리]
- 노말라이징 플로우 (Normalizing Flows - RealNVP, Glow):
  가역 신경망 구조(Coupling Layers)를 연쇄 연결하여 야코비안 행렬식이 삼각행렬이 되도록 설계함으로써 $\ln p(\mathbf{x}) = \ln p(\mathbf{z}) + \sum \ln |\det J_i|$ 로 고화질 이미지 생성 및 밀도 추정 수행.
- 역변환 샘플링 (Inverse Transform Sampling):
  컴퓨터 난수 생성기(NumPy, PyTorch)에서 uniform 난수로부터 지수분포($X = -\frac{1}{\lambda}\ln(1-U)$) 및 카이제곱 난수 동적 생성.
- 정보기하학 (Information Geometry & KL Divergence):
  확률분포가 형성하는 리만 다양체(Statistical Manifold) 위에서의 거리 척도로 Kullback-Leibler (KL) Divergence 및 Bregman Divergence 적용.
