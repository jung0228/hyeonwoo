# 📐 3.9 & 3.10 Rotations and Chapter 3 Summary (회전 변환과 해석기하학 총결산)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 3.9 & 3.10 전수 분석 & 4단계 정밀 해설 노트


## 🌐 0. 3.8절과의 연결 및 자연스러운 빌드업: 왜 "회전 변환(Rotations)"으로 Chapter 3을 매듭짓는가?

우리는 지난 3.4절과 3.8절에서 직교 행렬(Orthogonal Matrix, $A^{-1} = A^\top$)이 벡터 공간의 길이와 각도를 100% 온전히 보존한다는 사실을 증명했습니다.

3.9절은 이러한 직교 변환의 가장 대표적이자 핵심적인 형태인 회전 변환(Rotation)을 2차원 평면, 3차원 공간, 그리고 일반 $n$차원 공간(기븐스 회전 Givens Rotation)으로 확장하여 체계적으로 규명합니다.

이어지는 3.10절(Further Reading)에서는 Chapter 3 전체에서 다룬 내적, 노름, 그람-슈미트, 직교 정사영, 회전 변환이 후속 AI 머신러닝 알고리즘(커널 기법, 가우시안 프로세스, 선형 회귀, 주성분 분석 PCA, 공액기울기법)으로 어떻게 뻗어나가는지 거대한 지도를 완성합니다.


## 1. ⚔️ Section 3.9: Rotations (회전 변환)


### 📌 1. 회전 변환의 수학적 정의 (Definition & Eq 3.74)

회전(Rotation)이란 유클리드 벡터 공간의 자기동형사상(Automorphism)으로서, 원점을 고정점(Fixed point)으로 유지한 채 공간의 평면을 각도 $\theta$ 만큼 회전시키는 선형 사상입니다.

- 회전 방향 규약: 양의 각도 $\theta > 0$ 에 대해 반시계 방향(Counterclockwise) 회전을 표준 관례로 정의합니다.
- 예시 회전 행렬: $R = \begin{bmatrix} -0.38 & -0.92 \\\\ 0.92 & -0.38 \end{bmatrix}$ (Eq 3.74, 각도 $\theta \approx 112.5^\circ$)
- 핵심 응용: 컴퓨터 그래픽스, 로보틱스 관절 각도 제어(Inverse Kinematics), 3차원 비전 자세 추정(Pose Estimation).


### 📌 2. 2차원 공간에서의 회전 변환 ($\mathbb{R}^2$: Section 3.9.1 & Eq 3.75~3.76)

2차원 유클리드 공간의 표준기저 $\mathbf{e}_1 = \begin{bmatrix} 1 \\\\ 0 \end{bmatrix}, \mathbf{e}_2 = \begin{bmatrix} 0 \\\\ 1 \end{bmatrix}$ 를 각도 $\theta$ 만큼 반시계 방향으로 회전시킨 상(Image)은 삼각함수에 의해 다음과 같이 결정됩니다 (Figure 3.16):

$$\Phi(\mathbf{e}_1) = \begin{bmatrix} \cos\theta \\\\ \sin\theta \end{bmatrix}, \quad \Phi(\mathbf{e}_2) = \begin{bmatrix} -\sin\theta \\\\ \cos\theta \end{bmatrix} \quad (\text{Eq 3.75})$$

이 회전된 기저 벡터들을 열벡터로 결합하면 2차원 회전 행렬 $R(\theta)$ 가 완성됩니다:

$$R(\theta) = \begin{bmatrix} \Phi(\mathbf{e}_1) & \Phi(\mathbf{e}_2) \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\\\ \sin\theta & \cos\theta \end{bmatrix} \quad (\text{Eq 3.76})$$

- 기저 변환 관점: 회전 변환은 기존 좌표계를 회전된 새로운 직교 기저계로 표현을 바꾸는 기저 변환(Basis Change) 연산입니다.


### 📌 3. 3차원 공간에서의 축 회전 변환 ($\mathbb{R}^3$: Section 3.9.2 & Eq 3.77~3.79)

3차원 공간에서는 1차원 회전축(Axis)을 고정하고 나머지 2차원 평면을 회전시킵니다.
반시계 방향의 정의는 회전축의 끝(Tip)에서 원점을 바라보는 시선 기준입니다.

1. $\mathbf{e}_1$ 축 기준 회전 ($x$축 고정, $yz$ 평면 회전):
   $$R_1(\theta) = \begin{bmatrix} 1 & 0 & 0 \\\\ 0 & \cos\theta & -\sin\theta \\\\ 0 & \sin\theta & \cos\theta \end{bmatrix} \quad (\text{Eq 3.77})$$

2. $\mathbf{e}_2$ 축 기준 회전 ($y$축 고정, $xz$ 평면 회전):
   $$R_2(\theta) = \begin{bmatrix} \cos\theta & 0 & \sin\theta \\\\ 0 & 1 & 0 \\\\ -\sin\theta & 0 & \cos\theta \end{bmatrix} \quad (\text{Eq 3.78})$$
   (주의: $y$축 끝에서 원점을 내려다볼 때 $z$축에서 $x$축 방향으로 회전하므로 부호 위치가 $R_1, R_3$ 와 반대로 배치됩니다.)

3. $\mathbf{e}_3$ 축 기준 회전 ($z$축 고정, $xy$ 평면 회전):
   $$R_3(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\\\ \sin\theta & \cos\theta & 0 \\\\ 0 & 0 & 1 \end{bmatrix} \quad (\text{Eq 3.79})$$


### 📌 4. $n$차원 공간에서의 기븐스 회전 (Givens Rotation: Section 3.9.3 & Definition 3.11)

$n$차원 공간 $\mathbb{R}^n$ 에서의 회전은 $(n-2)$개의 축을 그대로 고정하고, 특정한 두 축 $(i, j)$ 로 구성된 2차원 평면 상에서만 각도 $\theta$ 회전을 수행하는 것으로 정의됩니다. 이를 기븐스 회전(Givens Rotation)이라 부릅니다:

$$R_{ij}(\theta) = \begin{bmatrix} I_{i-1} & 0 & \dots & 0 & 0 \\\\ 0 & \cos\theta & 0 & -\sin\theta & 0 \\\\ 0 & 0 & I_{j-i-1} & 0 & 0 \\\\ 0 & \sin\theta & 0 & \cos\theta & 0 \\\\ 0 & 0 & \dots & 0 & I_{n-j} \end{bmatrix} \in \mathbb{R}^{n \times n} \quad (\text{Eq 3.80})$$

- 핵심 성분: $r_{ii} = \cos\theta, \; r_{ij} = -\sin\theta, \; r_{ji} = \sin\theta, \; r_{jj} = \cos\theta$ 이며 나머지는 단위행렬 성분 ($r_{kk} = 1$).
- 수치 선형대수 응용: QR 분해를 수행할 때 행렬의 특정 비대각 성분을 0으로 정밀 소거하는 고속 알고리즘의 핵심입니다.


### 📌 5. 회전 변환의 핵심 기하학적 성질 (Properties of Rotations: Section 3.9.4)

1. 거리 보존 (Distance Preservation):
   $$\VertR_\theta(\mathbf{x}) - R_\theta(\mathbf{y})\Vert = \Vert\mathbf{x} - \mathbf{y}\Vert$$
   (회전 후에도 두 점 사이의 유클리드 거리는 완벽히 불변입니다.)

2. 각도 보존 (Angle Preservation):
   $$\cos\omega(R_\theta \mathbf{x}, R_\theta \mathbf{y}) = \cos\omega(\mathbf{x}, \mathbf{y})$$
   (회전 후에도 두 벡터가 이루는 사이각은 전혀 왜곡되지 않습니다.)

3. 비가환성(Non-commutativity)과 차원별 차이:
   - 3차원 이상의 공간에서는 회전 변환의 순서를 바꾸면 결과가 완전히 달라집니다 ($R_1 R_2 \neq R_2 R_1$, 비가환군 Non-Abelian Group).
   - 2차원 평면 공간에서 동일한 중심점(원점)을 기준으로 회전할 때만 교환법칙이 유일하게 성립합니다 ($R(\phi)R(\theta) = R(\theta)R(\phi)$, 가환군/아벨군 Abelian Group).


## 2. ⚔️ Section 3.10: Further Reading (해석기하학의 실전 AI 확장)


### 📌 1. 최적화 및 수치 선형대수 (Optimization & Numerical Linear Algebra)
- 그람-슈미트 직교화는 수치해석의 크릴로프 부분공간 기법(Krylov Subspace Methods)인 공액기울기법(Conjugate Gradients, CG) 및 GMRES(Generalized Minimal Residual) 알고리즘의 뼈대를 이룹니다.
- 반복적으로 갱신되는 잔차 오차(Residual Error)들을 서로 상호 직교(Orthogonal)하게 만들어 최소 단계 안에 수렴하도록 보장합니다.

### 📌 2. 커널 기법과 가우시안 프로세스 (Kernel Methods & Gaussian Processes: Ch 12)
- 머신러닝의 수많은 선형 알고리즘은 오직 데이터 간의 내적 $\langle \mathbf{x}, \mathbf{y} \rangle$ 연산만으로 표현 가능합니다.
- 커널 트릭(Kernel Trick)을 사용하면 데이터를 무한차원 특징 공간으로 명시적으로 매핑하지 않고도, 커널 함수 $k(\mathbf{x}, \mathbf{y})$ 를 통해 고차원 내적을 암묵적으로 계산하여 비선형 분류(Kernel SVM, Kernel PCA)와 확률적 비선형 회귀(Gaussian Process)를 완성합니다.

### 📌 3. 직교 정사영과 머신러닝 (Projections & Machine Learning: Ch 9 & Ch 10)
- 선형 회귀 (Linear Regression - Ch 9): 타겟 데이터 벡터를 입력 특성 행렬의 열공간 위로 직교 정사영하여 잔차 제곱 오차를 최소화합니다.
- 주성분 분석 (PCA - Ch 10): 고차원 데이터를 재구성 오차(Reconstruction Error)가 최소가 되는 저차원 주성분 부분공간으로 직교 정사영하여 최적의 차원 축소를 수행합니다.


## 🧠 3. 4단계 정밀 개념 해설


### 1️⃣ [1단계 개념 정의]
- 회전 변환 (Rotation): 원점을 고정한 채 공간의 길이와 각도를 온전히 보존하면서 방향만 각도 $\theta$ 만큼 틀어주는 직교 자기동형사상($R^\top R = I, \; \det(R) = 1$)입니다.
- 기븐스 회전 (Givens Rotation): $n$차원 공간에서 선택된 두 평면 축 $(i, j)$ 에 대해서만 회전을 가하는 국소적 회전 행렬입니다.


### 2️⃣ [2단계 왜 쓰는가?]
- 데이터의 본질적 기하학적 형태(거리와 각도)를 훼손하지 않으면서, 주성분 축 정렬(PCA), 좌표계 일치(Point Cloud Registration), 수치적 행렬 삼각화(QR 분해)를 수행하기 위해 사용합니다.


### 3️⃣ [3단계 상황별 직관 & Trade-off]
- 2차원 회전 vs 3차원 이상 회전의 가환성 차이:
  - 2차원은 회전축이 원점 1개뿐이므로 회전 순서를 바꾸어도 결과가 같습니다 ($R(\theta_1)R(\theta_2) = R(\theta_2)R(\theta_1)$).
  - 3차원 이상은 회전축이 여러 개이므로, $x$축 회전 후 $y$축 회전을 하는 것과 $y$축 회전 후 $x$축 회전을 하는 것은 완전히 다른 최종 자세(Orientation)를 만들어냅니다.


### 4️⃣ [4단계 실전 AI 연결고리]
- 3D 컴퓨터 비전 & NeRF / 3D Gaussian Splatting: 카메라의 3차원 위치 및 자세(Extrinsic Matrix $[R \mid \mathbf{t}]$)를 표현할 때 $3 \times 3$ 직교 회전 행렬 $R \in \text{SO}(3)$ 을 그대로 사용합니다.
- 로보틱스 강화학습 (Reinforcement Learning): 로봇 팔 관절의 순방향/역방향 기구학(Forward/Inverse Kinematics)에서 각 관절의 회전 변환 행렬 곱으로 엔드이펙터의 3차원 위치를 제어합니다.
- Data Augmentation (컴퓨터 비전): 이미지 인식 모델 학습 시 회전 변환 $R(\theta)$ 를 무작위 적용하여 모델이 회전 불변(Rotation Invariant) 특성을 학습하도록 유도합니다.
