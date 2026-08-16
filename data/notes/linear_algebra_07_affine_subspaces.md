# 📐 07. 어파인 공간과 비동차 해공간 (Affine Subspaces & Non-Homogeneous Systems)

## 1. ⚔️ 근본 개념 정의 & 존재 이유
- 어파인 공간 (Affine Subspace $x_p + U$): 원점($0$)을 지나지 않는, 부분공간 $U$가 특수해 벡터 $x_p$만큼 평행이동(Translation)된 공간.
- 존재 이유: $Ax = b$ ($b \ne 0$) 처럼 원점을 지나지 않는 비동차(Non-homogeneous) 데이터 위치 관계를 수학적으로 표현하기 위함.


## 📝 2. MML 교재 연습문제 풀이 (MML Ch 2.8)

### [Problem 7] Ex 2.7 - 어파인 해공간 수식 증명
- 문제: $Ax = b$ 의 특수해 $x_p = \begin{bmatrix} 1 \\\\ 0 \\\\ 2 \end{bmatrix}$ 이고 $\text{Kernel}(A)$ 의 기저가 $v_1 = \begin{bmatrix} 2 \\\\ 1 \\\\ 0 \end{bmatrix}$ 일 때, 전체 해집합 $S$가 벡터공간이 아닌 어파인 공간(Affine Subspace)임을 증명하시오.

- 수식 유도:
  - 일반해: $S = \left\{ \begin{bmatrix} 1 \\\\ 0 \\\\ 2 \end{bmatrix} + c_1 \begin{bmatrix} 2 \\\\ 1 \\\\ 0 \end{bmatrix} \;\middle|\; c_1 \in \mathbb{R} \right\}$
  - 증명: 영벡터 $0 = [0, 0, 0]^T \notin S$ 이다 ($c_1$에 어떤 실수를 넣어도 $[0, 0, 0]^T$가 될 수 없음!).
  - $\therefore S$는 덧셈 닫힘과 영원소 공리를 위배하여 벡터공간이 아니며, 원점에서 $x_p$만큼 평행이동된 어파인 공간(Affine Subspace)이다!


## 🔍 3. 비판적 맹점 & 실전 AI 연결

### 1) 벡터공간 닫힘 공리 파탄 맹점
- 어파인 공간은 원점을 지나지 않아 벡터의 기본 공리인 덧셈 닫힘($u+v \in S$)과 스칼라배 닫힘($c u \in S$)이 파탄남.
- 따라서 일반적 선형 연산(선형 결합)을 자유롭게 적용할 수 없는 한계가 있음.

### 2) 실전 AI 연결 (신경망 Bias $b$의 표현력 확장)
- 신경망 Layer $y = Wx + b$ 에서 Bias $b$가 추가되는 순간, 피처 공간이 순수 원점을 지나는 선형 공간에서 어파인 공간(Affine Space)으로 확장되어 위치 평행이동 표현력이 비약적으로 상승함.
