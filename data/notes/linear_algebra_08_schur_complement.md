# 📐 08. 블록 행렬과 슈르 보간 (Schur Complement & Block Matrix Inverse)

## 1. ⚔️ 근본 개념 정의 & 존재 이유
- 슈르 보간 (Schur Complement $S$): 거대한 블록 행렬 $M = \begin{bmatrix} A & B \\\\ C & D \end{bmatrix}$ 에서 특정 블록 $A$를 소거하여 얻어지는 조건부 공간 행렬 $S = D - C A^{-1} B$.
- 존재 이유: 거대 차원 행렬 전체의 역행렬을 직접 구하지 않고, 블록 단위로 쪼개어 가우시안 조건부 확률과 역행렬을 효율적으로 계산하기 위함.


## 📝 2. MML 교재 연습문제 풀이 (MML Ch 2.3)

### [Problem 8] 슈르 보간을 이용한 블록 역행렬 유도
- 문제: $M = \begin{bmatrix} A & B \\\\ C & D \end{bmatrix}$ 의 역행렬을 슈르 보간 $S = D - C A^{-1} B$ 로 표현하시오.

- 수식 유도 2단계:
  1. 가우스 소거 변환:
     $$\begin{bmatrix} I & 0 \\\\ -C A^{-1} & I \end{bmatrix} \begin{bmatrix} A & B \\\\ C & D \end{bmatrix} = \begin{bmatrix} A & B \\\\ 0 & D - C A^{-1} B \end{bmatrix} = \begin{bmatrix} A & B \\\\ 0 & S \end{bmatrix}$$

  2. Determinant 분해 공식 유도:
     $$\det(M) = \det(A) \cdot \det(S) = \det(A) \cdot \det(D - C A^{-1} B)$$


## 🔍 3. 비판적 맹점 & 실전 AI 연결

### 1) 서브 블록 가역성(Invertibility) 맹점
- 슈르 보간을 계산하려면 반드시 메인 블록 $A$의 역행렬 $A^{-1}$이 존재해야만 연산이 가능하다는 치명적 제약이 있음.

### 2) 실전 AI 연결 (Gaussian Process & VAE 조건부 확률)
- 다변량 가우시안 분포 $P(X, Y)$에서 $Y$가 주어졌을 때의 조건부 확률 $P(X \mid Y)$의 공분산 행렬 계산 수식이 바로 슈르 보간 $S = \Sigma_{XX} - \Sigma_{XY} \Sigma_{YY}^{-1} \Sigma_{YX}$ 임.
