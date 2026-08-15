# 📐 04. 행렬 분해, 고유값/고유벡터, SVD (Eigen & SVD)

## 1. ⚔️ 근본 개념 정의 & 존재 이유
- 고유값/고유벡터 ($Ax = \lambda x$): 변환 후에도 방향이 변하지 않는 고집 센 고유축($x$)과 길이 팽창율($\lambda$).
- Spectral Theorem: 실수 대칭행렬($A=A^T$)은 항상 서로 직교하는 고유벡터로 분해됨 ($A = Q \Lambda Q^T$).
- SVD ($A = U \Sigma V^T$): 세상의 모든 직사각형 행렬($m \times n$)을 [회전 $\rightarrow$ 수축/팽창 $\rightarrow$ 회전] 3단계 축으로 해체.

---

## 📝 2. MML 교재 연습문제 풀이 (MML Ch 4.1 ~ 4.2)

### [Problem 6] Ex 4.1 - Spectral Theorem 대칭행렬 직교 분해 백지 증명
- 증명: $\lambda_1 x_1^T x_2 = (Ax_1)^T x_2 = x_1^T A^T x_2 = x_1^T A x_2 = \lambda_2 x_1^T x_2$
  - $(\lambda_1 - \lambda_2)(x_1^T x_2) = 0 \implies \mathbf{x_1 \perp x_2}$ 증명 완료!
  - 정규직교행렬 $Q^T Q = I \implies \mathbf{A = Q \Lambda Q^T}$.

### [Problem 7] Ex 4.2 - SVD와 $A^TA$ Eigendecomposition 대입 증명
- 증명: $A^T A = (U \Sigma V^T)^T (U \Sigma V^T) = V \Sigma^T U^T U \Sigma V^T = \mathbf{V (\Sigma^T \Sigma) V^T}$
  - $V$는 $A^T A$의 고유벡터, 特異値 $\sigma_i = \sqrt{\lambda_i(A^TA)}$.

---

## 🔍 3. 비판적 맹점 & 실전 AI 연결
- Eigendecomposition 한계: $n \times n$ 정방행렬에서만 작동, 비대칭행렬 시 축 비틀림 ➡️ SVD로 극복.
- SVD Outlier 민감성: 제곱 오차 한계 ➡️ $L_1$ Norm의 Robust PCA ($M = L + S$) 로 비디오 배경/객체 분리.
- LoRA (Low-Rank Adaptation): SVD의 Truncated SVD 원리로 대형 모델 가중치 $\Delta W$를 상위 Rank $r$개의 $B \times A$로 쪼개 파라미터 99.9% 절감.
