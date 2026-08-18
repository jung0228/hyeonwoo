# SVD & Eigendecomposition (특이값 분해 및 고유값)

## 핵심 아이디어
임의의 $m \times n$ 행렬을 회전(정규직교 기저 변환), 스케일링(특이값에 의한 신축), 다시 회전이라는 세 개의 기하학적 기본 변환의 곱으로 완벽히 분해하여, 행렬의 본질적인 랭크와 에너지(분산)를 파악하는 선형대수학의 궁극적 분해 정리입니다.

---

## 핵심 수식

### 1. 특이값 분해 (Singular Value Decomposition)
임의의 행렬 $A \in \mathbb{R}^{m \times n}$에 대해:
$$A = U \Sigma V^T$$
* $U \in \mathbb{R}^{m \times m}$: $A A^T$의 정규직교 고유벡터들 (좌특이벡터, $U^T U = I$)
* $\Sigma \in \mathbb{R}^{m \times n}$: 대각 성분에 특이값 $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$을 갖는 행렬
* $V \in \mathbb{R}^{n \times n}$: $A^T A$의 정규직교 고유벡터들 (우특이벡터, $V^T V = I$)

### 2. 최적 저랭크 근사 (Eckart-Young-Mirsky Theorem)
랭크 $k < r$인 가장 최적의 근사 행렬 $A_k$:
$$A_k = \sum_{i=1}^{k} \sigma_i u_i v_i^T, \quad \min_{\text{rank}(B)=k} \|A - B\|_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}$$

### 3. 고유값 분해 (Eigendecomposition)
정방 대칭 행렬 $S \in \mathbb{R}^{n \times n}$에 대해 (Spectral Theorem):
$$S = Q \Lambda Q^T \quad (Q \text{는 직교행렬}, \Lambda \text{는 실수 고유값 대각행렬})$$

---

## 직관적 설명
복잡하게 비틀린 고차원 데이터의 타원체를 가장 긴 주축(장축), 두 번째 긴 축, 세 번째 축 순서로 정렬하여 관찰하는 것입니다. 가장 큰 특이값 몇 개만 남겨도 원본 데이터나 이미지의 90% 이상의 핵심 정보를 거의 완벽히 복원할 수 있습니다.

---

## 연결 개념
- [[pca]] : SVD를 공분산 행렬에 적용한 차원 축소 알고리즘
- [[linear_algebra_ch2_6_basis_rank]] : 행렬의 기저, 차원, 행공간·열공간 분해
- [[linear_algebra_ch3_7_orthogonal_projections]] : 정규직교 기저로의 사영과 최소제곱해

---

## 참고
- Mathematics for Machine Learning (Deisenroth et al., Chapter 4)
- Gilbert Strang, Linear Algebra and Its Applications
