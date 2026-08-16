# 📐 2.7 & 2.8 Linear Mappings & Affine Spaces (선형사상과 어파인 공간)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.7 & 2.8 완전 해부


## 1. ⚔️ Section 2.7: Linear Mappings (선형 사상)
- Kernel / Nullspace ($\ker(T)$): $T(x) = 0$ 으로 소실되는 입력 부분공간.
- Image / Column Space ($\text{Im}(T)$): $T(x)$에 의해 출력 공간에 도달하는 부분공간.
- Rank-Nullity Theorem: $\dim(V) = \text{Nullity}(T) + \text{Rank}(T)$


## 2. ⚔️ Section 2.8: Affine Spaces (아핀 공간)
- 아핀 공간: $L = x_0 + U$ (원점을 지나지 않고 $x_p$ 점만큼 평행 이동된 부분공간).
- 신경망 이식: $y = \sigma(W x + b)$에서 편향(Bias) $b$가 아핀 공간 평행 이동 역할 담당.
