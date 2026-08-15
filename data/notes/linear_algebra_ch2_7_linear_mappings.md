# 📐 2.7 Linear Mappings (선형 사상)

> **POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.7 완전 해부**

---

## 1. ⚔️ 4단계 개념 구조화

### 1️⃣ [1단계 명확한 개념 정의]
- **선형 사상 (Linear Mapping)**: $T(u+v) = T(u)+T(v)$ 및 $T(cu) = cT(u)$를 만족하는 사상.
- **Kernel / Nullspace ($\ker(T)$)**: $T(x) = 0$ 으로 소실되는 입력 부분공간.
- **Image / Column Space ($\text{Im}(T)$)**: $T(x)$에 의해 출력 공간에 도달하는 부분공간.
- **Rank-Nullity Theorem**: $\dim(V) = \text{Nullity}(T) + \text{Rank}(T)$

---

### 2️⃣ [2단계 실전 AI 연결고리]
- **Autoencoder 차원 소실**: 라텐트 공간으로 축소 시 소실되는 정보가 $\text{Nullity}$, 살아남아 생성되는 정보가 $\text{Rank}$.
