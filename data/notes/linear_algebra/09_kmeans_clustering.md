# 📐 09. K-Means 클러스터링의 선형대수학적 본질 & 비판적 맹점 (K-Means Clustering)

## 1. ⚔️ 근본 개념 정의 & 존재 이유
- **K-Means 클러스터링**: $N$개의 고차원 데이터를 $K$개의 군집(Cluster)으로 묶고, 각 군집의 중심점(Centroid $\mu_k$)과의 Euclidean 거리 제곱 합(Inertia / WCSS)을 최소화하는 비지도학습 알고리즘.
- **선형대수학적 본질**: 
  - 고차원 데이터 $X \in \mathbb{R}^{N \times d}$를 $K$개의 대표 기저 벡터(Centroid $\mu_1, \dots, \mu_K$) 공간으로 정사영(Projection)하여 **One-Hot 원소로 기저변환(Change of Basis)**하는 공간 사상 기술.

---

## 📝 2. 수식 유도 & 반복 최적화 (2-Step EM Algorithm)

### 1단계: Assignment Step (거리 최소화 사상)
- 데이터 $x_i$를 가장 가까운 중심점 $\mu_k$에 할당 (Indicator $r_{ik} \in \{0, 1\}$):
  $$r_{ik} = \begin{cases} 1 & \text{if } k = \arg\min_j \|x_i - \mu_j\|^2 \\ 0 & \text{otherwise} \end{cases}$$

### 2단계: Update Step (중심점 갱신 - 기저 재정비)
- 할당된 데이터들의 산술 평균으로 중심점 $\mu_k$ 갱신:
  $$\mu_k = \frac{\sum_{i=1}^N r_{ik} x_i}{\sum_{i=1}^N r_{ik}}$$

---

## 🔍 3. 비판적 맹점 & 실전 AI 연결

### ① 수치적 맹점 1: 구형(Spherical) 군집과 $L_2$ Norm 오차의 한계
- K-Means는 $L_2$ Euclidean 거리를 쓰기 때문에 **모든 군집이 완벽한 동그라미(구형) 모양이라고 강제 가정**함.
- 초승달 모양(Non-convex) 데이터나 긴 타원형 데이터가 들어오면 공간을 엉뚱하게 찢어버려 실패함.

### ② 수치적 맹점 2: 초기 중심점 민감성 & Local Minima
- 초기 중심점 $\mu_k$를 랜덤하게 잡으면 **Local Minima(지역 최적해)에 갇혀 완전히 잘못된 군집화**를 내놓음 ➡️ **K-Means++** (첫 점을 잡은 후 멀리 떨어진 점을 다음 중심점으로 선택)로 개선.

### ③ 실전 AI 연결 (VQ-VAE & LLM 수량화/Quantization)
- **VQ-VAE (Vector Quantized VAE)**: 이미지/음성의 연속적 Latent 벡터를 K-Means 원리로 코드북(Codebook)의 가장 가까운 이산 토큰(Discrete Token)으로 바꿈.
- **LLM 4-bit Quantization (가중치 양자화)**: 수억 개의 32-bit 실수 가중치를 K-Means로 $K=16$개 대표값으로 군집화하여 메모리를 80% 아낌.

---

## ⚡ 4. K-Means 시간복잡도 $O(I \cdot K \cdot N \cdot d)$ 와 3가지 극복 솔루션

- **기본 시간복잡도**: $O(I \cdot K \cdot N \cdot d)$ ($N$: 데이터 개수, $K$: 군집 개수, $d$: 차원 수, $I$: 반복 횟수).
- **문제점**: 빅데이터 환경($N \ge 1,000,000$)이나 고차원($d \ge 1,000$) 및 군집 수 $K$가 커지면 연산 병목이 폭발함.

### 🚀 해결 방안 3가지 (Engineering & Algorithmic Solutions)

1. **Mini-batch K-Means (샘플링 기반 연산)**:
   - 전체 데이터 $N$개 대신 무작위로 작은 **Mini-batch(예: 1,024개)** 샘플만 추출하여 중심점 $\mu_k$를 경사하강법(SGD) 방식으로 경량 갱신.
   - 시간복잡도를 $O(I \cdot K \cdot b \cdot d)$ ($b \ll N$) 로 파격 절감.

2. **차원 축소 전처리 (PCA / SVD / AutoEncoder)**:
   - 고차원 $d$를 그대로 넣지 않고, **PCA / SVD 기저 분해**나 AutoEncoder로 핵심 차원 $d' \ll d$ (예: $1024 \to 32$) 로 압축 후 K-Means 수행.

3. **삼각 부등식 거리 생략법 (Elkan's K-Means)**:
   - 삼각 부등식($\|x_i - \mu_c\| \le \|x_i - \mu_a\| + \|\mu_a - \mu_c\|$)을 활용해, **확실히 먼 중심점과의 거리는 일일이 $L_2$ 연산을 계산하지 않고 Skip**하여 수치 연산량을 $80\%$ 이상 절감.
