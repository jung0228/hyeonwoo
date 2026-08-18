# Information Theory & Entropy (정보이론 및 엔트로피)

## 핵심 아이디어
불확실성(Uncertainty)의 정도를 수학적으로 정량화하고, 두 확률 분포 간의 차이와 정보의 전달 효율을 측정하여 머신러닝의 손실 함수(Cross-Entropy Loss, KL Divergence) 및 생성 모델의 근본 원리를 제공하는 수학적 토대입니다.

---

## 핵심 수식

### 1. 섀넌 엔트로피 (Shannon Entropy)
이산 확률 변수 $X$에 대해 정보량의 기댓값:
$$H(P) = -\sum_{x} P(x) \log_2 P(x) = \mathbb{E}_{X \sim P}[-\log P(X)]$$

### 2. 쿨백-라이블러 발산 (KL Divergence / Relative Entropy)
실제 분포 $P$와 근사 분포 $Q$ 사이의 정보 손실/거리 척도:
$$D_{\text{KL}}(P \parallel Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} = \mathbb{E}_{P}\left[\log \frac{P(X)}{Q(X)}\right] \ge 0$$

### 3. 교차 엔트로피 (Cross-Entropy)
$$H(P, Q) = -\sum_{x} P(x) \log Q(x) = H(P) + D_{\text{KL}}(P \parallel Q)$$
- 머신러닝에서 정답 분포 $P$가 고정되었을 때, $H(P, Q)$ 최소화는 $D_{\text{KL}}(P \parallel Q)$ 최소화와 완전히 일치합니다.

### 4. 상호 정보량 (Mutual Information)
$$I(X; Y) = H(X) - H(X|Y) = D_{\text{KL}}(P(X,Y) \parallel P(X)P(Y))$$

---

## 직관적 설명
"내일 아침에 해가 뜬다"는 100% 확실하므로 정보량이 0에 가깝지만, "내일 로또 1등에 당첨되었다"는 극도로 일어날 확률이 낮기 때문에 엄청난 양의 정보(Surprise)를 담고 있습니다. 엔트로피는 이러한 '놀람과 불확실성의 평균치'를 뜻합니다.

---

## 연결 개념
- [[cross_entropy]] : 딥러닝 분류 문제 및 언어 모델링의 표준 손실 함수
- [[mle_map]] : 최대우도추정(MLE)과 KL Divergence 최소화의 수학적 등가성
- [[vae]] : 잠재 변수 사후확포와 사전분포 간의 거리를 좁히는 정규화 항 ($D_{\text{KL}}$)

---

## 참고
- Claude Shannon, "A Mathematical Theory of Communication" (1948)
- David MacKay, "Information Theory, Inference, and Learning Algorithms"
