---
title: 경사하강법과 최적화 알고리즘
dek: 손실 함수의 최솟값을 찾아가는 원리 — Batch GD부터 Adam까지, 그리고 복습 퀴즈.
tags: [Math]
date: Mar 2026
readtime: 12 min read
slug: gradient-descent
katex: true
---

## 왜 최적화가 필요한가

머신러닝 모델을 학습시킨다는 것은 결국 **손실 함수(loss function) $\mathcal{L}(\theta)$를 최소화하는 파라미터 $\theta$를 찾는 것**이다. 이전 포스트에서 다룬 MLE도 결국 로그 우도를 최대화(= 음의 로그 우도를 최소화)하는 최적화 문제였고, 교차 엔트로피 손실도 정보이론에서 유도된 동일한 목적 함수다.

문제는 대부분의 실전 모델에서 $\mathcal{L}(\theta)$의 닫힌 해(closed-form solution)가 존재하지 않는다는 점이다. 수백만 개의 파라미터, 비선형 활성화 함수, 복잡한 아키텍처 — 여기서 해석적으로 $\nabla_\theta \mathcal{L} = 0$을 풀 수는 없다. 그래서 우리는 **반복적으로 조금씩 내리막을 걷는** 전략을 택한다.

<div class="pullquote">
  <strong>핵심:</strong> 경사하강법은 "지금 서 있는 자리에서 가장 가파른 내리막 방향으로 한 걸음씩 이동"하는 반복 알고리즘이다.
</div>

## 경사하강법의 수학적 정의

파라미터 업데이트 규칙은 단순하다:

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}(\theta_t)$$

- $\eta$ (eta): **학습률(learning rate)** — 한 번에 얼마나 크게 이동할지
- $\nabla_\theta \mathcal{L}$: 손실 함수의 **그래디언트** — 각 파라미터에 대한 편미분 벡터

그래디언트 $\nabla_\theta \mathcal{L}$는 손실이 *가장 빠르게 증가하는* 방향을 가리키므로, 여기서 빼면 가장 빠르게 감소하는 방향으로 이동하게 된다.

<div class="callout">
  <strong>직관적 예시:</strong> 안개 낀 산에서 하산한다고 상상해보자. 시야가 없어 전체 지형을 볼 수 없지만, 발 아래 경사는 느낄 수 있다. 경사하강법은 바로 그 경사만 이용해 조금씩 아래로 내려가는 전략이다. 학습률이 너무 크면 건너편 산으로 튀어 오르고, 너무 작으면 아주 느리게 내려간다.
</div>

<svg viewBox="0 0 600 270" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:580px;display:block;margin:2rem auto">
  <defs>
    <marker id="arr" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">
      <path d="M0,0 L7,3.5 L0,7 Z" fill="#e84444"/>
    </marker>
  </defs>
  <rect width="600" height="270" fill="#f9f8f4" rx="8"/>
  <!-- Axes -->
  <line x1="50" y1="20" x2="50" y2="230" stroke="#bbb" stroke-width="1.5"/>
  <line x1="50" y1="230" x2="570" y2="230" stroke="#bbb" stroke-width="1.5"/>
  <text x="295" y="258" text-anchor="middle" font-size="12" fill="#888" font-family="sans-serif">파라미터 θ →</text>
  <text x="22" y="130" text-anchor="middle" font-size="12" fill="#888" font-family="sans-serif" transform="rotate(-90,22,130)">Loss ↑</text>
  <!-- Single clean loss curve: high-left, global min ~x=160, saddle ~x=300, local min ~x=430, high-right -->
  <path d="M 55,45 C 90,45 110,205 160,210 C 210,215 265,140 305,138 C 345,136 390,178 430,180 C 465,182 510,80 565,40"
        fill="none" stroke="#1a56c4" stroke-width="2.8"/>
  <!-- Global min -->
  <circle cx="160" cy="210" r="6" fill="#e84444"/>
  <text x="160" y="228" text-anchor="middle" font-size="11" fill="#e84444" font-family="sans-serif">global min</text>
  <!-- Local min -->
  <circle cx="430" cy="180" r="5" fill="#f59e0b"/>
  <text x="430" y="198" text-anchor="middle" font-size="11" fill="#f59e0b" font-family="sans-serif">local min</text>
  <!-- Descent steps (starting from right, stepping toward local min) -->
  <circle cx="540" cy="55" r="4" fill="#e84444" opacity="0.7"/>
  <circle cx="500" cy="68" r="4" fill="#e84444" opacity="0.7"/>
  <circle cx="470" cy="110" r="4" fill="#e84444" opacity="0.7"/>
  <circle cx="445" cy="158" r="4" fill="#e84444" opacity="0.7"/>
  <line x1="540" y1="55" x2="503" y2="67" stroke="#e84444" stroke-width="1.6" marker-end="url(#arr)" stroke-dasharray="none"/>
  <line x1="500" y1="68" x2="473" y2="108" stroke="#e84444" stroke-width="1.6" marker-end="url(#arr)"/>
  <line x1="470" y1="110" x2="447" y2="156" stroke="#e84444" stroke-width="1.6" marker-end="url(#arr)"/>
  <text x="560" y="46" font-size="11" fill="#e84444" font-family="sans-serif" text-anchor="middle">시작</text>
</svg>

## 배치 방식에 따른 세 가지 변형

### Batch Gradient Descent (전체 배치)

전체 훈련 데이터 $N$개를 모두 써서 그래디언트를 계산한다:

$$\nabla_\theta \mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta \ell(x_i, y_i; \theta)$$

- **장점**: 안정적인 수렴, 정확한 그래디언트
- **단점**: 데이터가 크면 한 스텝에 엄청난 계산량

### Stochastic Gradient Descent (SGD)

매 스텝마다 **무작위로 하나의 샘플**만 사용:

$$\nabla_\theta \mathcal{L} \approx \nabla_\theta \ell(x_i, y_i; \theta)$$

- **장점**: 빠른 업데이트, 노이즈 덕분에 local minima 탈출 가능
- **단점**: 경로가 불안정(zigzag)

### Mini-batch Gradient Descent

**$B$개의 샘플**로 그래디언트를 추정 (실전에서 거의 항상 이 방식):

$$\nabla_\theta \mathcal{L} \approx \frac{1}{B} \sum_{i \in \mathcal{B}} \nabla_\theta \ell(x_i, y_i; \theta)$$

배치 크기 $B$는 보통 32~512 사이. 배치가 클수록 배치 GD에, 작을수록 SGD에 가까워진다.

<div class="ornament">· · ·</div>

## 모멘텀(Momentum)

SGD의 zigzag 문제를 해결하는 첫 번째 방법이 모멘텀이다. **과거 그래디언트의 방향을 "속도"로 누적**해서 관성처럼 활용한다:

$$v_{t+1} = \beta v_t + (1-\beta)\nabla_\theta \mathcal{L}(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta \cdot v_{t+1}$$

$\beta$는 보통 0.9. 좁고 긴 골짜기(ill-conditioned surface)에서 특히 효과적이다 — 방향이 일정한 축은 속도가 붙고, 진동하는 축은 상쇄된다.

## Adam: 실전의 왕

**Adam(Adaptive Moment Estimation)**은 오늘날 딥러닝에서 기본으로 쓰이는 옵티마이저다. 그래디언트의 1차 모멘트(평균)와 2차 모멘트(분산)를 동시에 추적한다:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla_\theta \mathcal{L}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla_\theta \mathcal{L})^2$$

편향 보정 후 업데이트:

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$

- **$\hat{v}_t$의 역할**: 자주 업데이트된 파라미터는 분모가 커져 학습률이 작아짐 → **파라미터별 적응형 학습률**
- **기본값**: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

<div class="pullquote">
  <strong>왜 Adam이 강력한가:</strong> 각 파라미터마다 학습률을 자동으로 조절하기 때문에 하이퍼파라미터에 덜 민감하고, 희소(sparse)한 그래디언트에서도 잘 작동한다.
</div>

## 학습률의 중요성

학습률 $\eta$는 최적화에서 가장 중요한 하이퍼파라미터다:

- **너무 크면**: 수렴하지 않고 발산하거나 loss가 폭발
- **너무 작으면**: 수렴은 하지만 훈련이 너무 오래 걸림
- **적절하면**: 빠르고 안정적으로 좋은 최솟값에 도달

실전에서는 **학습률 스케줄링**을 함께 쓴다 — 처음엔 크게, 점점 작아지도록. 대표적으로 Cosine Annealing, Warmup + Decay 등이 있다.

<div class="ornament">· · ·</div>

## 연결: MLE, 정보이론, 그리고 경사하강법

이전 포스트들에서 다룬 개념들이 어떻게 연결되는지 정리해보자:

1. **MLE** → 우도 $p(D|\theta)$를 최대화 = 음의 로그 우도 $-\log p(D|\theta)$를 최소화
2. **교차 엔트로피** → $H(p, q) = -\sum p \log q$ 는 분류 문제의 손실 함수이자 MLE의 다른 표현
3. **경사하강법** → 위 손실 함수를 실제로 최소화하는 수치적 알고리즘
4. **MAP** → 정규화항(L2 regularization = Gaussian prior, L1 = Laplace prior)이 붙은 경사하강법

$$\underbrace{-\log p(\theta|D)}_{\text{MAP 목적함수}} = \underbrace{-\log p(D|\theta)}_{\text{Cross-Entropy Loss}} + \underbrace{-\log p(\theta)}_{\text{Regularizer}}$$

<svg viewBox="0 0 620 180" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:600px;display:block;margin:2rem auto">
  <rect width="620" height="180" fill="#f9f8f4" rx="8"/>
  <!-- Nodes -->
  <rect x="20" y="65" width="110" height="44" rx="6" fill="#dbeafe" stroke="#93c5fd" stroke-width="1.5"/>
  <text x="75" y="83" text-anchor="middle" font-size="12" fill="#1e3a8a" font-family="sans-serif">MLE / MAP</text>
  <text x="75" y="99" text-anchor="middle" font-size="11" fill="#1e3a8a" font-family="sans-serif">목적 함수 정의</text>
  <rect x="180" y="65" width="130" height="44" rx="6" fill="#dcfce7" stroke="#86efac" stroke-width="1.5"/>
  <text x="245" y="83" text-anchor="middle" font-size="12" fill="#14532d" font-family="sans-serif">정보이론</text>
  <text x="245" y="99" text-anchor="middle" font-size="11" fill="#14532d" font-family="sans-serif">Cross-Entropy Loss</text>
  <rect x="360" y="65" width="120" height="44" rx="6" fill="#fef3c7" stroke="#fcd34d" stroke-width="1.5"/>
  <text x="420" y="83" text-anchor="middle" font-size="12" fill="#78350f" font-family="sans-serif">경사하강법</text>
  <text x="420" y="99" text-anchor="middle" font-size="11" fill="#78350f" font-family="sans-serif">수치적 최적화</text>
  <rect x="530" y="65" width="70" height="44" rx="6" fill="#fce7f3" stroke="#f9a8d4" stroke-width="1.5"/>
  <text x="565" y="83" text-anchor="middle" font-size="12" fill="#831843" font-family="sans-serif">학습된</text>
  <text x="565" y="99" text-anchor="middle" font-size="11" fill="#831843" font-family="sans-serif">모델 $\theta^*$</text>
  <!-- Arrows -->
  <line x1="130" y1="87" x2="178" y2="87" stroke="#888" stroke-width="1.8" marker-end="url(#a2)"/>
  <line x1="310" y1="87" x2="358" y2="87" stroke="#888" stroke-width="1.8" marker-end="url(#a2)"/>
  <line x1="480" y1="87" x2="528" y2="87" stroke="#888" stroke-width="1.8" marker-end="url(#a2)"/>
  <defs>
    <marker id="a2" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#888"/>
    </marker>
  </defs>
  <text x="310" y="160" text-anchor="middle" font-size="12" fill="#555" font-family="sans-serif">딥러닝 학습 파이프라인</text>
</svg>

<div class="ornament">· · ·</div>

## 복습 퀴즈

지금까지 다룬 수학 포스트 전반을 복습하는 퀴즈다. 각 문제를 풀고 "정답 확인" 버튼을 눌러보자.

<div id="quiz-container" style="margin:2rem 0">

<style>
.quiz-block {
  background: #fff;
  border: 1px solid #d8d8d2;
  border-radius: 10px;
  padding: 1.4rem 1.6rem;
  margin-bottom: 1.4rem;
}
.quiz-block .q-num {
  font-family: -apple-system, sans-serif;
  font-size: 0.75rem;
  font-weight: 700;
  color: #1a56c4;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 0.5rem;
}
.quiz-block .q-text {
  font-size: 1.05rem;
  font-weight: 600;
  margin-bottom: 1rem;
  line-height: 1.5;
}
.quiz-block label {
  display: flex;
  align-items: flex-start;
  gap: 0.5rem;
  margin-bottom: 0.55rem;
  cursor: pointer;
  font-size: 0.97rem;
  line-height: 1.5;
}
.quiz-block input[type=radio] { margin-top: 3px; flex-shrink: 0; }
.quiz-btn {
  margin-top: 0.8rem;
  padding: 0.4rem 1.1rem;
  background: #1a56c4;
  color: #fff;
  border: none;
  border-radius: 5px;
  font-family: -apple-system, sans-serif;
  font-size: 0.85rem;
  cursor: pointer;
}
.quiz-btn:hover { background: #1446a8; }
.quiz-result {
  margin-top: 0.8rem;
  padding: 0.6rem 1rem;
  border-radius: 6px;
  font-size: 0.93rem;
  display: none;
  line-height: 1.6;
}
.quiz-result.correct { background: #d4f7d4; color: #145214; border: 1px solid #86efac; }
.quiz-result.wrong   { background: #ffd6d6; color: #7f1d1d; border: 1px solid #fca5a5; }
.quiz-score-box {
  background: #fef3c7;
  border: 1px solid #fcd34d;
  border-radius: 10px;
  padding: 1.2rem 1.6rem;
  text-align: center;
  margin-top: 1rem;
  display: none;
}
.quiz-score-box .score-num {
  font-size: 2.4rem;
  font-weight: 800;
  color: #92400e;
  font-family: 'Playfair Display', serif;
}
.quiz-score-box .score-label {
  font-size: 0.95rem;
  color: #78350f;
  font-family: -apple-system, sans-serif;
}
</style>

<div class="quiz-block" id="q1">
  <div class="q-num">문제 1 · 경사하강법</div>
  <div class="q-text">학습률(learning rate) $\eta$가 너무 클 때 나타나는 현상은?</div>
  <label><input type="radio" name="q1" value="a"> 수렴이 매우 느려진다</label>
  <label><input type="radio" name="q1" value="b"> 손실이 발산하거나 수렴하지 않는다</label>
  <label><input type="radio" name="q1" value="c"> 항상 전역 최솟값(global minimum)에 도달한다</label>
  <label><input type="radio" name="q1" value="d"> 그래디언트가 0이 된다</label>
  <button class="quiz-btn" onclick="check('q1','b','학습률이 너무 크면 파라미터가 최솟값을 지나쳐 반대편으로 튀는 과정을 반복하며 발산합니다. 너무 작으면 수렴은 하지만 느려집니다.')">정답 확인</button>
  <div class="quiz-result" id="q1-result"></div>
</div>

<div class="quiz-block" id="q2">
  <div class="q-num">문제 2 · MLE</div>
  <div class="q-text">최대 우도 추정(MLE)의 목적을 올바르게 설명한 것은?</div>
  <label><input type="radio" name="q2" value="a"> 파라미터의 사전 분포 $p(\theta)$를 최대화한다</label>
  <label><input type="radio" name="q2" value="b"> 관측 데이터가 주어졌을 때 사후 확률 $p(\theta|D)$를 최대화한다</label>
  <label><input type="radio" name="q2" value="c"> 관측 데이터가 주어진 파라미터에서 나올 확률 $p(D|\theta)$를 최대화한다</label>
  <label><input type="radio" name="q2" value="d"> 손실 함수와 정규화항의 합을 최소화한다</label>
  <button class="quiz-btn" onclick="check('q2','c','MLE는 likelihood $p(D|\\theta)$를 최대화하는 파라미터를 찾습니다. 사전 분포를 추가하면 MAP가 됩니다. (d)는 MAP에 해당합니다.')">정답 확인</button>
  <div class="quiz-result" id="q2-result"></div>
</div>

<div class="quiz-block" id="q3">
  <div class="q-num">문제 3 · 정보이론</div>
  <div class="q-text">엔트로피 $H(X)$가 최대가 되는 조건은?</div>
  <label><input type="radio" name="q3" value="a"> 하나의 사건이 확률 1을 가질 때</label>
  <label><input type="radio" name="q3" value="b"> 모든 사건이 균등한 확률을 가질 때</label>
  <label><input type="radio" name="q3" value="c"> 사건의 수가 1개일 때</label>
  <label><input type="radio" name="q3" value="d"> 확률 분포가 정규분포를 따를 때</label>
  <button class="quiz-btn" onclick="check('q3','b','엔트로피는 불확실성의 척도입니다. 모든 사건이 균등한 확률을 가질 때(균등 분포) 불확실성이 가장 크므로 엔트로피가 최대가 됩니다. 하나의 사건만 확실하면 $H=0$입니다.')">정답 확인</button>
  <div class="quiz-result" id="q3-result"></div>
</div>

<div class="quiz-block" id="q4">
  <div class="q-num">문제 4 · 경사하강법</div>
  <div class="q-text">Adam 옵티마이저가 일반 SGD보다 유리한 주요 이유는?</div>
  <label><input type="radio" name="q4" value="a"> 그래디언트를 계산하지 않아도 된다</label>
  <label><input type="radio" name="q4" value="b"> 각 파라미터마다 적응형 학습률을 적용한다</label>
  <label><input type="radio" name="q4" value="c"> 배치 크기를 자동으로 결정한다</label>
  <label><input type="radio" name="q4" value="d"> 항상 전역 최솟값을 보장한다</label>
  <button class="quiz-btn" onclick="check('q4','b','Adam은 그래디언트의 2차 모멘트(분산)를 추적하여 자주 업데이트된 파라미터는 학습률을 줄이고, 희소한 파라미터는 학습률을 높이는 적응형 방식입니다.')">정답 확인</button>
  <div class="quiz-result" id="q4-result"></div>
</div>

<div class="quiz-block" id="q5">
  <div class="q-num">문제 5 · 정보이론</div>
  <div class="q-text">KL Divergence $D_{KL}(P \| Q)$에 대한 설명으로 틀린 것은?</div>
  <label><input type="radio" name="q5" value="a"> $P = Q$이면 $D_{KL}(P \| Q) = 0$이다</label>
  <label><input type="radio" name="q5" value="b"> 항상 $D_{KL}(P \| Q) \geq 0$이다</label>
  <label><input type="radio" name="q5" value="c"> $D_{KL}(P \| Q) = D_{KL}(Q \| P)$이다 (대칭성)</label>
  <label><input type="radio" name="q5" value="d"> 분포 $P$를 분포 $Q$로 근사할 때의 정보 손실을 나타낸다</label>
  <button class="quiz-btn" onclick="check('q5','c','KL Divergence는 비대칭입니다. 일반적으로 $D_{KL}(P\\|Q) \\neq D_{KL}(Q\\|P)$입니다. 이것이 KL Divergence가 진정한 \"거리 지표\"가 아닌 이유입니다.')">정답 확인</button>
  <div class="quiz-result" id="q5-result"></div>
</div>

<div class="quiz-block" id="q6">
  <div class="q-num">문제 6 · MAP</div>
  <div class="q-text">L2 정규화(weight decay)를 추가한 손실 함수가 MAP 추정에 대응될 때, 파라미터의 사전 분포는?</div>
  <label><input type="radio" name="q6" value="a"> Laplace 분포</label>
  <label><input type="radio" name="q6" value="b"> 균등(Uniform) 분포</label>
  <label><input type="radio" name="q6" value="c"> Gaussian(정규) 분포</label>
  <label><input type="radio" name="q6" value="d"> Bernoulli 분포</label>
  <button class="quiz-btn" onclick="check('q6','c','$-\\log p(\\theta) \\propto \\|\\theta\\|^2$ 이 되려면 $p(\\theta) \\propto \\exp(-\\|\\theta\\|^2/2\\sigma^2)$, 즉 Gaussian 사전 분포입니다. L1 정규화는 Laplace 사전 분포에 해당합니다.')">정답 확인</button>
  <div class="quiz-result" id="q6-result"></div>
</div>

<div class="quiz-block" id="q7">
  <div class="q-num">문제 7 · 고유값 / SVD</div>
  <div class="q-text">행렬 $A$의 특이값 분해(SVD)를 $A = U\Sigma V^T$라 할 때, 특이값(singular values)은 무엇인가?</div>
  <label><input type="radio" name="q7" value="a"> 행렬 $A$의 고유값(eigenvalues)</label>
  <label><input type="radio" name="q7" value="b"> 행렬 $A^TA$의 고유값의 제곱근</label>
  <label><input type="radio" name="q7" value="c"> 행렬 $U$의 대각 원소</label>
  <label><input type="radio" name="q7" value="d"> 행렬 $A$의 행렬식(determinant)의 제곱근</label>
  <button class="quiz-btn" onclick="check('q7','b','특이값 $\\sigma_i$는 $A^TA$의 고유값 $\\lambda_i$의 제곱근입니다: $\\sigma_i = \\sqrt{\\lambda_i}$. $U$는 $AA^T$의 고유벡터, $V$는 $A^TA$의 고유벡터로 구성됩니다.')">정답 확인</button>
  <div class="quiz-result" id="q7-result"></div>
</div>

<div class="quiz-block" id="q8">
  <div class="q-num">문제 8 · 경사하강법</div>
  <div class="q-text">모멘텀(Momentum)을 사용하는 주된 이유는?</div>
  <label><input type="radio" name="q8" value="a"> 그래디언트 계산량을 줄이기 위해</label>
  <label><input type="radio" name="q8" value="b"> 배치 크기를 자동 조절하기 위해</label>
  <label><input type="radio" name="q8" value="c"> 진동(oscillation)을 줄이고 수렴을 가속화하기 위해</label>
  <label><input type="radio" name="q8" value="d"> 사전 분포를 통합하기 위해</label>
  <button class="quiz-btn" onclick="check('q8','c','모멘텀은 과거 그래디언트 방향을 누적하여 관성을 만듭니다. 방향이 일정한 축에서는 속도가 붙어 수렴이 빨라지고, 진동하는 축에서는 반대 방향끼리 상쇄되어 zigzag가 줄어듭니다.')">정답 확인</button>
  <div class="quiz-result" id="q8-result"></div>
</div>

<div class="quiz-block" id="q9">
  <div class="q-num">문제 9 · 정보이론</div>
  <div class="q-text">분류 모델의 Cross-Entropy Loss $\mathcal{L} = -\sum_k y_k \log \hat{y}_k$가 MLE와 동치인 이유는?</div>
  <label><input type="radio" name="q9" value="a"> 두 공식 모두 분산을 최소화하기 때문이다</label>
  <label><input type="radio" name="q9" value="b"> 카테고리 분포의 음의 로그 우도가 정확히 교차 엔트로피이기 때문이다</label>
  <label><input type="radio" name="q9" value="c"> SGD가 교차 엔트로피를 자동으로 최소화하기 때문이다</label>
  <label><input type="radio" name="q9" value="d"> 정규화항이 0일 때만 성립한다</label>
  <button class="quiz-btn" onclick="check('q9','b','카테고리 분포에서 데이터의 로그 우도는 $\\sum_k y_k \\log \\hat{y}_k$이고, 이를 최대화 = 교차 엔트로피를 최소화. 따라서 분류 문제의 Cross-Entropy Loss는 MLE의 직접적인 구현입니다.')">정답 확인</button>
  <div class="quiz-result" id="q9-result"></div>
</div>

<div class="quiz-block" id="q10">
  <div class="q-num">문제 10 · 종합</div>
  <div class="q-text">딥러닝 모델 학습 과정에서 "과적합(overfitting)"을 방지하는 데 MAP 추정이 도움이 되는 이유는?</div>
  <label><input type="radio" name="q10" value="a"> MAP는 학습 데이터를 더 많이 사용하기 때문이다</label>
  <label><input type="radio" name="q10" value="b"> 사전 분포가 파라미터가 너무 극단적인 값을 갖지 않도록 제약하기 때문이다</label>
  <label><input type="radio" name="q10" value="c"> MAP는 항상 MLE보다 작은 모델을 생성하기 때문이다</label>
  <label><input type="radio" name="q10" value="d"> 경사하강법의 학습률을 자동으로 줄이기 때문이다</label>
  <button class="quiz-btn" onclick="check('q10','b','MAP의 사전 분포 $p(\\theta)$는 파라미터에 대한 우리의 믿음(예: 작은 값을 선호)을 반영합니다. 이것이 정규화항으로 작용하여 파라미터가 데이터에만 과도하게 맞추는 것을 방지합니다.')">정답 확인</button>
  <div class="quiz-result" id="q10-result"></div>
</div>

<div style="text-align:center;margin-top:1.5rem">
  <button class="quiz-btn" style="background:#065f46;font-size:0.95rem;padding:0.6rem 1.8rem" onclick="showScore()">전체 점수 확인</button>
</div>

<div class="quiz-score-box" id="score-box">
  <div class="score-num" id="score-num">0 / 10</div>
  <div class="score-label" id="score-label"></div>
</div>

</div>

<script>
(function() {
  const answers = {};
  window.check = function(id, correct, explanation) {
    const sel = document.querySelector('input[name="' + id + '"]:checked');
    const resultEl = document.getElementById(id + '-result');
    if (!sel) { resultEl.style.display='block'; resultEl.className='quiz-result wrong'; resultEl.innerHTML='보기를 선택해주세요.'; return; }
    const chosen = sel.value;
    answers[id] = (chosen === correct);
    if (chosen === correct) {
      resultEl.className = 'quiz-result correct';
      resultEl.innerHTML = '✓ 정답! ' + explanation;
    } else {
      resultEl.className = 'quiz-result wrong';
      resultEl.innerHTML = '✗ 오답. ' + explanation;
    }
    resultEl.style.display = 'block';
  };
  window.showScore = function() {
    const total = 10;
    let correct = 0;
    for (let k in answers) { if (answers[k]) correct++; }
    const box = document.getElementById('score-box');
    document.getElementById('score-num').textContent = correct + ' / ' + total;
    const labels = ['다시 한번 복습해보자!', '기초를 다져보자!', '조금 더 힘내자!', '절반 돌파!', '꽤 잘하고 있어!', '훌륭해!', '거의 다 왔어!', '아주 잘했어!', '대단해!', '완벽한 이해!', '만점 달성! 완벽하다 🎉'];
    document.getElementById('score-label').textContent = labels[correct] || '';
    box.style.display = 'block';
    box.scrollIntoView({behavior:'smooth', block:'center'});
  };
})();
</script>

<div class="footnote">
  참고: <a href="mle-map.html">MLE & MAP 추정</a> · <a href="information-theory.html">엔트로피·교차 엔트로피·KL Divergence</a> · <a href="eigendecomposition-svd.html">고유값 분해와 SVD</a>
</div>
