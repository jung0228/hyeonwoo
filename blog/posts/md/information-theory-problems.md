---
title: 정보이론 심화 문제
dek: VAE의 ELBO, Forward/Reverse KL, 정보 병목 이론 — 대학원 수준 서술형 문제 5선.
desc: 정보이론 포스트의 심화 버전. ELBO 유도, KL 비대칭성, 정보 병목 이론, Rényi 엔트로피, Data Processing Inequality를 다룬다.
tags: [Math]
date: Mar 2026
readtime: 20 min read
slug: information-theory-problems
katex: true
---

이 포스트는 [정보이론 기초: 엔트로피, 교차 엔트로피, KL Divergence](information-theory.html)의 심화 문제집이다. 현대 딥러닝 이론의 핵심 개념들 — VAE, GAN, SSL, 정보 병목 — 이 모두 정보이론의 언어로 쓰여있다. 각 문제를 충분히 고민한 후 답안을 확인하자.

<style>
.prob-block{background:#fff;border:1px solid #d8d8d2;border-radius:10px;padding:1.5rem 1.7rem;margin-bottom:1.3rem}
.prob-meta{display:flex;align-items:center;gap:.6rem;margin-bottom:.65rem;flex-wrap:wrap}
.prob-num{font-family:-apple-system,sans-serif;font-size:.72rem;font-weight:700;color:#fff;background:#1a56c4;padding:.18em .75em;border-radius:20px;letter-spacing:.04em}
.prob-tag{font-family:-apple-system,sans-serif;font-size:.72rem;color:#666;border:1px solid #ccc;padding:.15em .65em;border-radius:20px}
.prob-q{font-size:1.04rem;font-weight:600;line-height:1.65;margin-bottom:1rem}
.prob-toggle{border:1px solid #bbb;background:none;padding:.38rem 1.1rem;border-radius:5px;font-family:-apple-system,sans-serif;font-size:.82rem;cursor:pointer;color:#555}
.prob-toggle:hover{background:#f0eeea}
.prob-ans{margin-top:1rem;padding:1.2rem 1.4rem;background:#f4f2ec;border-radius:7px;font-size:.96rem;line-height:1.8;display:none}
.prob-ans.open{display:block}
.prob-ans p{margin-bottom:.8rem}
.prob-ans p:last-child{margin-bottom:0}
.kw{background:#fef3c7;padding:.05em .35em;border-radius:3px;font-weight:600}
.kw2{background:#d4f7d4;padding:.05em .35em;border-radius:3px;font-weight:600}
.kw3{background:#ffd6d6;padding:.05em .35em;border-radius:3px;font-weight:600}
.prob-formula{background:#e8e5de;padding:.65rem 1rem;border-radius:5px;margin:.6rem 0;font-size:.94rem;overflow-x:auto}
</style>
<script>
function tp(btn){var a=btn.nextElementSibling;var o=a.classList.toggle('open');btn.textContent=o?'답안 닫기 ▲':'모범 답안 보기 ▾';}
</script>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">문제 1</span><span class="prob-tag">VAE · ELBO · Jensen 부등식</span></div>
<div class="prob-q">VAE(Variational Autoencoder)의 목적 함수인 ELBO가 아래와 같이 분해됨을 Jensen 부등식을 이용하여 유도하라. 각 항의 정보이론적 의미를 설명하고, KL 항이 왜 잠재 공간(latent space)의 정규화 역할을 하는지 논하라.
$$\log p(x) \;\geq\; \underbrace{\mathbb{E}_{q(z|x)}[\log p(x|z)]}_{\text{재구성 항}} - \underbrace{D_{KL}(q(z|x)\,\|\,p(z))}_{\text{정규화 항}}$$</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>유도.</strong> 변분 분포 $q(z|x)$를 도입하여 로그 우도를 전개한다:</p>
<div class="prob-formula">$$\log p(x) = \log \int p(x,z)\,dz = \log \int q(z|x)\frac{p(x,z)}{q(z|x)}\,dz = \log\,\mathbb{E}_{q(z|x)}\!\left[\frac{p(x,z)}{q(z|x)}\right]$$</div>
<p>로그 함수의 오목성에 <span class="kw">Jensen 부등식</span> $\log \mathbb{E}[f] \geq \mathbb{E}[\log f]$을 적용하면:</p>
<div class="prob-formula">$$\log p(x) \;\geq\; \mathbb{E}_{q(z|x)}\!\left[\log\frac{p(x,z)}{q(z|x)}\right] = \mathbb{E}_{q}\!\left[\log p(x|z)\right] + \mathbb{E}_{q}\!\left[\log\frac{p(z)}{q(z|x)}\right]$$</div>
<p>두 번째 항이 $-D_{KL}(q(z|x)\|p(z))$이므로 ELBO 분해가 완성된다. 등호는 $q(z|x) = p(z|x)$, 즉 변분 사후 분포가 참 사후 분포와 일치할 때 성립한다.</p>

<p><strong>재구성 항의 의미.</strong> $\mathbb{E}_{q(z|x)}[\log p(x|z)]$는 인코더 $q(z|x)$가 샘플링한 잠재 코드 $z$로부터 디코더 $p(x|z)$가 원본 $x$를 얼마나 잘 복원하는지를 측정한다. 픽셀 단위 교차 엔트로피 혹은 MSE로 구현되며, <span class="kw">최대화 방향이 재구성 품질 향상</span>이다.</p>

<p><strong>KL 항의 정규화 역할.</strong> $D_{KL}(q(z|x)\|p(z))$를 최소화하는 것은 인코더가 생성하는 사후 분포 $q(z|x)$를 사전 분포 $p(z) = \mathcal{N}(0,I)$에 가깝게 만든다는 뜻이다. 이 제약이 없으면 인코더는 각 입력 $x$를 잠재 공간의 고립된 점으로 매핑하여 디코더의 학습을 쉽게 만들 수 있지만, <span class="kw3">잠재 공간이 연속적이지 않아 새로운 샘플 생성이 불가능</span>해진다. KL 항은 잠재 공간을 정규 분포 주변에 체계적으로 배치시켜 생성 모델로서의 기능을 보장한다. KL 항 앞에 $\beta > 1$ 계수를 붙인 $\beta$-VAE는 표현의 disentanglement를 강화한다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">문제 2</span><span class="prob-tag">Forward KL · Reverse KL · Mode Averaging vs Seeking</span></div>
<div class="prob-q">분포 $p$를 $q$로 근사할 때 $D_{KL}(p \| q)$ (Forward KL)를 최소화하는 것과 $D_{KL}(q \| p)$ (Reverse KL)를 최소화하는 것은 매우 다른 $q$를 유도한다. $p$가 다봉(multimodal) 분포일 때 각각의 경우 $q$가 어떻게 행동하는지 수식과 직관으로 설명하라. 이 차이가 VAE와 GAN의 학습 방식 차이와 어떻게 연결되는지 논하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>Forward KL: $\min_q D_{KL}(p \| q)$.</strong> 전개하면 $D_{KL}(p\|q) = \mathbb{E}_p[\log p] - \mathbb{E}_p[\log q]$이므로 최소화는 $\mathbb{E}_p[\log q(x)]$를 최대화, 즉 $p(x) > 0$인 모든 $x$에서 $q(x)$도 커야 한다. $p(x) > 0$인데 $q(x) \approx 0$이면 $\log q \to -\infty$이므로 무한한 패널티가 발생한다. 따라서 <span class="kw">$q$는 $p$의 support 전체를 커버</span>해야 하고, $p$가 두 개의 봉우리를 가지면 $q$는 두 봉우리 사이 어딘가를 포괄하는 "평균적" 분포가 된다. 이를 <span class="kw">zero-avoiding 또는 mean-seeking</span>이라 한다.</p>

<p><strong>Reverse KL: $\min_q D_{KL}(q \| p)$.</strong> $D_{KL}(q\|p) = \mathbb{E}_q[\log q] - \mathbb{E}_q[\log p]$이므로 $q(x) > 0$인 영역에서만 페널티가 발생한다. $p(x) \approx 0$인 곳에 $q$를 배치하면 $-\mathbb{E}_q[\log p]$가 커지므로, $q$는 자연스럽게 $p$의 <span class="kw">확률 질량이 높은 봉우리 하나에만 집중</span>하려 한다. $p$가 다봉 분포라면 $q$는 봉우리 하나를 선택하고 나머지를 무시한다. 이를 <span class="kw3">zero-forcing 또는 mode-seeking</span>이라 한다.</p>

<p><strong>VAE vs GAN 연결.</strong> VAE의 학습은 ELBO 최대화 = $D_{KL}(q(z|x)\|p(z))$ 최소화이므로 Reverse KL을 따르지만, 재구성 항에 의해 데이터 분포를 커버하려는 압력도 동시에 존재한다. 결과적으로 VAE는 <span class="kw2">평균적이고 흐릿한(blurry) 이미지</span>를 생성하는 경향이 있다 — mode-covering의 흔적이다. GAN의 판별자(discriminator)는 실제 분포 $p_\text{data}$와 생성 분포 $p_G$ 사이의 거리를 최소화하는데, 기본 GAN의 손실은 Jensen-Shannon Divergence, WGAN은 Wasserstein 거리와 연결된다. GAN 생성자는 판별자를 속일 수 있는 <span class="kw">하나의 그럴듯한 mode에 집중</span>하므로 mode-seeking 경향을 보이며, mode collapse 문제가 발생한다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">문제 3</span><span class="prob-tag">정보 병목 이론 · Mutual Information</span></div>
<div class="prob-q">Tishby et al.의 정보 병목(Information Bottleneck) 원리 $\min_{p(t|x)}\; I(X;T) - \beta\, I(T;Y)$를 설명하라. 이것이 딥러닝의 중간 표현(representation) 학습을 어떻게 설명하는지 논하고, 이 이론에 대한 비판적 관점(Saxe et al., 2019)도 함께 제시하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>정보 병목 원리.</strong> 입력 $X$, 레이블 $Y$, 중간 표현 $T$가 마르코프 체인 $Y - X - T$를 형성한다고 가정한다. 목표는 $T$가 <span class="kw">$Y$에 관한 정보 $I(T;Y)$는 최대한 보존</span>하면서, 동시에 <span class="kw">$X$의 불필요한 정보 $I(X;T)$는 최소한으로 압축</span>하는 표현을 찾는 것이다. $\beta$는 두 목표 사이의 trade-off를 조절한다. $\beta \to 0$이면 완전 압축(모든 정보 버림), $\beta \to \infty$이면 완전 보존이다.</p>

<p><strong>딥러닝 해석.</strong> Tishby & Schwartz-Ziv(2017)는 딥러닝 훈련을 두 단계로 해석했다: 초반의 <span class="kw">fitting 단계</span>(레이블 정보 $I(T;Y)$ 빠르게 증가)와 후반의 <span class="kw2">compression 단계</span>($I(X;T)$ 감소, 과제와 무관한 입력 정보 제거). 이 관점에서 딥러닝의 일반화는 표현이 충분 통계량(sufficient statistic)에 가까워지는 과정으로 이해된다. 또한 네트워크 깊이에 따라 정보 병목이 연속적으로 정제된다는 직관을 제공한다.</p>

<p><strong>비판적 관점.</strong> Saxe et al.(2019)은 두 가지 핵심 반론을 제시했다. 첫째, compression 단계는 <span class="kw3">비선형 활성화 함수(tanh 등)에서는 관찰되지만 ReLU에서는 나타나지 않는다</span> — 즉 활성화 함수에 따라 결론이 달라지므로 보편적 원리가 아니다. 둘째, 연속 분포에서의 mutual information 추정은 추정 방법에 따라 매우 달라지며, 관찰된 compression이 실제 표현 변화가 아닌 추정 편향일 수 있다. 따라서 정보 병목 이론은 딥러닝을 해석하는 하나의 관점은 제공하지만, 보편적 학습 원리로 받아들이기엔 실험적 지지가 부족하다는 것이 현재 학계의 합의에 가까운 평가다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">문제 4</span><span class="prob-tag">Rényi 엔트로피 · Shannon 엔트로피 극한</span></div>
<div class="prob-q">Rényi 엔트로피 $H_\alpha(X) = \frac{1}{1-\alpha}\log\sum_i p_i^\alpha$에서 $\alpha \to 1$일 때 Shannon 엔트로피로 수렴함을 L'Hôpital 법칙을 이용하여 증명하라. $\alpha = 2$ (Collision entropy)와 $\alpha \to \infty$ (Min-entropy)의 수식과 의미를 설명하고, 이 두 특수 케이스가 암호학에서 각각 어떤 역할을 하는지 서술하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>$\alpha \to 1$ 수렴 증명.</strong> $f(\alpha) = \log\sum_i p_i^\alpha$, $g(\alpha) = 1-\alpha$로 놓으면 $H_\alpha = f(\alpha)/g(\alpha)$이고, $\alpha = 1$에서 $f(1) = \log\sum_i p_i = \log 1 = 0$, $g(1) = 0$. L'Hôpital을 적용한다:</p>
<div class="prob-formula">$$\lim_{\alpha\to 1} H_\alpha = \lim_{\alpha\to 1} \frac{f'(\alpha)}{g'(\alpha)} = \frac{\dfrac{d}{d\alpha}\log\sum_i p_i^\alpha\Big|_{\alpha=1}}{-1}$$</div>
<p>$\frac{d}{d\alpha}\log\sum_i p_i^\alpha = \frac{\sum_i p_i^\alpha \log p_i}{\sum_i p_i^\alpha}$이고, $\alpha=1$에서 $= \sum_i p_i \log p_i$. 따라서:</p>
<div class="prob-formula">$$\lim_{\alpha\to 1} H_\alpha = -\sum_i p_i \log p_i = H(X)$$</div>

<p><strong>$\alpha = 2$: Collision Entropy.</strong> $H_2(X) = -\log\sum_i p_i^2 = -\log P(\text{collision})$. 두 독립 샘플이 같은 값을 가질 확률(충돌 확률)의 로그다. <span class="kw">암호학적 의미</span>: 공격자가 두 번의 시도로 같은 값을 맞출 확률과 직결된다. 해시 함수의 충돌 저항성(collision resistance)을 분석할 때 핵심 지표로, $H_2$가 크면 충돌을 찾기 어렵다.</p>

<p><strong>$\alpha \to \infty$: Min-Entropy.</strong> $H_\infty(X) = -\log\max_i p_i$. 가장 큰 확률을 가진 사건 하나만 고려한다. <span class="kw2">암호학적 의미</span>: 공격자가 최적 전략(가장 높은 확률의 값을 추측)으로 단 한 번에 맞출 확률을 정량화한다. 난수 생성기(RNG)의 품질, 특히 <span class="kw">추측 가능성(guessability)</span>을 평가하는 데 쓰인다. NIST의 난수 표준(SP 800-90B)에서 Min-entropy가 핵심 지표로 사용된다. Min-entropy가 클수록 어떤 전략으로도 단번에 맞추기 어렵다는 의미다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">문제 5</span><span class="prob-tag">Data Processing Inequality · 충분 통계량 · 대조 학습</span></div>
<div class="prob-q">Data Processing Inequality(DPI)는 어떤 확률적 처리 $f$를 거쳐도 상호 정보량이 감소함을 말한다: $I(X;Y) \geq I(f(X);Y)$. 이것이 딥러닝에서 충분 통계량(sufficient statistic)과 어떻게 연결되는지 설명하라. 또한 대조 학습(contrastive learning)의 InfoNCE 손실이 Mutual Information의 하한(lower bound)을 최대화하는 것임을 논하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>DPI와 충분 통계량.</strong> DPI에 따르면 어떤 함수 $f$를 적용해도 $I(f(X);Y) \leq I(X;Y)$이다. 등호가 성립하는 — 즉 정보 손실 없이 $Y$에 대한 모든 정보를 보존하는 — 함수 $f(X)$를 <span class="kw">충분 통계량(sufficient statistic)</span>이라 한다. 이 개념은 표현 학습의 궁극적 목표를 수학적으로 정의한다: 레이블 $Y$에 대한 정보를 전혀 잃지 않으면서 입력 $X$를 최대한 압축한 표현. 딥러닝의 중간 표현이 충분 통계량에 가까울수록 불필요한 정보(배경, 텍스처 등)를 제거하고 레이블과 관련된 구조만 포착한다. 최소 충분 통계량(minimal sufficient statistic)은 레이블 정보를 보존하는 가장 압축된 표현이며, 정보 병목 이론의 이상적 목표이기도 하다.</p>

<p><strong>InfoNCE와 Mutual Information 하한.</strong> InfoNCE 손실(Oord et al., 2018)은 대조 학습에서 positive pair $(x, y)$와 $N-1$개의 negative sample로 구성된다:</p>
<div class="prob-formula">$$\mathcal{L}_\text{InfoNCE} = -\mathbb{E}\!\left[\log\frac{e^{f(x)^T g(y)/\tau}}{\frac{1}{N}\sum_{j=1}^N e^{f(x)^T g(y_j)/\tau}}\right]$$</div>
<p>van den Oord et al.은 이 손실을 최소화하는 것이 $I(X;Y) \geq \log N - \mathcal{L}_\text{InfoNCE}$를 통해 Mutual Information의 <span class="kw2">하한을 최대화</span>하는 것과 동치임을 보였다. 직관적으로, 모델이 $N$개의 후보 중 올바른 positive pair를 잘 구분할수록 두 표현 사이에 공유된 정보(Mutual Information)가 크다는 뜻이다. 배치 크기 $N$이 클수록 하한이 타이트해지며, 대규모 배치가 대조 학습에서 성능을 향상시키는 이유가 여기 있다. DPI는 인코더를 거쳐도 Mutual Information이 감소하지 않도록 — 즉 <span class="kw">충분 통계량에 가까운 표현을 학습</span>하도록 — InfoNCE가 암묵적으로 압박한다고 해석할 수 있다.</p>

</div>
</div>

<div class="footnote">
  이전 포스트: <a href="information-theory.html">정보이론 기초</a> · 관련: <a href="mle-map-problems.html">MLE & MAP 심화 문제</a>
</div>
