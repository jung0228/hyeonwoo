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
.prob-ans{margin-top:0;padding:0 1.4rem;background:#f4f2ec;border-radius:7px;font-size:.96rem;line-height:1.8;visibility:hidden;max-height:0;overflow:hidden}
.prob-ans.open{visibility:visible;max-height:9999px;margin-top:1rem;padding:1.2rem 1.4rem}.prob-lv{font-family:-apple-system,sans-serif;font-size:.7rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#888;margin:2rem 0 1rem;padding-bottom:.4rem;border-bottom:1px solid #d8d8d2}
.prob-ans p{margin-bottom:.8rem}
.prob-ans p:last-child{margin-bottom:0}
.kw{background:#fef3c7;padding:.05em .35em;border-radius:3px;font-weight:600}
.kw2{background:#d4f7d4;padding:.05em .35em;border-radius:3px;font-weight:600}
.kw3{background:#ffd6d6;padding:.05em .35em;border-radius:3px;font-weight:600}
.prob-formula{background:#e8e5de;padding:.65rem 1rem;border-radius:5px;margin:.6rem 0;font-size:.94rem;overflow-x:auto}
</style>
<script>
function tp(btn){var a=btn.nextElementSibling;var o=a.classList.toggle('open');btn.textContent=o?'답안 닫기 ▲':'모범 답안 보기 ▾';if(o&&window.renderMathInElement){renderMathInElement(a,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}]});}}
</script>

<div class="prob-lv">학부 기초 문제 — 개념과 직관</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">기초 1</span><span class="prob-tag">엔트로피 · 불확실성</span></div>
<div class="prob-q">공정한 동전($p=0.5$)과 편향된 동전($p=0.9$)의 엔트로피를 각각 계산하라. 계산 결과를 바탕으로 "엔트로피는 불확실성의 척도"라는 말의 의미를 직관적으로 설명하고, 날씨 예보에서 "맑을 확률 99%"와 "맑을 확률 50%"의 엔트로피 차이가 일상적으로 어떤 의미를 갖는지 논하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>계산.</strong> $H = -p\log_2 p - (1-p)\log_2(1-p)$.</p>
<div class="prob-formula">$$H(0.5) = -0.5\log_2 0.5 - 0.5\log_2 0.5 = 1 \text{ bit}$$
$$H(0.9) = -0.9\log_2 0.9 - 0.1\log_2 0.1 \approx 0.469 \text{ bit}$$</div>
<p>공정한 동전의 엔트로피가 약 2.1배 크다.</p>

<p><strong>직관.</strong> 편향된 동전($p=0.9$)을 던지기 전에 우리는 이미 "앞면이 나올 것 같다"고 꽤 확신한다. 결과를 알아도 새로 얻는 <strong>정보의 양이 적다</strong>. 반면 공정한 동전은 완전히 예측 불가능 — 결과를 알았을 때 최대의 정보를 얻는다. 엔트로피는 "결과를 알기 전 평균적으로 얼마나 놀랄 것인가"를 측정한다.</p>

<p><strong>날씨 예보의 함의.</strong> "맑을 확률 99%"는 $H \approx 0.08$ bit — 내일 날씨에 대해 거의 확신이 있으므로 우산 챙길지 결정하기 쉽다. "50%"는 $H = 1$ bit — 완전히 불확실하므로 예보 자체가 의사결정에 도움이 안 된다. 기상 예보의 품질은 엔트로피를 얼마나 줄이는지(정보 이득)로 평가할 수 있다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">기초 2</span><span class="prob-tag">교차 엔트로피 · 손실 계산</span></div>
<div class="prob-q">이진 분류 모델이 두 가지 전략을 쓴다: (A) 항상 $[0.5, 0.5]$ 출력, (B) 정답을 맞추는 완벽한 모델. 레이블 분포가 균등(50/50)일 때 각 전략의 기대 교차 엔트로피 손실을 계산하고 비교하라. 전략 (B)에서 손실이 정확히 0이 아닌 이유를 설명하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>전략 (A) 손실.</strong> 정답이 클래스 0이든 1이든 $\hat{y}_\text{정답} = 0.5$. 기대 손실 = $-\log(0.5) = 1$ bit (log밑 2) 또는 $\ln 2 \approx 0.693$ (자연로그). 이것이 레이블의 엔트로피 $H(Y) = 1$ bit와 같다 — 우연이 아니다.</p>

<p><strong>전략 (B) 손실.</strong> 완벽한 모델이라면 정답 클래스에 확률 1을 부여 → $-\log(1) = 0$. 기대 손실 = $0$. 단, 실제 소프트맥스 모델에서 확률이 정확히 1.0에 도달할 수 없다(지수 함수의 성질상). 따라서 <strong>실전에서 교차 엔트로피 손실이 0인 경우는 이론적 극한</strong>이다.</p>

<p><strong>통찰.</strong> 전략 (A)의 손실 = 레이블 엔트로피 $H(Y)$. 전략 (B)의 손실 = 0. 교차 엔트로피 $H(p, q)$는 $H(p, q) = H(p) + D_{KL}(p\|q)$로 분해된다. 완벽한 모델이면 $q = p$이므로 $D_{KL} = 0$, 즉 $H(p,q) = H(p)$가 하한이다. 우리가 줄일 수 있는 것은 $D_{KL}$ 부분 뿐이고, 레이블 자체의 불확실성 $H(p)$는 줄일 수 없다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">기초 3</span><span class="prob-tag">KL Divergence · 비대칭성</span></div>
<div class="prob-q">분포 $P = [0.9, 0.1]$, $Q = [0.5, 0.5]$에 대해 $D_{KL}(P \| Q)$와 $D_{KL}(Q \| P)$를 각각 계산하라. 두 값이 다른 이유를 이 수치를 이용해 직관적으로 설명하고, "P를 Q로 근사할 때의 비효율성"이라는 해석과 연결하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>계산.</strong></p>
<div class="prob-formula">$$D_{KL}(P\|Q) = 0.9\log\frac{0.9}{0.5} + 0.1\log\frac{0.1}{0.5} \approx 0.9(0.848) + 0.1(-1.609) \approx 0.602 \text{ nats}$$
$$D_{KL}(Q\|P) = 0.5\log\frac{0.5}{0.9} + 0.5\log\frac{0.5}{0.1} \approx 0.5(-0.588) + 0.5(1.609) \approx 0.511 \text{ nats}$$</div>

<p><strong>비대칭성의 직관.</strong> $D_{KL}(P\|Q)$는 "실제 분포 $P$의 관점에서 $Q$를 쓸 때의 비효율". $P$는 클래스 0에 90% 확신이 있는데, $Q$는 50%밖에 안 준다 — 이 차이가 크다. 반면 $D_{KL}(Q\|P)$는 "균등 분포 $Q$의 관점에서 $P$를 쓸 때의 비효율". $Q$는 클래스 1에도 50%를 부여하는데, $P$는 10%만 준다 — 클래스 1에 대한 비용이 크다.</p>

<p><strong>"근사 비효율성" 해석.</strong> $D_{KL}(P\|Q) = 0.602$는 실제 분포가 $P$인데 $Q$ 기반 코드를 쓰면 샘플당 평균 0.602 nats의 추가 비트가 필요함을 의미한다. 방향이 바뀌면 비효율의 성격도 바뀐다. 이 비대칭성이 KL이 진정한 "거리"가 아닌 이유이고, VAE에서 어느 방향 KL을 쓰는지가 모델 행동에 큰 차이를 만드는 이유다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">기초 4</span><span class="prob-tag">정보 이득 · 결정 트리</span></div>
<div class="prob-q">결정 트리에서 "정보 이득(information gain)"은 어떤 특성으로 분기했을 때 레이블의 불확실성이 얼마나 줄어드는지를 측정한다. 이것을 조건부 엔트로피 $H(Y|X)$와 상호 정보량 $I(X;Y) = H(Y) - H(Y|X)$로 표현하고, 구체적인 수치 예시로 어떤 특성이 더 좋은 분기 기준인지 계산하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>개념 정의.</strong> 특성 $X$로 분기한 후 레이블 $Y$의 잔여 불확실성 = 조건부 엔트로피 $H(Y|X) = \sum_x p(x) H(Y|X=x)$. 정보 이득 = $I(X;Y) = H(Y) - H(Y|X)$. $I(X;Y)$가 클수록 $X$가 $Y$에 대한 정보를 많이 담고 있다.</p>

<p><strong>수치 예시.</strong> 10개 샘플, 레이블 50/50 (5개 양성 5개 음성). $H(Y) = 1$ bit. 특성 A: $X=0$이면 4양성/1음성, $X=1$이면 1양성/4음성. 각 5개씩.</p>
<div class="prob-formula">$$H(Y|A=0) = H(0.8, 0.2) \approx 0.722, \quad H(Y|A=1) = H(0.2, 0.8) \approx 0.722$$
$$I(A;Y) = 1 - 0.722 = 0.278 \text{ bit}$$</div>
<p>특성 B: $X=0$이면 5양성/0음성, $X=1$이면 0양성/5음성. $H(Y|B=0) = H(Y|B=1) = 0$. $I(B;Y) = 1 - 0 = 1$ bit. <strong>특성 B가 레이블을 완벽히 분리 → 정보 이득 최대</strong>. 결정 트리는 매 분기에서 $I(X;Y)$가 가장 큰 특성을 선택한다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">기초 5</span><span class="prob-tag">교차 엔트로피 분해 · 하한</span></div>
<div class="prob-q">교차 엔트로피가 $H(p, q) = H(p) + D_{KL}(p \| q)$로 분해됨을 보여라. 이 분해를 통해 (1) 교차 엔트로피의 최솟값이 엔트로피 $H(p)$임을 증명하고, (2) 딥러닝 모델을 학습할 때 우리가 실제로 줄이고 있는 것이 무엇인지 설명하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>분해 유도.</strong></p>
<div class="prob-formula">$$H(p,q) = -\sum_x p(x)\log q(x) = -\sum_x p(x)\log p(x) + \sum_x p(x)\log\frac{p(x)}{q(x)} = H(p) + D_{KL}(p\|q)$$</div>

<p><strong>(1) 최솟값 증명.</strong> $D_{KL}(p\|q) \geq 0$ (깁스 부등식, 등호는 $p = q$일 때). 따라서 $H(p, q) \geq H(p)$이고, 등호는 $q = p$일 때 성립. 즉 교차 엔트로피의 최솟값은 레이블 분포 $p$의 엔트로피 $H(p)$이다. ∎</p>

<p><strong>(2) 무엇을 줄이는가.</strong> 훈련 과정에서 레이블 분포 $p$ (데이터)는 고정되어 있고 $H(p)$는 변하지 않는다. 모델 $q_\theta$만 바뀐다. 따라서 교차 엔트로피 손실을 최소화하는 것은 정확히 $D_{KL}(p \| q_\theta)$를 최소화하는 것 — <strong>모델 분포를 데이터 분포에 가깝게 만드는 것</strong>이다. 이것이 딥러닝 학습이 정보이론적으로 하는 일이다.</p>

</div>
</div>

<div class="prob-lv">대학원 심화 문제 — 엄밀한 논증</div>

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
