# Diffusion Model

카테고리: Generative Models  
자신감: ⭐⭐ (기초)  
마지막 복습: 2026-08-10


## 한 문장 요약

데이터에 노이즈를 점진적으로 추가하는 forward process와, 노이즈에서 데이터를 복원하는 reverse process를 학습하는 생성 모델.


## 핵심 아이디어

### Forward Process (고정, 학습 없음)

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

T번 반복하면 $x_T \approx \mathcal{N}(0, I)$ (순수 가우시안 노이즈)

핵심 트릭: $x_t$를 $x_0$에서 직접 샘플링 가능

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### Reverse Process (학습)

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta)$$

모델이 각 step에서 추가된 노이즈 $\epsilon$을 예측:

$$\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t}\left[\Vert\epsilon - \epsilon_\theta(x_t, t)\Vert^2\right]$$


## VAE와의 연결

> "VAE랑 Diffusion이 근본적으로 같냐?"

| | VAE | Diffusion |
|---|---|---|
| Encoder | $q_\phi(z\Vertx)$ | $q(x_{1:T}\Vertx_0)$ (고정) |
| Decoder | $p_\theta(x\Vertz)$ | $p_\theta(x_{0:T})$ |
| ELBO | 단일 KL | T step KL의 합 |
| Latent | 연속, 저차원 | 데이터와 동일 차원 |

결론: 둘 다 ELBO 최대화. Diffusion은 "T개의 VAE를 순서대로 쌓은 것"으로 볼 수 있음.


## GAN과의 비교

| | GAN | Diffusion |
|---|---|---|
| 학습 방식 | Adversarial (불안정) | MSE (안정적) |
| 다양성 | Mode collapse 위험 | 높은 다양성 |
| 속도 | 빠름 | 느림 (T step) |
| 품질 | 선명 | 고품질 |


## 주요 변형

- DDPM: 기본 형태
- DDIM: 결정론적 샘플링으로 속도 개선
- Score-based / SDE: 연속 시간 관점
- Latent Diffusion (SD): pixel space → latent space에서 diffuse


## 체크리스트

- [x] Forward process 수식 이해
- [x] Reparameterization으로 $x_t$ 직접 샘플링
- [x] ELBO와 VAE 연결 설명
- [ ] Score matching과의 연결
- [ ] CFG(Classifier-Free Guidance) 설명
- [ ] DDIM 샘플링 과정 설명
