---
# VAE (Variational Autoencoder)

카테고리: Generative Models  
자신감: ⭐⭐⭐ (중급)  
마지막 복습: 2026-08-10

---

## 한 문장 요약

VAE는 데이터를 확률적 잠재 공간(latent space)에 인코딩하고, 그 분포에서 샘플링해서 새 데이터를 생성하는 모델이다.

---

## 핵심 아이디어

일반 Autoencoder는 입력 → 하나의 벡터로 압축. VAE는 입력 → 분포(μ, σ)로 압축.

$$q_\phi(z|x) \approx p_\theta(z|x)$$

### ELBO (Evidence Lower Bound)

$$\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))$$

- 첫 번째 항: Reconstruction Loss — 원본을 잘 복원해야 함
- 두 번째 항: KL Divergence — latent가 표준정규분포에 가까워야 함

### Reparameterization Trick

직접 $z \sim q(z|x)$를 샘플링하면 역전파 불가.  
대신 $z = \mu + \sigma \cdot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$로 분리.

---

## Diffusion과의 연결

> "VAE랑 Diffusion이 근본적으로 같냐?" — 면접 질문

둘 다 ELBO를 최대화하는 프레임워크:
- VAE: 단일 step latent (한 번에 인코딩/디코딩)
- Diffusion: T step에 걸친 점진적 noise 추가/제거가 각 step의 VAE

Diffusion의 ELBO:
$$\mathcal{L} = \mathbb{E}\left[\sum_t D_{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))\right]$$

→ 각 step이 조건부 VAE처럼 동작. 근본 목적함수 구조는 동일!

---

## VQ-VAE (Discrete Token과의 연결)

VAE의 연속 latent space → 코드북의 이산 인덱스로 양자화  
→ 이미지/음성의 discrete token 표현 가능 (HCX Omni 8B에서 활용)

---

## 체크리스트

- [x] ELBO 유도할 수 있다
- [x] Reparameterization trick 설명할 수 있다
- [x] Diffusion과의 관계 설명할 수 있다
- [ ] VAE의 posterior collapse 문제 설명할 수 있다
- [ ] β-VAE 설명할 수 있다
