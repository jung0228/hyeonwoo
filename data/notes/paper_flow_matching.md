# 📄 [Theory/Paper] Flow Matching & Consistency Models in Omnimodal Generation
- **Key References**: Lipman et al. (ICLR 2023), Song et al. (OpenAI 2023), AlphaFlow / MotionPCM (2025-2026)
- **Domain**: Generative Theory / Fast Trajectory Sampling / Consistency Distillation
- **Connected**: [[diffusion]], [[paper_dynin_omni]], [[paper_cosmos]], [[rq_physical_world_model]]

---

## 1. Problem Formulation & Theoretical Bottleneck (왜 Flow Matching인가?)
- **확산 모델(Diffusion)의 근본적 병목**:
  - 확률 미분 방정식(SDE) 기반 곡선 궤적(Curved Trajectory)을 따르기 때문에, 깨끗한 샘플을 얻기 위해 수십 번($N \ge 20 \sim 50$)의 순차적 신경망 평가(NFE)가 불가피함 $\to$ 실시간 로봇 제어(50Hz) 및 초저지연 음성(160ms)에 치명적.
- **Flow Matching의 수학적 돌파구**:
  - 데이터 분포 $p_0$와 노이즈 분포 $p_1$ 사이를 **최단 직선 궤적(Straight Paths)**으로 잇는 최적 수송(Optimal Transport) 벡터장을 학습:
  $$v_t(\mathbf{x}) = \frac{d\mathbf{x}_t}{dt}, \quad \mathbf{x}_t = (1 - t)\mathbf{x}_0 + t\mathbf{x}_1$$
  - 선형 궤적으로 인해 $1 \sim 4$번의 스텝(Euler Step)만으로 고품질 비디오/음성 샘플링 가능.

---

## 2. 1-Step Consistency Distillation
- 궤적 상의 임의의 시점 $\mathbf{x}_t$를 원점 $\mathbf{x}_0$로 직접 투영하는 자기일관성 함수(Self-Consistency Function) $f_\theta(\mathbf{x}_t, t) = \mathbf{x}_0$ 학습:
  $$d(f_\theta(\mathbf{x}_{t_{n+1}}, t_{n+1}), f_\theta(\mathbf{x}_{t_n}, t_n)) = 0$$

---

## 3. Omnimodal Generation에서의 미해결 과제
- ⚠️ **이산-연속 하이브리드 토큰의 궤적 충돌**: 텍스트의 이산 토큰 공간과 비디오/로봇 액션의 연속 벡터 공간 사이에서 최적 수송 직선 궤적이 왜곡되는 현상 잔존.
- **후속 연구 기회**: Dynin-Omni의 공유 토큰 공간에 Flow Matching을 접목하여 **1-Step Real-Time Omnimodal Sampler** 설계 $\rightarrow$ [[rq_physical_world_model]].
