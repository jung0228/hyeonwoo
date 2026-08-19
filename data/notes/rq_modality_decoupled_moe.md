# 🔭 [RQ] Modality-Decoupled Sparse MoE Routing for Continual Omni-Learning
- **Researcher**: 정현우 (Jeong Hyeonwoo)
- **Domain**: Multimodal Architecture / Continual Learning / Sparse MoE
- **Core Challenge**: 새 로봇 액션이나 비디오 도메인을 학습할 때 기존 텍스트 수학 추론 및 음성 억양이 파괴되는 비대칭 망각(Catastrophic Forgetting) 해결
- **Connected**: [[hcx_omni]], [[paper_dynin_omni]], [[transformer]], [[lora]], [[rq_physical_world_model]]

---

## 1. Macro Why (왜 이 문제가 결정적인가?)
인간은 새로운 운동(테니스)이나 로봇 조작을 배운다고 해서 모국어 문법이나 수학적 계산 능력을 잃어버리지 않습니다. 하지만 현재의 옴니 파운데이션 모델은 단일 공유 파라미터 백본을 사용하기 때문에, 새로운 모달리티(로봇 액션 $\mathbf{a}_t$, 고해상도 비디오)를 추가 파인튜닝하는 순간 기존 모달리티의 그래디언트와 정면 충돌(Gradient Interference)하여 텍스트 추론 능력이 급락합니다.

---

## 2. Mathematical Pathology (수학적 원인: 그래디언트 내적의 음수화)
모달리티 $A$(텍스트)와 모달리티 $B$(로봇 모터 액션)의 그래디언트 벡터 $\mathbf{g}_A, \mathbf{g}_B$가 파라미터 공간에서 직교하지 않고 반대 방향을 가리킬 때:

$$\mathbf{g}_A \cdot \mathbf{g}_B < 0 \quad (\text{Gradient Conflict \& Destructive Interference})$$

---

## 3. Hyeonwoo's Solution: Modality-Decoupled Sparse MoE & Orthogonal Subspace
1. **모달리티 전용 전문가 라우터 (Modality-Specific Router)**:
   - 입력 토큰의 모달리티 타입에 따라 공통 추론 전문가(Shared Reasoning Experts)와 모달리티 특화 전문가(Action/Audio/Vision Dedicated Experts)로 조건부 활성화.
   $$\mathbf{y} = \sum_{i \in \text{TopK}} G(\mathbf{x})_i E_i(\mathbf{x}) + \mathbf{E}_{\text{shared}}(\mathbf{x})$$
2. **직교 그래디언트 투영 (Orthogonal Gradient Projection)**:
   - 로봇 액션 그래디언트 $\mathbf{g}_{\text{action}}$을 기존 텍스트 그래디언트 $\mathbf{g}_{\text{text}}$의 직교 여공간(Null Space)으로 투영하여 텍스트 지식 보존율 100% 달성:
   $$\tilde{\mathbf{g}}_{\text{action}} = \mathbf{g}_{\text{action}} - \frac{\mathbf{g}_{\text{action}} \cdot \mathbf{g}_{\text{text}}}{\|\mathbf{g}_{\text{text}}\|^2} \mathbf{g}_{\text{text}}$$
