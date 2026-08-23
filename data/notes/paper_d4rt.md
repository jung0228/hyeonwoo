# D4RT: Dynamic 4D Scene Reconstruction & Tracking

> **CVPR 2026 Best Paper Award** 🏆  
> **저자**: Google DeepMind, UCL, University of Oxford  
> **키워드**: `4D Dynamic Reconstruction`, `Feedforward Transformer`, `Query-Based Decoder`, `Spatiotemporal Tracking`

---

## 💡 핵심 아이디어

기존의 3D/4D 시공간 재구성 기법(COLMAP, NeRF, 4D Gaussian Splatting 등)은 깊이 추정, 광학 흐름(Optical Flow), 카메라 포즈 추정을 각각 독립된 별도의 파이프라인으로 수행하거나, 비디오마다 수십 분 이상의 Test-time Optimization(역전파 최적화)을 필요로 했습니다.

**D4RT (Dynamic 4D Reconstruction and Tracking)**는 모노큘러 비디오로부터 동적 4D 씬(3D 공간 + 시간축 변화)의 기하학적 구조, 카메라 포즈, 물체 모션을 **단 한 번의 Forward Pass (Single-Pass Feedforward Transformer)**로 300배 이상 빠르게 재구성하는 통합 아키텍처입니다.

---

## 🏗️ 아키텍처 및 메커니즘

1. **Global Self-Attention Encoder**:
   - 입력 비디오 프레임 전체 $V = \{I_1, I_2, \dots, I_T\}$를 글로벌 셀프 어텐션 렌즈로 교차 처리합니다.
   - 프레임 간 시공간 동역학(Temporal Evolution)과 변위 관계를 파악하여 공통 Latent **Global Scene Representation**을 형성합니다.

2. **On-Demand Query-Based Decoder**:
   - 모든 프레임의 모든 픽셀을 무겁게 렌더링하는 대신, 경량 쿼리 인터페이스를 도입했습니다.
   - 타겟 타임스텝 $t$, 카메라 좌표 $C$, 프레임 내 2D 좌표 $(u, v)$를 쿼리로 주면 해당 지점의 **3D 위치 $(X, Y, Z)$와 가시성(Occlusion Status) 및 모션 벡터**를 즉시 반환합니다.

$$\text{Query}(u, v, t) \longrightarrow \left( X(t), Y(t), Z(t), \text{Visibility}(t) \right)$$

3. **Single-Pass Inference**:
   - 테스트 타임 최적화(Optimization)나 다단계 융합 단계 없이, 단 한 번의 순전파만으로 시공간 기하 구조를 정밀 재구성합니다.

---

## ⚖️ Trade-off 및 한계

- **장점**: 기존 최적화 방식 대비 300배 빠른 속도, 가림(Occlusion) 상태에서도 물체 추적 지속성 완벽 유지.
- **한계**: 대규모 비디오 시퀀스 입력 시 Transformer의 메모리 복잡도 $O(N^2)$ 증가 부담.

---

## 🔗 연결 개념
- [[paper_virst]] (Spatiotemporal RVOS)
- [[paper_cosmos]] (World Models & Physical Simulation)
- [[rq_physical_world_model]] (현우의 연구 과제: 물리 세계 모델)
