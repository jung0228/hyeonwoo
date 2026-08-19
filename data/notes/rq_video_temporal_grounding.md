# 🔭 [RQ] Streaming Long-Video Temporal Grounding & Memory
- **Researcher**: 정현우 (Jeong Hyeonwoo)
- **Domain**: Multimodal / Long-Video Understanding / Systems for AI
- **Research Background**: KAIST DAVIAN 랩 Video Moment Retrieval 평가 파이프라인 주도 경험 연계
- **Connected**: [[long_video_understanding]], [[streamkv]], [[paper_momentseeker]], [[kv_cache]]

---

## 1. Macro Why (거시적 당위성: 왜 이 문제를 풀어야 하는가?)
1분짜리 숏폼 영상도 VRAM이 터져 프레임을 버려야 하는 현재의 구조적 한계를 극복하지 못하면, 실시간 수술 로봇, 24시간 자율주행 모니터링, 장시간 회의 요약 에이전트는 결코 실현될 수 없습니다. 초장거리 스트리밍 비디오에서 메모리를 상수 $O(1)$로 묶어두면서도 1초 미만의 결정적 순간(Moment)을 100% 포착하는 시간적 인지 구조가 필수적입니다.

---

## 2. Prior Art Pathology & Frontier Blind Spot (기존 SOTA의 결함)
- **Uniform Sampling의 태생적 한계**: 1시간짜리 영상을 32프레임으로 균등 추출하면, 2초짜리 결정적 단서는 99% 확률로 샘플링 단계에서 영구 삭제됨.
- **StreamKV 등 기존 KV 압축의 맹점**: 누적 어텐션 점수가 낮은 토큰을 즉시 Evict하므로, 발생 당시에는 어텐션이 낮았으나 30분 뒤 중요한 복선으로 작용하는 Transient Visual Clue를 보존하지 못함.

---

## 3. Hyeonwoo's Core Hypothesis & 4-Vector Strategy (핵심 가설 및 4대 발굴 벡터)
- **발굴 벡터**: **[실패 모드 군집화 (Failure Cluster Synthesis)]** + **[병목 역전 (Bottleneck Targeting)]**
- **핵심 가설**:
  1. *실패 모드 해결*: Momentseeker 오답 군집의 80%가 '단일 프레임 모션 스파이크' 구간에서 발생한다는 분석에 기반하여, 시각 모션 엔트로피(Motion Entropy)가 급변하는 프레임을 동적으로 감지하는 **Event-Driven Adaptive Frame Sampler**를 전단에 배치.
  2. *병목 역전*: KV Cache 퇴출 시 단순 Top-K 어텐션 점수뿐만 아니라 시간적 앵커 스코어(Temporal Anchor Metric)를 반영하는 2계층 메모리 계층(SRAM L1 Recent Buffer + HBM Compressed Anchor Memory) 구축.

---

## 4. Evaluation & Verification Plan (검증 파이프라인)
- **Benchmarks**: Momentseeker, LongVideoBench, LVBench, Charades-STA.
- **Success Metric**: Momentseeker R@1 (IoU=0.5) 15%p 향상, GPU Memory 소비량 60% 절감 달성.
