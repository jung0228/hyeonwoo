# 논문 분석 리포트: LongVALE & Momentseeker → TCVP 적용 검토
**작성일**: 2026-04-04 | **대상**: TCVP 논문에 대한 참조 논문 분석 및 학습 전략 제시

---

## 📋 Executive Summary

### 세 논문의 위치 관계
- **TCVP** (준비 중): Video understanding baseline
- **LongVALE**: Omni-modal (V+A+S) long video understanding + 경량 dataset으로 SOTA 달성
- **Momentseeker**: Long Video Moment Retrieval (LVMR) benchmark + temporal grounding 능력 평가

### 핵심 학습 포인트
1. **멀티모달 통합의 효율성** (LongVALE): 적은 데이터로도 높은 성능 가능 → 데이터 큐레이션 전략 중요
2. **Temporal Reasoning의 어려움** (Momentseeker): 현재 MLLM들도 fine-grained temporal grounding에서 저조 → 이 부분이 경쟁력 포인트
3. **Foundation 모델 미세조정의 한계**: 대규모 모델도 특정 능력(temporal understanding)에서는 제한적

---

## 1️⃣ LongVALE 상세 분석

### 1.1 논문 개요
- **제목**: LongVALE: Vision-Audio-Language-Event Benchmark Towards Time-Aware Omni-Modal Perception of Long Videos
- **학회**: CVPR 2025
- **핵심**: 첫 omni-modal(V+A+S) long video benchmark + 경량 dataset으로 SOTA 달성

### 1.2 주요 실험 설정

#### 아키텍처 구성
```
Input Video (Long)
    ↓
Multi-Modal Encoders:
  - Vision: CLIP ViT-L/14 (100 frames, CLS token features)
  - Audio: BEATs (5.12s clips, variable tokens)
  - Speech: Whisper-Large-v2 (5.12s clips, variable tokens)
    ↓
Multi-Modal Adapters (Audio/Speech adapters randomly initialized)
    ↓
Token Concatenation (sequence dimension)
    ↓
LLM: Vicuna-7b (LoRA fine-tuning for audio/speech adapters)
```

#### 훈련 전략 (2 Stage)
**Stage 1: Boundary Perception Tuning**
- 목표: Omni-modal event boundaries 이해
- 데이터: Template-based dialogue (single-turn: dense captioning, multi-turn: grounding + segment captioning)
- 데이터량: 7,240 QA dialogues
- 비고: VTimeLLM의 visual-only data도 추가

**Stage 2: Instruction Tuning**
- 목표: Free-form dialogues로 다양한 reasoning 능력 강화
- 데이터: Gemini-1.5-Pro로 생성한 고품질 instruction data
- 데이터량: 25.4K QA dialogues (평균 3.6개/video)
- 특징: 시간 인지, 복합 추론 강조

### 1.3 핵심 실험 결과 및 해석

#### 표 1: SOTA 비교
| 지표 | 수치 | 의의 |
|------|------|------|
| Omni-TVG mIoU | 11.0 | VTimeLLM(6.4) 대비 71% 향상 |
| Omni-DVC CIDEr | 7.9 | 오디오/음성 정보 통합의 가치 증명 |
| Omni-SC CIDEr | 20.3 | Segment-level 이해도 우수 |

**핵심**: LongVALE-LLM(7B)이 32.7K audio-visual samples(10% 미만)로 AVicuna(7B)의 1.1M 샘플 능가

#### 표 2: Audio-Visual Correlation (AVC) Ablation
```
AVC 없음:
  - TVG mIoU: 23.7 → 25.6 (+7.4%)
  - DVC CIDEr: 3.5 → 7.3 (+108.6% 🔑)
  - SC CIDEr: 10.4 → 21.1 (+102.9% 🔑)
```
**중요성**: Captioning 성능에 critical (107% 이상 향상)

#### 표 3: Training Data Stage Ablation
| Stage | TVG mIoU | DVC CIDEr | SC CIDEr |
|-------|----------|-----------|----------|
| Baseline (V only) | 12.6 | 0.1 | 0.4 |
| + Boundary Perception | 25.6 | 7.3 | 21.1 |
| + Instruction Tuning | 26.0 | 7.8 | 25.1 |

**해석**: Instruction tuning이 특히 SC(segment captioning) 성능에 크리티컬

#### 표 4: Modality 조합 효과
```
V only:      12.6 / 2.2 / 14.6
V+A:         15.6 / 3.1 / 17.6 (+23.8% / +40.9% / +20.5%)
V+S:         15.2 / 2.9 / 17.3
V+A+S:       17.1 / 3.0 / 18.9 (+35.7% / +36.4% / +29.5%)
```
**패턴**: Audio가 Speech보다 더 효과적 (Speech는 transcription이라 redundancy?)

### 1.4 제로샷 성능 (AVSD + Music-AVQA)
- **AVSD**: 54.8 (AVicuna 53.1 대비 +1.7%)
- **Music-AVQA**: 49.4 (AVicuna 49.6 대비 -0.2%, 거의 동등)
- **의의**: 32.7K 샘플로 460K~350K 수준 성능 → 데이터 품질과 큐레이션의 중요성

### 1.5 방법론적 특징
1. **Template-based dialogue**: 구조화된 QA로 안정적인 학습
2. **Audio-Visual Correlation**: 단순 concatenation이 아닌 semantic correlation 학습
3. **Visual adapter freeze + Audio/Speech adapter train**: 계산 효율성
4. **LoRA fine-tuning**: Foundation model 최소 수정

---

## 2️⃣ Momentseeker 상세 분석

### 2.1 논문 개요
- **제목**: MomentSeeker: A Comprehensive Benchmark for Long Video Moment Retrieval
- **학회**: CVPR 2025
- **핵심**: Long-form video에서 query에 맞는 temporal interval 찾기 (Moment Retrieval)

### 2.2 Task 정의

#### Hierarchical Task Taxonomy
```
Global-Level (장거리 temporal reasoning)
├─ Causal Reasoning: "Why does...?" → 인과관계 파악
└─ Spatial Relation: "How many people opposite to...?" → 공간 이해

Event-Level (특정 event 로컬라이제이션)
├─ Description Location: 상세한 텍스트 설명 매칭
├─ Action Recognition: 특정 action 식별 및 분류
└─ Anomaly Detection: 비정상 행동 감지 (텍스트 큐 없이)

Object-Level (세밀한 시각 지각)
├─ Object Recognition: "What did you put...?"
├─ Object Localization: "Where was the scale?"
├─ Attribute Classification: "What color is...?"
└─ OCR-based Reasoning: 텍스트 읽기 + 분석
```

#### 데이터셋 특징
```
Dataset 비교:
- Video 길이: 평균 1201.9초 (≈ 20분) ✓ longest
- 도메인: 영화, 스포츠, 보안카메라, 유튜브, 애니메이션 (6개 도메인)
- 쿼리 유형: 텍스트(5), 텍스트+이미지(2), 텍스트+비디오(2) 분포 = 5:2:2
- 총 샘플: 268개 비디오, 1,800개 쿼리
```

### 2.3 평가 메트릭

#### Recall@1 (R@1)
- Top-1 prediction이 any ground-truth moment와 IoU > threshold 일치
- Standard metric (많은 moment retrieval 벤치마크에서 사용)

#### Mean Average Precision@5 (mAP@5)
- Top-5 predictions의 정확도 + ranking quality
- Multi-moment answers에서 더 정교한 평가
- Precision at each correct prediction 평균화

### 2.4 핵심 실험 결과

#### 표 5: 메인 결과 요약
**Generation-based methods** (End-to-end LLM):
```
GPT-4o (state-of-the-art):      R@1 = 18.2%, mAP@5 = 18.9%
Gemini 2.5 Pro (best):          R@1 = 29.6%, mAP@5 = 31.4% ← SOTA
Qwen2.5VL-72B:                  R@1 = 17.2%, mAP@5 = 16.9%
InternVL3-8B (lightweight):     R@1 = 5.9%,  mAP@5 = 6.1%
```

**Retrieval-based methods** (Chunking + similarity ranking):
```
InternVideo2-1B (best):         R@1 = 19.7%, mAP@5 = 26.6% ← Retrieval SOTA
LanguageBind-428M:             R@1 = 18.2%, mAP@5 = 25.4%
MM-Ret-148M:                   R@1 = 12.4%, mAP@5 = 17.7%
```

### 2.5 분석 결과

#### 🔴 Finding 1: LVMR는 매우 어려운 문제
- Gemini 2.5 Pro도 R@1 = 29.6% (2/3 실패)
- Counting goals 같은 discrete tasks에서 특히 저조
- **→ Temporal grounding의 난제가 여전함**

#### 🔴 Finding 2: Generation vs Retrieval 성능 격차
```
경량 Retriever (MM-Ret 148M) > 대형 Generator (InternVL3-8B)
12.4 R@1 > 5.9 R@1

하지만 모델 크기 증가 시 격차 감소:
InternVL3-38B (5.9 → 15.8 R@1) ← 거의 추격
```
**해석**: Generation 방식이 theoretically 우월하지만, 현재 MLLM의 temporal reasoning 능력 부족

#### 🔴 Finding 3: 멀티모달 쿼리 처리 약함
```
TMR (Text-only):     ✓ Best performance
IMR (Image+Text):    ✗ Significant degradation
VMR (Video+Text):    ✗ Worse than IMR
```
**특징**: 특히 generation methods에서 심각 → 교차모달 reasoning이 여전히 미흡

#### 🔴 Finding 4: Position Bias in Generation
```
InternVL3-8B:       ▯▯▁▁▁ (시작부분 bias)
Qwen2.5VL-7B:       ▮▁▁▁▁ (매우 강한 시작부분 bias)
Qwen2.5VL-72B:      ▁▁▁▁▁ (더 균형잡혀있음)
InternVL3-38B:      ▁▁▁▁▁ (position-insensitive)
```
**원인**: Context length 제약 → 이후 frames에 대한 정보 손실

#### 🔴 Finding 5: Context Length 제약
```
Qwen2.5VL-72B 효과:
- 768 frames 지원 ≈ 6.4분 (2fps 기준)
- Short videos (<8min)에서 retrieval 방식보다 우수
- 정보 손실 minimal
- Global-level tasks에서 full context 이용 가능

→ Longer context = better performance (일관된 패턴)
```

#### 🔴 Finding 6: Moment 품질이 downstream LVU 성능 예측
```
MomentSeeker 성능 ↔ LVU task 성능 (positive correlation)
- 우수한 retriever (InternVideo2) → 우수한 LVU 결과
- RAG pipeline에서 critical component
```

### 2.6 세부 분석: 비디오 길이와 모델 성능

#### Heatmap 분석 (Accuracy vs Duration)
```
Short videos (<10min):   Qwen2.5VL-72B > Retrieval methods
Medium videos (10-30min): Retrieval methods 우수
Long videos (>30min):     All methods 급격히 하락

- Retrieval: Larger candidate pool → ranking difficulty ↑
- Generation: Aggressive downsampling → information loss ↑
```

---

## 3️⃣ TCVP 적용 가능 기법 리스팅

### 3.1 데이터 큐레이션 전략

#### ✅ LongVALE에서 배운 점
| 기법 | 효과 | 적용 가능성 |
|------|------|-----------|
| **Boundary-Aware Training** | 시간 경계 명시적 학습 | ★★★★★ 높음 |
| **Audio-Visual Correlation** | Semantic-level 멀티모달 통합 | ★★★★ 높음 (TCVP가 V+A라면) |
| **Template-based QA** | 구조화된 학습 데이터 | ★★★★ 높음 |
| **Two-Stage Training** | Boundary Perception → Instruction | ★★★★ 높음 |
| **Visual-only Data 혼합** | 일반성 향상 | ★★★ 중간 |

#### ✅ Momentseeker에서 배운 점
| 기법 | 효과 | 적용 가능성 |
|------|------|-----------|
| **Hierarchical Task Design** | 세밀한 능력 평가 | ★★★★ 높음 |
| **Multi-modal Query Support** | TMR/IMR/VMR 동시 지원 | ★★★ 중간 (데이터 확보 필요) |
| **Retrieval + Generation 조합** | RAG-based approach | ★★★★ 높음 |
| **IoU Threshold 다층화** | 난이도 조절 | ★★★ 중간 |

### 3.2 모델 아키텍처 개선점

#### LongVALE 방식
```
강점:
✓ Minimal adapter (computational efficiency)
✓ Foundation model 최소 수정 (LoRA)
✓ Modality-specific encoding → shared LLM space

약점:
✗ Whisper speech redundancy (text-based, visual 정보 이미 내포)
✗ Fixed 100-frame sampling (long video에는 info loss)
```

#### Momentseeker 인사이트
```
Chunking-based retrieval의 강점:
✓ Linear temporal representation 유지
✓ Position bias 감소
✓ Efficient ranking

Generation-based의 잠재력:
✓ Longer context 제공 시 더 나은 성능
✓ Full video understanding 가능
✗ 현재 context length 제약 (768 frames까지도 제한적)
```

### 3.3 학습 전략 상향

#### Three-Stage 학습 제안
```
Stage 1: Foundation Model Adaptation
  └─ Audio/Speech encoders + Adapters 학습
  └─ 데이터: LongVALE 스타일 template-based 100K

Stage 2: Temporal Boundary Perception
  └─ Moment grounding (Momentseeker-style)
  └─ 데이터: 정확한 temporal annotation 10K

Stage 3: High-Quality Instruction Tuning
  └─ Diverse reasoning tasks
  └─ 데이터: LLM-generated instruction 50K+
```

---

## 4️⃣ TCVP 적용 시 주의점: Foundation 모델의 한계

### 4.1 Foundation 모델 미세조정의 특성

#### 경험적 관찰 (Momentseeker 분석)
```
모델 크기별 temporal reasoning 능력:
- 7B:    R@1 = 5-12% (매우 낮음)
- 13B:   R@1 = 7-15% (약간 높음)
- 38B:   R@1 = 15-20% (실용적)
- 72B:   R@1 = 17-30% (최고, 하지만 여전히 낮음)

→ 학습 데이터 & 모델 크기 모두 중요하지만,
  temporal understanding은 특히 어려운 능력
```

### 4.2 배치 사이즈의 중요성

#### LongVALE에서의 암시적 신호
```
Stage 1 (Boundary Perception):
  - 7,240개 QA → 어떤 배치 사이즈? (논문에 명시 없음)
  - Likely 16-32 (7B model 기준)

Stage 2 (Instruction Tuning):
  - 25.4K QA → larger batch 가능
  - Likely 32-64

예상 학습 곡선:
- 초기 (epoch 1-3): 급격한 성능 향상
- 중반 (epoch 3-5): 완화된 향상
- 후기 (epoch 5+): Plateau or overfitting risk
```

### 4.3 데이터 크기 vs 성능의 관계

#### LongVALE 증거
```
AVicuna (7B):
  - 1.1M audio-visual pairs
  - AVSD: 53.1, Music-AVQA: 49.6

LongVALE-LLM (7B):
  - 0.7M pairs (36% 감소)
  - AVSD: 54.8 (+1.7%), Music-AVQA: 49.4 (-0.2%)

결론: 데이터 양보다 품질 (curated audio-visual correlation)이 critical
```

#### Momentseeker 암시
```
Generation methods의 성능:
- 더 많은 frames (768 vs 96) → 약 2-3% R@1 향상
- 모델 크기 (8B vs 72B) → 약 10-12% R@1 향상

→ 단순 스케일링은 한계 (temporal reasoning은 구조적 문제)
```

### 4.4 최악의 경우: 학습이 효과 없을 수 있다

#### Foundation 모델의 "plateau" 현상
```
시나리오 1: 부족한 temporal annotation
- 모델이 이미 visual feature에 과적합
- Audio/temporal boundary 추가 학습 시 minimal improvement
- 오히려 원래 성능 저하 가능

시나리오 2: 불충분한 배치 사이즈
- Gradient variance 너무 높음
- Instability 유발
- 수렴 불가능

시나리오 3: 데이터의 noise
- 부정확한 temporal labels
- 멀티모달 간 alignment 부족
- 모델이 spurious correlations 학습
```

### 4.5 따라서 필요한 검증 구조

#### Pre-training 전 체크리스트
```
□ Temporal annotation quality audit
  └─ Inter-annotator agreement ≥ 80%?
  └─ Edge case (scene transition, silence) 처리?

□ 데이터 크기 적절성
  └─ 최소 5,000개 boundary-annotated samples?
  └─ 다양한 도메인 커버?

□ 배치 사이즈 최적화
  └─ Gradient accumulation vs batch size tradeoff 분석?
  └─ Learning rate warmup 충분?

□ 멀티모달 alignment 검증
  └─ Audio-visual correlation 수동 샘플 검토?
  └─ Timestamp 동기화 오차 ≤ 100ms?
```

---

## 5️⃣ TCVP 적용 권장 로드맵

### Phase 1: Foundation 분석 (1-2주)
```
1.1 TCVP 현재 상태 분석
    - 데이터셋 크기 및 annotation quality
    - 현재 모델 baseline 성능
    - Temporal reasoning 능력 평가

1.2 LongVALE 스타일 데이터 준비
    - Audio-visual correlation annotation
    - Template-based QA generation (GPT-4)
    - Quality check (inter-annotator agreement)

1.3 Momentseeker 스타일 평가 설계
    - Hierarchical task 설계 (Global/Event/Object)
    - 난이도별 subset 생성
```

### Phase 2: 초기 실험 (2-3주)
```
2.1 LongVALE-style Boundary Perception
    - 5,000개 annotated segments로 시작
    - BatchSize = 16, Learning Rate = 2e-4 (LoRA)
    - 3 epochs with validation every 500 steps

2.2 Ablation Study
    - Audio-only vs V+A vs V+A+Speech
    - Boundary perception alone vs + instruction tuning
    - 결과와 LongVALE와 비교 (scaling insights)

2.3 정량 평가
    - Recall@1@0.3 IoU, Recall@1@0.5 IoU
    - 기존 VTimeLLM 대비 %improve
```

### Phase 3: Moment Retrieval 추가 (3-4주)
```
3.1 Momentseeker 평가 적용
    - 상위 3개 task types로 시작 (Global + Event)
    - 300개 annotated moments
    - R@1 & mAP@5 측정

3.2 Retrieval vs Generation 비교
    - Chunking-based retrieval (10-sec chunks)
    - Direct generation (end-to-end)
    - 각각의 장단점 분석

3.3 의사결정
    - 어느 방향이 TCVP 특성상 적합한가?
    - 데이터 요구사항 vs available resources
```

### Phase 4: 최종 검증 및 배포 (2-3주)
```
4.1 최종 모델 학습
    - 전체 데이터로 3-stage training
    - Cross-validation으로 hyperparameter optimize
    - Test set 최종 평가

4.2 논문화
    - Experimental results 정리
    - Ablation studies visualization
    - Error analysis (where & why fail)

4.3 준하님 리뷰
    - 성능 개선이 유의미한가?
    - Foundation model 학습의 가치가 있는가?
```

---

## 6️⃣ 권장 사항 (Recommendations)

### 🎯 최우선 추천
1. **LongVALE의 AVC (Audio-Visual Correlation) 방식 도입**
   - 단순 concatenation이 아닌 semantic-level 통합
   - 데이터 품질 우선 (양 보다 질)
   - 기대 효과: 10-30% 성능 향상

2. **Temporal Boundary Annotation 우선 투자**
   - Momentseeker에서 본 바와 같이 temporal grounding은 어려운 문제
   - 정확한 annotation (±100ms 이내) 필수
   - 충분한 negative examples (false moments) 포함

3. **두 단계 학습 방식 도입**
   - Stage 1: Boundary perception (시간 기초)
   - Stage 2: Instruction tuning (다양한 reasoning)
   - 각 stage에서 성능 평가로 효과 검증

### ⚠️ 주의사항
1. **Foundation 모델의 한계 인식**
   - 학습한다고 성능이 반드시 향상되지 않음
   - Temporal reasoning은 특히 어려운 능력 (현재 SOTA도 R@1 ≈ 30%)
   - 성능 향상이 없다면, 모델 크기 증대 고려 (7B → 13B/30B)

2. **배치 사이즈의 중요성**
   - 너무 작은 배치 (<8): 불안정한 학습
   - 너무 큰 배치 (>64): gradient accumulation 고려
   - 권장: 16-32 (7B model), 32-64 (13B+ model)

3. **데이터 품질 관리**
   - 부정확한 temporal labels는 모델을 오도함
   - 멀티모달 간 시간 동기화 검증 필수
   - 이상치 감지 및 제거

### 🚀 추가 고려사항
- **Knowledge distillation**: Larger model (30B+)로부터 학습 후 7B로 distill
- **Curriculum learning**: 쉬운 frames부터 어려운 frames로 순차 학습
- **Ensemble**: 여러 모델 조합 (retrieval + generation)

---

## 7️⃣ 참고 수치

### LongVALE 핵심 수치
- 데이터: 105K omni-modal events
- 훈련 데이터: 32.7K audio-visual samples (≈ 7,240 + 25,400 QA)
- 모델: Vicuna-7b + LoRA
- 성능: AVSD 54.8 (AVicuna 53.1), Music-AVQA 49.4 (AVicuna 49.6)
- **Key Metric**: Audio-Visual Correlation → 100% 이상 성능 향상 (caption tasks)

### Momentseeker 핵심 수치
- 데이터: 268 videos, 1,800 queries, 1,201.9초 평균 길이
- SOTA (Gemini 2.5 Pro): R@1 = 29.6%, mAP@5 = 31.4%
- 7B models 평균: R@1 ≈ 5-12%
- **Key Insight**: Context length 제약 (768 frames ≈ 6.4분) = 성능 ceiling

---

## 📝 최종 결론

### 1. TCVP 적용의 가치
✅ **높음** - LongVALE의 두 단계 학습과 Momentseeker의 hierarchical evaluation은 TCVP 개선에 직접 적용 가능

### 2. 예상 성과
- **보수적**: 5-10% 성능 향상 (데이터 충분히 좋은 경우)
- **중간**: 15-25% 성능 향상 (LongVALE 방식 완전 도입)
- **낙관적**: 30-50% 성능 향상 (모델 크기 증대 + 완전한 재구현)

### 3. 성공의 핵심 요소
1. **데이터 품질** > 데이터 양
2. **정확한 Temporal annotation** 필수
3. **Foundation 모델의 한계** 인식
4. **배치 사이즈 최적화** 필요
5. **두 단계 학습** 반드시 구현

### 4. 실패 위험 요소
❌ 부정확한 temporal labels
❌ 불충분한 배치 사이즈
❌ 무작정 데이터 양 확대
❌ Foundation 모델의 한계 무시

---

**이 리포트는 준하님의 검토와 의견을 기반으로 다음 단계가 결정될 예정입니다.**
