# 🔧 TCVP에 적용 가능한 LongVALE/Momentseeker 실험들

**목표**: 당신의 TCVP를 더 강하게 만들기 위한 구체적 실험 제안

---

## 📋 Applicable Experiments by Source

### 1️⃣ LongVALE에서 즉시 적용 가능 (5가지)

#### ✅ 1-1. Two-Stage Training (Model 추가)

**LongVALE에서**: Boundary Perception → Instruction Tuning

**TCVP에 적용하는 방법**:
```
Stage 1: Moment Localization Tuning
  Input: Comment timestamp + moment boundary
  Task: Learn to predict temporal spans from comments
  Data: TCVP의 timestamped comments

Stage 2: Query Understanding Tuning
  Input: Query + modality label
  Task: Learn to match queries with correct modalities
  Data: TCVP의 comment-grounded queries

Result:
  └─ Model improvement 측정 가능
  └─ Ablation으로 각 stage의 기여도 분석
```

**예상 효과**:
```
Current: Embedding-based retrieval만 (LanguageBind, CLIP)
After:   MLLM fine-tuning으로 성능 향상 expected
```

#### ✅ 1-2. Audio-Visual Correlation (AVC) 명시적 분석

**LongVALE에서**: AVC reasoning으로 caption 100%+ 개선

**TCVP에 적용하는 방법**:
```
Current TCVP:
  Comment → Modality gating (binary: V or A)
           ↓
           Separate visual/audio captions
           ↓
           Generate query

Proposed Enhancement:
  Comment → Modality gating
           ↓
           For each comment, analyze correlation type:
             ├─ Complementary: "Speaker says X, visual shows X"
             ├─ Synchronous: "Music + dance happen together"
             ├─ Enhanced: "Speech + background noise atmosphere"
             ├─ Causality: "Action triggers reaction"
             └─ Corrective: "Audio contradicts visual"
           ↓
           Include correlation type in query

Result:
  └─ Richer dataset annotations
  └─ Models can learn V-A relationships
```

**구체적 실험**:
```
1. Manually annotate 500 samples with AVC type
2. Train model with/without AVC reasoning
3. Measure: Does AVC reasoning improve performance?

Expected: Similar to LongVALE (10-100% improvement)
```

#### ✅ 1-3. Ablation: Comment Filtering Threshold (τ)

**LongVALE에서**: Window size ablation ([-9s, +9s] 결정)

**TCVP에 적용하는 방법**:
```
Current: τ = 0.3 (fixed)

Systematic ablation:
  τ = 0.1 → 거의 모든 comment 유지 (noise 많음)
  τ = 0.2 → 더 많은 comment (기준보다 아래)
  τ = 0.3 → Current setting (balanced)
  τ = 0.4 → More filtering (stricter)
  τ = 0.5 → Very strict (high-quality only)

Measure for each τ:
  ├─ % comments retained
  ├─ Quality (human preference)
  ├─ Diversity (category distribution)
  └─ Model performance
```

**표 형태로 정리**:
```
| τ    | Retained | Quality | Model R@1 |
|------|----------|---------|-----------|
| 0.1  | 95%      | 45%     | 28%       |
| 0.2  | 92%      | 62%     | 32%       |
| 0.3  | 85.7%    | 70%     | 35%       |
| 0.4  | 70%      | 78%     | 34%       |
| 0.5  | 55%      | 82%     | 32%       |
```

#### ✅ 1-4. Ablation: Window Size ([-9s, +9s] optimality)

**LongVALE에서**: Window size 결정 과정 참고

**TCVP에 적용하는 방법**:
```
Current: Symmetric [-9s, +9s] window

Ablation options:
  [-3s, +15s]  (asymmetric, more after)
  [-9s, +3s]   (asymmetric, more before)
  [-9s, +9s]   (symmetric, current)
  [-12s, +12s] (larger)
  [-6s, +6s]   (smaller)

Measure:
  ├─ Moment coverage (% of what captions capture)
  ├─ Query relevance
  ├─ Computational cost
  └─ Model performance

Expected result:
  └─ [-9s, +9s] is indeed optimal (like LongVALE's finding)
```

#### ✅ 1-5. Zero-Shot Evaluation (Generalization)

**LongVALE에서**: AVSD, Music-AVQA에서 zero-shot 성공

**TCVP에 적용하는 방법**:
```
Current: TCVP dataset만 평가

Proposed: 다른 benchmark에서 zero-shot test
  ├─ QVHighlights (moment retrieval, visual-focused)
  ├─ TVR (TV-based, temporal reasoning)
  ├─ CharadesSTA (action recognition)
  └─ TALL (temporal action localization)

Setup:
  Train on: TCVP
  Test on: Other benchmarks (without re-training)

Measure:
  └─ Recall@1, Recall@5

Expected insight:
  └─ TCVP queries가 얼마나 generalizable한가?
```

---

### 2️⃣ Momentseeker에서 즉시 적용 가능 (6가지)

#### ✅ 2-1. Hierarchical Difficulty Evaluation

**Momentseeker에서**: Global/Event/Object 3-level difficulty

**TCVP에 적용하는 방법**:
```
By video category (already in TCVP):

Category 1: PODCASTS (Speech-heavy)
  Difficulty: Hard
  Reason: Audio-centric, requires understanding
  Expected performance: Lower R@1

Category 2: EDUCATION (Mixed V+A)
  Difficulty: Medium
  Reason: Balanced modalities

Category 3: SPORTS (Action-heavy)
  Difficulty: Easy
  Reason: Visual-obvious moments
  Expected performance: Higher R@1
```

**구체적 실험**:
```
Test each model on different categories:

              Podcast  Education  Sports
CLIP          12%      24%        38%
LanguageBind  18%      32%        45%
Qwen2.5-Omni  22%      35%        42%

Analysis:
  ├─ Which categories are harder?
  ├─ Do audio-centric comments need better models?
  └─ Visual dominance confirmed?
```

#### ✅ 2-2. Position Bias Analysis (Heatmap)

**Momentseeker에서**: Position bias in generation models

**TCVP에 적용하는 방법**:
```
Question: Do models prefer early/middle/late timestamps?

Analysis:
  Comment timestamp distribution:
    ├─ Normalize by video duration
    ├─ Create bins: [0-25%], [25-50%], [50-75%], [75-100%]
    └─ Measure model performance per bin

Visualization (heatmap style):
  X-axis: Video position (0% → 100%)
  Y-axis: Model
  Color: Accuracy (%)

Example:
  Qwen2.5-Omni: ▯▯▁▁▁ (start bias)
  LanguageBind: ▁▁▁▁▁ (balanced)

Insight:
  └─ Do comments at video start perform better?
     (If yes, suggests attention/context issues)
```

#### ✅ 2-3. Multi-Modal Query Extension (TMR/IMR/VMR)

**Momentseeker에서**: Text-only, Image-conditioned, Video-conditioned

**TCVP에 적용하는 방법**:
```
Current TCVP: Text-only comments

Enhancement:
  TMR (Text Moment Retrieval) - Current
    └─ "Why did he tear the bill?"

  IMR (Image-conditioned)
    └─ [Image of torn bill] + "What happens next?"
    └─ Could use comment's context image

  VMR (Video-conditioned)
    └─ [Video segment of setup] + "Find the punch line"
    └─ Could use surrounding comment context

Setup:
  From TCVP comments, extract:
    ├─ Key frame at timestamp
    ├─ Context video window
    └─ Modify query to be conditional

Measure:
  ├─ Text-only R@1
  ├─ Image+Text R@1
  ├─ Video+Text R@1

Expected: Difficulty TMR > IMR > VMR
```

#### ✅ 2-4. Generation vs Retrieval Comparison

**Momentseeker에서**: Two paradigms systematic comparison

**TCVP에 적용하는 방법**:
```
Current: Mostly embedding-based retrieval

Proposed: Both paradigms evaluated

Retrieval Setting:
  Video → 10-sec segments
        → Embed with CLIP/LanguageBind
        → Rank by query similarity
        → Return top-1 segment

Generation Setting:
  Video → Sample 100 frames
        → Feed to MLLM (Qwen2.5-Omni, etc.)
        → MLLM predicts timestamp directly

Comparison table:
             | Retrieval | Generation
─────────────┼───────────┼────────────
Visual R@1   | 30.5%     | 22%
Audio R@1    | 18%       | 14%
Mixed R@1    | 24%       | 18%

Analysis:
  ├─ Which is better for TCVP?
  ├─ Hybrid approach possible?
  └─ Why does retrieval dominate?
```

#### ✅ 2-5. Context Length Ablation

**Momentseeker에서**: Frame number impact (96 vs 768 frames)

**TCVP에 적용하는 방법**:
```
Question: How many frames does MLLM need for TCVP?

Ablation:
  30 frames  (15 sec sampling)
  60 frames  (5 sec sampling)
  100 frames (2 sec sampling) ← TCVP current
  200 frames (1 sec sampling)
  256 frames (0.5 sec sampling)

Measure for each:
  ├─ Model R@1 performance
  ├─ Inference time
  ├─ GPU memory
  └─ Diminishing returns point

Expected result:
  100-150 frames might be sweet spot
  (Beyond this: diminishing returns)
```

#### ✅ 2-6. Modality-Specific Performance Heatmap

**Momentseeker에서**: Sub-task performance visualization

**TCVP에 적용하는 방법**:
```
Current: Average performance per category

Proposed: Granular breakdown

For each model, create heatmap:

              Podcast  Sports  Comedy  Gaming  Education
CLIP          12%      38%     25%     18%     22%
LanguageBind  18%      45%     32%     24%     28%
Qwen-Omni     22%      42%     35%     28%     32%

Analysis:
  ├─ Which categories are model-specific hard?
  ├─ Do visual models fail on audio content?
  ├─ Is there a ranking: Sports > Others > Podcasts?
  └─ Category-specific improvements needed?
```

---

### 3️⃣ 복합 적용 (Both + Combined)

#### ✅ 3-1. Two-Stage Training + AVC Reasoning

**결합**: LongVALE + LongVALE

**TCVP에 적용**:
```
Stage 1: Boundary Perception with AVC
  ├─ Learn to identify moments from comments
  ├─ Explicitly model audio-visual relationships
  └─ Data: TCVP + AVC annotations

Stage 2: Instruction Tuning with Modality Awareness
  ├─ Learn to handle vision/audio queries separately
  ├─ Generate better predictions per modality
  └─ Data: Comment-grounded + modality labels

Result: Combined benefit of both
```

#### ✅ 3-2. Hierarchical Difficulty + Two-Stage Training

**결합**: Momentseeker + LongVALE

**TCVP에 적용**:
```
Stage 1: Easy category training (Sports, Comedy)
  └─ Fast convergence, quick validation

Stage 2: Hard category training (Podcasts)
  └─ Curriculum learning approach

Measure:
  ├─ Stage 1-only performance
  ├─ Stage 1+2 final performance
  └─ Curriculum benefit quantification
```

#### ✅ 3-3. Zero-Shot + Hierarchical Evaluation

**결합**: LongVALE + Momentseeker

**TCVP에 적용**:
```
Train on: TCVP (hierarchical evaluation by category)

Test zero-shot on:
  ├─ QVHighlights (Sports-heavy, easy)
  ├─ TVR (TV-based, medium)
  └─ CharadesSTA (Action-heavy, medium)

Analysis:
  Do easy categories from TCVP transfer to easy categories in benchmarks?

Example:
  TCVP Sports (45%) → QVHighlights (40%) ✓ Transfer works
  TCVP Podcasts (22%) → TVR (25%) ✓ Transfer possible
```

---

## 🎯 Implementation Priority Ranking

### Tier 1: 반드시 추가해야 할 실험 (3개)

```
1. ✅ 1-3. Comment Filtering Threshold Ablation (τ)
   ├─ Why: τ=0.3 왜 선택했는지 정당화
   ├─ Effort: Low (table 추가)
   └─ Impact: High (methodological rigor)

2. ✅ 2-1. Hierarchical Difficulty by Category
   ├─ Why: TCVP의 카테고리별 난이도 분석
   ├─ Effort: Low (그룹핑만 하면 됨)
   └─ Impact: High (insights 제공)

3. ✅ 1-1. Two-Stage Training (Model)
   ├─ Why: Current는 dataset만, model contribution 필요
   ├─ Effort: Medium (model 구현)
   └─ Impact: Very High (결과 개선)
```

### Tier 2: 강력한 추가 분석 (3개)

```
4. ✅ 1-2. Audio-Visual Correlation Analysis
   ├─ Why: TCVP의 modality gating을 더 깊이 있게
   ├─ Effort: Medium (500 annotation)
   └─ Impact: High (differentiation from others)

5. ✅ 2-2. Position Bias Heatmap
   ├─ Why: 모델의 시간 위치 편향 파악
   ├─ Effort: Low (visualization)
   └─ Impact: Medium (diagnostic)

6. ✅ 1-5. Zero-Shot Evaluation
   ├─ Why: TCVP의 generalization 증명
   ├─ Effort: Medium (다른 dataset 평가)
   └─ Impact: High (broader validation)
```

### Tier 3: 선택 사항 (향후 작업)

```
7. ✅ 2-3. Multi-Modal Query Extension
   ├─ Effort: High (dataset 확장 필요)
   └─ Can be future work

8. ✅ 2-4. Generation vs Retrieval Comparison
   ├─ Effort: Medium
   └─ Good for discussion

9. ✅ 2-5. Context Length Ablation
   ├─ Effort: Low-Medium
   └─ Nice to have
```

---

## 📊 실행 로드맵

### Phase 1: 즉시 (Tier 1 - 3가지)
```
Week 1:
  └─ τ ablation 실험 실행 → 표 추가

Week 1-2:
  └─ Category별 성능 분석 → heatmap 추가

Week 2-3:
  └─ Two-stage model training 시작
```

### Phase 2: 2주차 (Tier 2 - 3가지)
```
Week 3:
  └─ AVC 분석 (500 sample annotation)

Week 3-4:
  └─ Position bias heatmap

Week 4:
  └─ Zero-shot evaluation (QVHighlights, TVR)
```

### Phase 3: 최종 정리
```
Week 4+:
  └─ 모든 실험 결과를 논문에 통합
  └─ Ablation section 확대
  └─ Analysis section 강화
```

---

## 📈 예상 효과

### Before (현재 TCVP)
```
Strengths:
  ✓ Novel idea (YouTube comments)
  ✓ Real user intent (70% preference)
  ✓ Audio-visual balance (45.9%)

Weaknesses:
  ✗ Dataset only (no model)
  ✗ Limited ablations
  ✗ No zero-shot validation
```

### After (적용 후)
```
Strengths:
  ✓ Novel idea
  ✓ Real user intent
  ✓ Audio-visual balance
  ✓ Strong model experiments (2-stage training)
  ✓ Comprehensive ablations (τ, category, window)
  ✓ Zero-shot generalization proven
  ✓ Deep analysis (position bias, AVC)

Result: Much stronger paper! 🚀
```

---

**이 실험들은 모두 당신의 TCVP 논문의 핵심 강점을 더욱 부각시킵니다!**
