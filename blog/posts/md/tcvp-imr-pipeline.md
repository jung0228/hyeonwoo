---
title: "TCVP-IMR: 타임스탬프 댓글로 IMR 학습 데이터 자동 구축하기"
dek: MomentSeeker가 수동으로 만든 것을 YouTube 댓글로 자동화할 수 있을까.
tags: ["Video", "Multimodal", "LLM"]
date: "2026-03-15"
readtime: 10 min read
slug: tcvp-imr-pipeline
katex: false
---

## 배경: IMR이라는 과제

기존 VMR(Video Moment Retrieval)은 텍스트 쿼리 하나로 영상 속 특정 구간을 찾는 문제다. 그런데 실제 유저의 검색 의도는 텍스트만으로 표현하기 어려울 때가 많다. "이 장면이랑 비슷한 부분 찾아줘"처럼, **이미지와 텍스트를 함께** 써야 자연스러운 경우다.

**IMR(Image-conditioned Moment Retrieval)**은 이 공백을 채운다:

<div class="callout">
  <strong>IMR 정의</strong><br>
  Input: 텍스트 쿼리 q<sub>T</sub> + 참조 이미지 q<sub>I</sub> + 영상 V<br>
  Output: 영상 내 타임스탬프 구간 [t<sub>start</sub>, t<sub>end</sub>]
</div>

MomentSeeker(2025)는 장시간 영상(평균 1,202초)을 대상으로 TMR / IMR / VMR 세 가지 쿼리 유형을 포괄하는 벤치마크를 발표했다. 이 중 IMR 쿼리는 약 360개. 핵심은 이것이 **전부 수동 어노테이션**이라는 점이다.

<div class="ornament">· · ·</div>

## MomentSeeker가 IMR 데이터를 만든 방식

구축 과정은 간단하지만 손이 많이 간다.

1. **q_I 선택:** 어노테이터가 참조 이미지를 직접 고른다. 같은 영상에서 "핵심 장면의 keyframe"을 뽑거나, 다른 영상에서 의미적으로 관련된 프레임을 가져온다.
2. **q_T 작성:** 선택한 이미지에 "묶인" 질문을 자연어로 직접 쓴다. 예: *"What's the color of the dog that once appeared on this road?"*
3. **타임스탬프 마킹:** 대상 영상에서 그 질문의 답이 되는 구간을 직접 표시한다.
4. **논리 일관성 검증:** (q_I, q_T, 정답 구간) 세 요소가 서로 맞는지 확인한다.

결과: 정교하지만 스케일이 안 된다. 360개가 한계다. 그리고 논문 자체도 말한다.

<div class="pullquote">
  "No model was specifically designed or fine-tuned for IMR — better training data specifically designed for IMR is needed."
</div>

현재 MomentSeeker IMR 최고 성능은 Gemini-2.5-Pro(29.6 R@1). 전용 학습 데이터가 없는 상태에서 retrieval 모델인 LanguageBind는 18.2 R@1에 그친다. **IMR 전용 학습 데이터가 존재하지 않는다는 게 병목이다.**

<div class="ornament">· · ·</div>

## TCVP가 가진 것

TCVP(Timestamped Comment-guided VMR Pipeline)는 YouTube의 타임스탬프 댓글을 활용해 VMR 학습 데이터를 구축한다. 유저가 남긴 댓글은 이런 형태다:

```
Video V
├── "2:15 - circuit board 처음 등장"         @ t=135s
├── "8:30 - circuit board 완성 후 테스트"    @ t=510s
└── "12:00 - 최종 동작 확인"                @ t=720s
```

각 (댓글 텍스트, 타임스탬프)는 영상 내 특정 순간을 **실제 유저가 가리킨 것**이다. keyframe도 추출할 수 있고, 텍스트도 있다.

여기서 핵심을 발견할 수 있다. 같은 영상 안에서 **같은 대상(circuit board)이 서로 다른 타임스탬프에 등장한다.** 이 두 타임스탬프를 연결하면 자연스럽게 IMR 쌍이 만들어진다.

<div class="callout">
  <strong>핵심 아이디어</strong><br>
  t<sub>i</sub>의 keyframe = q<sub>I</sub><br>
  t<sub>j</sub>의 댓글 텍스트 → LLM으로 질문 변환 = q<sub>T</sub><br>
  [t<sub>j</sub> - δ, t<sub>j</sub> + δ] = ground truth<br><br>
  사람이 하나도 개입하지 않는다.
</div>

<div class="ornament">· · ·</div>

## 자동 구축 파이프라인

### Step 1: keyframe 추출 + CLIP 임베딩

각 타임스탬프 댓글에 대해 해당 시점의 keyframe을 추출하고 CLIP visual embedding을 계산한다.

```python
for video V:
    for (comment C_i, timestamp t_i) in V.comments:
        K_i = extract_keyframe(V, t_i)
        e_i = CLIP.encode_image(K_i)
```

### Step 2: 시각적 유사 쌍 탐색

같은 영상 내에서 "관련 있지만 동일하지 않은" 프레임 쌍을 찾는다. 유사도 범위가 핵심이다.

```python
for all pairs (i, j) in same video V:
    temporal_gap = |t_i - t_j|
    visual_sim   = cosine_similarity(e_i, e_j)

    if temporal_gap > 30s and 0.55 < visual_sim < 0.82:
        valid_pairs.add((i, j))  # i → j 방향
```

| visual_sim | 의미 | 판단 |
|---|---|---|
| > 0.85 | 거의 동일한 프레임 | 탈락: 너무 쉬움 |
| 0.55 ~ 0.82 | 같은 대상, 다른 상황 | **채택** |
| < 0.50 | 관련 없는 장면 | 탈락: 시각 힌트 무의미 |

### Step 3: q_T 자동 생성 (LLM)

댓글 텍스트를 자연어 질문으로 변환한다. q_I(참조 이미지)를 함께 제공해 질문이 이미지와 "묶이도록" 유도한다.

```
Prompt:
  아래 댓글은 영상의 특정 순간을 설명합니다.
  첨부된 이미지(q_I)를 참조 시각 힌트로 활용하여,
  영상에서 댓글이 묘사하는 순간을 묻는 자연어 질문을 작성하세요.

  댓글: "8:30 - circuit board 완성 후 테스트 장면"

Output q_T:
  "When does the person test the circuit board
   that first appeared in this image?"
```

### Step 4: 품질 필터링

자동 생성된 쌍을 다층 필터로 걸러낸다.

| 필터 | 기준 | 이유 |
|---|---|---|
| q_T 형식 | "?" 포함 여부 | 질문이 아닌 경우 제거 |
| q_I ↔ GT 시각 관련성 | CLIP(K_i, K_j) > 0.5 | 참조 이미지가 정답과 무관한 경우 |
| q_T ↔ q_I 정합성 | CLIP_text(q_T, K_i) > threshold | 텍스트와 이미지가 어긋난 경우 |
| 영상별 쌍 수 제한 | 영상당 최대 N쌍 | 특정 영상 과표현 방지 |

### Step 5: Ground truth 구간 확정

```
Ground truth = [t_j - 15s, t_j + 15s]
```

기존 TCVP 논문의 temporal window 방식을 그대로 계승한다. Shot boundary detection을 추가 적용하면 더 정밀한 구간 설정이 가능하다.

<div class="ornament">· · ·</div>

## 전체 파이프라인 그림

<div style="overflow-x:auto; margin: 2rem 0;">
<svg viewBox="0 0 720 420" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:720px;font-family:'Source Serif 4',serif;">
  <!-- Background -->
  <rect width="720" height="420" fill="#fafaf8" rx="8"/>

  <!-- Title -->
  <text x="360" y="34" text-anchor="middle" font-size="15" font-weight="700" fill="#1a1a1a">TCVP-IMR 자동 구축 파이프라인</text>

  <!-- Step boxes -->
  <!-- Step 1 -->
  <rect x="30" y="60" width="150" height="72" rx="6" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="105" y="85" text-anchor="middle" font-size="11" font-weight="700" fill="#92400e">Step 1</text>
  <text x="105" y="101" text-anchor="middle" font-size="10" fill="#78350f">Keyframe 추출</text>
  <text x="105" y="116" text-anchor="middle" font-size="10" fill="#78350f">+ CLIP 임베딩</text>
  <text x="105" y="131" text-anchor="middle" font-size="9" fill="#a16207">(e_i per timestamp)</text>

  <!-- Arrow 1→2 -->
  <line x1="180" y1="96" x2="205" y2="96" stroke="#9ca3af" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Step 2 -->
  <rect x="205" y="60" width="155" height="72" rx="6" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="282" y="85" text-anchor="middle" font-size="11" font-weight="700" fill="#1e40af">Step 2</text>
  <text x="282" y="101" text-anchor="middle" font-size="10" fill="#1e3a8a">유사 쌍 탐색</text>
  <text x="282" y="116" text-anchor="middle" font-size="10" fill="#1e3a8a">0.55 &lt; sim &lt; 0.82</text>
  <text x="282" y="131" text-anchor="middle" font-size="9" fill="#1d4ed8">temporal gap &gt; 30s</text>

  <!-- Arrow 2→3 -->
  <line x1="360" y1="96" x2="385" y2="96" stroke="#9ca3af" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Step 3 -->
  <rect x="385" y="60" width="145" height="72" rx="6" fill="#d4f7d4" stroke="#16a34a" stroke-width="1.5"/>
  <text x="457" y="85" text-anchor="middle" font-size="11" font-weight="700" fill="#14532d">Step 3</text>
  <text x="457" y="101" text-anchor="middle" font-size="10" fill="#166534">LLM으로 q_T 생성</text>
  <text x="457" y="116" text-anchor="middle" font-size="10" fill="#166534">댓글 → 질문 변환</text>
  <text x="457" y="131" text-anchor="middle" font-size="9" fill="#15803d">(q_I 이미지 참조)</text>

  <!-- Arrow 3→4 -->
  <line x1="530" y1="96" x2="555" y2="96" stroke="#9ca3af" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Step 4 -->
  <rect x="555" y="60" width="135" height="72" rx="6" fill="#ffd6d6" stroke="#ef4444" stroke-width="1.5"/>
  <text x="622" y="85" text-anchor="middle" font-size="11" font-weight="700" fill="#7f1d1d">Step 4</text>
  <text x="622" y="101" text-anchor="middle" font-size="10" fill="#7f1d1d">품질 필터링</text>
  <text x="622" y="116" text-anchor="middle" font-size="10" fill="#7f1d1d">CLIP 정합성 검증</text>
  <text x="622" y="131" text-anchor="middle" font-size="9" fill="#991b1b">+ 형식 검사</text>

  <!-- Down arrow from Step 4 -->
  <line x1="622" y1="132" x2="622" y2="165" stroke="#9ca3af" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Output box -->
  <rect x="180" y="165" width="480" height="60" rx="6" fill="#f3f0ea" stroke="#78716c" stroke-width="1.5"/>
  <text x="420" y="189" text-anchor="middle" font-size="11" font-weight="700" fill="#292524">TCVP-IMR Dataset</text>
  <text x="420" y="207" text-anchor="middle" font-size="10" fill="#57534e">(q_T, q_I, Video V) → Ground truth [t_j − 15s, t_j + 15s]</text>
  <text x="420" y="222" text-anchor="middle" font-size="9" fill="#78716c">수만 개 규모 · 자동 생성 · 실제 유저 의도</text>

  <!-- Down arrow to fine-tune -->
  <line x1="420" y1="225" x2="420" y2="258" stroke="#9ca3af" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Fine-tune box -->
  <rect x="260" y="258" width="320" height="50" rx="6" fill="#ede9fe" stroke="#7c3aed" stroke-width="1.5"/>
  <text x="420" y="280" text-anchor="middle" font-size="11" font-weight="700" fill="#4c1d95">LanguageBind Fine-tuning (IMR)</text>
  <text x="420" y="297" text-anchor="middle" font-size="9" fill="#5b21b6">TCVP-IMR으로 학습 → MomentSeeker IMR 평가</text>

  <!-- Down arrow to result -->
  <line x1="420" y1="308" x2="420" y2="340" stroke="#9ca3af" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Result box -->
  <rect x="230" y="340" width="380" height="55" rx="6" fill="#1a1a1a" stroke="#1a1a1a"/>
  <text x="420" y="362" text-anchor="middle" font-size="11" font-weight="700" fill="#fef3c7">MomentSeeker IMR R@1</text>
  <text x="420" y="380" text-anchor="middle" font-size="10" fill="#d6d3d1">baseline 18.2 (LanguageBind)  →  ??? (TCVP-LB)</text>

  <!-- Arrow marker -->
  <defs>
    <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#9ca3af"/>
    </marker>
  </defs>
</svg>
</div>

<div class="ornament">· · ·</div>

## MomentSeeker 대비 비교

| | MomentSeeker IMR | TCVP-IMR (자동) |
|---|---|---|
| 규모 | ~360개 | 수만 개 예상 |
| 구축 방법 | 인간 어노테이션 | 완전 자동 |
| q_I 출처 | 어노테이터가 선택 | 타임스탬프 keyframe |
| q_T 출처 | 어노테이터가 작성 | LLM(댓글 → 질문 변환) |
| 의도의 자연성 | 인위적 구성 | 실제 유저 검색 의도 |
| q_I 영상 관계 | 동/타 영상 혼합 | 동일 영상 내 (v1) |

<div class="ornament">· · ·</div>

## 남은 과제와 한계

**Threshold 결정이 핵심이다.** CLIP 유사도 범위 `[0.55, 0.82]`는 pilot 실험 없이는 임의적이다. 너무 좁히면 쌍이 부족하고, 너무 열면 품질이 떨어진다. 소규모 영상 집합으로 threshold를 먼저 잡아야 한다.

**Cross-video 쌍은 v1 범위 밖이다.** MomentSeeker IMR은 다른 영상에서 가져온 q_I도 포함한다. TCVP v1은 같은 영상 내 쌍에 집중하고, cross-video는 다음 단계로 미룬다.

**LLM 변환 품질.** 댓글이 너무 짧거나("2:30 ㅋㅋ") 비시각적이면 q_T 생성 품질이 낮다. Visual 타입 댓글 필터링(Mixed Modality 분류기)과 연동하면 이 문제를 줄일 수 있다.

**Shot boundary 활용 미비.** 현재 ground truth window는 ±15초로 단순하다. Shot boundary detection을 도입하면 더 정밀한 구간 레이블이 가능하다.

<div class="footnote">
  참고: <a href="https://arxiv.org/abs/2502.12558">MomentSeeker: A Comprehensive Benchmark for Long-Video Moment Retrieval (2025)</a>
</div>
