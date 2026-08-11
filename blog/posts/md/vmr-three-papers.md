---
title: "비디오 모멘트 검색의 현재: TCVP · LongVALE · Momentseeker 심층 분석"
dek: "유튜브 댓글에서 실제 사용자 의도를 추출하는 TCVP, 오디오-비전 통합의 LongVALE, VMR의 난제를 정량화한 Momentseeker — 세 논문이 가리키는 곳."
desc: 세 논문 완전 비교 분석 — 데이터 철학의 차이, 오디오 처리 방식, 그리고 VMR이 아직 70%를 틀리는 이유.
tags: [Vision, Video, Multimodal]
date: Apr 2026
readtime: 28 min read
slug: vmr-three-papers
katex: false
---

## 왜 비디오에서 "순간"을 찾는가

2시간짜리 유튜브 영상에서 특정 장면을 찾는 상황을 상상해보자. 사람이라면 타임라인을 마우스로 드래그하며 15분을 들인다. AI가 "사람이 놀란 표정을 짓는 장면"이라는 텍스트를 받고 자동으로 찾아준다면?

**Video Moment Retrieval(VMR)**은 이런 문제를 다루는 분야다. 텍스트 쿼리에 맞는 비디오 시간 구간을 찾아내는 기술로, 영상 플랫폼, 편집 소프트웨어, 비디오 분석 AI까지 폭넓게 응용된다.

이 글에서는 2024~2025년 VMR 분야의 주목할 연구 세 편을 상세히 분석한다:

- **TCVP** — 유튜브 댓글의 타임스탬프로 "실제 사용자 의도"를 포착
- **LongVALE** — 오디오-비전 상관성을 명시적으로 모델링하는 장편 영상 이해
- **Momentseeker** — "현재 모델들이 무엇을 못하는가"를 정량화한 벤치마크

세 논문 각각을 먼저 깊이 파고든 뒤, 비교 분석한다.

<div class="ornament">· · ·</div>

## Part 1: TCVP — 사용자가 직접 알려주는 중요한 순간

### 1.1 문제 정의: 기존 데이터셋의 세 가지 결함

기존 VMR 데이터셋이 공통적으로 가지는 문제가 있다.

**첫째, 순간 선택이 임의적이다.** 전문가가 "이 장면이 중요하다"고 판단해 주석을 달지만, 이게 실제로 시청자가 다시 보고 싶어 하는 순간인지는 검증하지 않는다.

**둘째, 쿼리가 캡션처럼 읽힌다.** "A man is speaking about tearing a dollar bill, explaining the Banach–Tarski theorem" 같은 쿼리는 직접 검색창에 입력할 표현이 아니다. 사람들이 실제로 검색하는 방식과 다르다.

**셋째, 오디오를 무시한다.** 기존 연구 대부분이 시각 정보에만 집중한다. 하지만 유튜브 댓글 분석 결과, 실제 사용자 관심의 약 50%는 오디오(음성, 음악, 효과음)에 기반한다.

TCVP(Timestamped Comment-aware Video Moment Retrieval)의 핵심 아이디어는 단순하다: **유튜브에 이미 존재하는 타임스탬프 댓글을 그대로 활용하자.**

<div class="pullquote">
  <strong>유튜브 댓글 하나에는 세 가지 정보가 동시에 담겨 있다: 언제(타임스탬프), 얼마나 중요한지(좋아요 수), 그리고 무엇인지(댓글 내용).</strong>
</div>

### 1.2 파이프라인: 4단계

<div style="margin: 2rem 0;">
<svg viewBox="0 0 740 180" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:740px;display:block;margin:0 auto;font-family:'Source Serif 4',serif;">
  <defs>
    <marker id="arrt" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#555"/>
    </marker>
  </defs>
  <!-- Stage 1 -->
  <rect x="10" y="50" width="148" height="80" rx="8" fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>
  <text x="84" y="78" text-anchor="middle" font-size="11" fill="#92400e" font-weight="bold">① 데이터 수집</text>
  <text x="84" y="95" text-anchor="middle" font-size="10" fill="#78350f">구독자 1M+ 채널</text>
  <text x="84" y="110" text-anchor="middle" font-size="10" fill="#78350f">타임스탬프 댓글 크롤</text>
  <text x="84" y="124" text-anchor="middle" font-size="10" fill="#78350f">비디오당 상위 20개</text>
  <!-- arrow -->
  <line x1="158" y1="90" x2="183" y2="90" stroke="#555" stroke-width="1.5" marker-end="url(#arrt)"/>
  <!-- Stage 2 -->
  <rect x="183" y="50" width="148" height="80" rx="8" fill="#d1fae5" stroke="#10b981" stroke-width="1.5"/>
  <text x="257" y="78" text-anchor="middle" font-size="11" fill="#065f46" font-weight="bold">② 모달리티 캡션</text>
  <text x="257" y="95" text-anchor="middle" font-size="10" fill="#047857">±9초 윈도우</text>
  <text x="257" y="110" text-anchor="middle" font-size="10" fill="#047857">비전 캡션 (≤20단어)</text>
  <text x="257" y="124" text-anchor="middle" font-size="10" fill="#047857">오디오 캡션 (≤20단어)</text>
  <!-- arrow -->
  <line x1="331" y1="90" x2="356" y2="90" stroke="#555" stroke-width="1.5" marker-end="url(#arrt)"/>
  <!-- Stage 3 -->
  <rect x="356" y="50" width="148" height="80" rx="8" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="430" y="78" text-anchor="middle" font-size="11" fill="#1d4ed8" font-weight="bold">③ 필터링 + 게이팅</text>
  <text x="430" y="95" text-anchor="middle" font-size="10" fill="#1e40af">코사인 유사도 τ=0.3</text>
  <text x="430" y="110" text-anchor="middle" font-size="10" fill="#1e40af">비전 vs 오디오 분류</text>
  <text x="430" y="124" text-anchor="middle" font-size="10" fill="#1e40af">무의미 댓글 제거</text>
  <!-- arrow -->
  <line x1="504" y1="90" x2="529" y2="90" stroke="#555" stroke-width="1.5" marker-end="url(#arrt)"/>
  <!-- Stage 4 -->
  <rect x="529" y="50" width="148" height="80" rx="8" fill="#fce7f3" stroke="#ec4899" stroke-width="1.5"/>
  <text x="603" y="78" text-anchor="middle" font-size="11" fill="#9d174d" font-weight="bold">④ 쿼리 생성</text>
  <text x="603" y="95" text-anchor="middle" font-size="10" fill="#be185d">GPT-4.1</text>
  <text x="603" y="110" text-anchor="middle" font-size="10" fill="#be185d">댓글 키워드 추출</text>
  <text x="603" y="124" text-anchor="middle" font-size="10" fill="#be185d">검색 지향 쿼리</text>
</svg>
</div>

**Stage 1 — 데이터 수집**

구독자 100만 명 이상 유튜브 채널의 영상을 크롤한다. 모든 타임스탬프 댓글을 수집하고, 같은 시점의 댓글은 좋아요 수 최다 것만 유지한다. 비디오당 상위 20개 댓글을 선택한다.

왜 상위 20개인가? 좋아요 수가 많다는 것은 많은 시청자가 공감한 순간이라는 의미다. "좋아요 = 중요도 신호"라는 전제다.

**Stage 2 — 모달리티별 캡션 생성**

각 타임스탬프 댓글의 ±9초 윈도우에서 Qwen2.5-Omni로 두 종류의 캡션을 생성한다:

- **비전 캡션**: "남자가 공을 차려고 준비하는 장면" (20단어 이하)
- **오디오 캡션**: "짜릿한 음악이 재생 중, 군중의 환호 소리" (20단어 이하)

캡션을 20단어 이하로 제한한 이유가 있다. 댓글 자체가 평균 10~20단어다. 긴 캡션과 짧은 댓글 사이의 유사도는 자연히 낮아진다. 짧은 캡션으로 맞추면 필터링 정확도가 올라간다.

9초 윈도우를 선택한 이유도 데이터 기반이다: 어블레이션 실험에서 ±9초가 97% 커버리지를 달성했다. 대부분의 댓글이 가리키는 순간은 해당 타임스탬프 기준 ±9초 안에 있다는 뜻이다.

**Stage 3 — 댓글 필터링 & 모달리티 게이팅**

모든 댓글이 의미 있지는 않다. "ㅋㅋㅋㅋ", "진짜 좋다", "lol" 같은 댓글은 영상 내용과 무관하다.

**필터링**: 각 댓글과 두 캡션(비전, 오디오) 사이의 코사인 유사도를 계산한다. 최댓값이 임계값 τ=0.3 미만이면 제거한다.

```
댓글: "경기장 개미쳤다"
  비전 유사도: 0.45 → 유지 (0.45 > 0.3)

댓글: "ㅋㅋㅋㅋ"
  비전 유사도: 0.08, 오디오 유사도: 0.05 → 제거 (둘 다 < 0.3)

댓글: "꼬마 언니 목소리 쩌네"
  비전 유사도: 0.15, 오디오 유사도: 0.35 → 유지 (0.35 > 0.3)
```

**모달리티 게이팅**: 남은 댓글을 "비전 관련" 또는 "오디오 관련"으로 분류한다. 비전 유사도가 높으면 시각적 순간, 오디오 유사도가 높으면 청각적 순간이다.

최종 분포:
- 오디오 관련: **45.9%**
- 비전 관련: **39.8%**
- 필터링 제거: 14.3%

이 결과는 YTCommentQA 연구의 "유튜브 Q&A의 50%는 오디오 관련"이라는 발견과 일치한다. 기존 VMR 데이터셋이 시각 편향적임을 보여주는 수치다.

**Stage 4 — 쿼리 생성**

GPT-4.1에 원본 댓글, 모달리티 분류, 해당 모달리티 캡션을 입력한다. GPT가 생성하는 쿼리는 세 가지 조건을 만족해야 한다:

1. 댓글의 키워드를 직접 활용할 것
2. 자연스러운 검색 쿼리 형식일 것 ("설명문"이 아닌 "검색어")
3. 20단어 이내일 것

예시:
- 댓글: "여기서 승부난다!!!"
- 캡션: "선수가 결정적인 순간에 슛을 날리는 장면"
- 생성 쿼리: "player scores decisive shot in final seconds"

이 과정이 중요한 이유: 같은 비디오 순간을 LongVALE 방식으로 주석 달면 "A man wearing a blue jersey is seen dribbling past two defenders before releasing a powerful shot that curves into the top corner of the goal"처럼 묘사적 문장이 된다. TCVP 방식은 사람이 실제로 검색창에 입력하는 표현을 만든다.

### 1.3 인간 평가 결과

**순간 선택의 타당성**: 댓글 기반 순간 선택과 무작위 선택 중 어느 쪽이 더 흥미로운 순간인지를 3명의 독립 평가자에게 물었다.

- 댓글 기반 선택 선호: **95%**
- 무작위 선택 선호: 5%

실제 사람들이 댓글을 달고 좋아요를 누른 순간이, 무작위 선택한 순간보다 압도적으로 "다시 보고 싶은 순간"이라는 검증이다.

**쿼리의 자연스러움**: 4가지 방법으로 생성한 쿼리를 비교했다.

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.87rem;">
  <thead>
    <tr style="background:#f5f5f0;">
      <th style="padding:0.6rem 0.9rem;text-align:left;border-bottom:2px solid #ddd;">방법</th>
      <th style="padding:0.6rem 0.9rem;text-align:left;border-bottom:2px solid #ddd;">선호도</th>
      <th style="padding:0.6rem 0.9rem;text-align:left;border-bottom:2px solid #ddd;">특성</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"><strong>TCVP (댓글 포함)</strong></td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"><span style="background:#d1fae5;padding:2px 6px;border-radius:3px;font-weight:bold;">70%</span></td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">검색 지향, 짧고 자연스럽다</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">TCVP (댓글 없이)</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">25%</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">캡션만 활용, 나쁘지는 않으나 어색</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">LongVALE</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">5%</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">묘사적, 검색창에 입력하기엔 너무 길다</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">Watch&Listen</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">0%</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">설명 위주, 검색 쿼리로 사용 불가</td>
    </tr>
  </tbody>
</table>

댓글이 있을 때와 없을 때의 차이가 70% vs 25%다. 댓글 자체가 핵심이다. 모달리티 캡션만 가지고는 자연스러운 검색 쿼리를 만들기 어렵다.

### 1.4 모델 벤치마크 결과

두 가지 평가 방식으로 진행됐다.

**방식 1 — MLLM 타임스탬프 직접 예측**

100개 균일 샘플 프레임을 모델에 넣고 직접 타임스탬프를 예측하게 한다. 결과가 충격적이다:

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.87rem;">
  <thead>
    <tr style="background:#f5f5f0;">
      <th style="padding:0.6rem 0.9rem;text-align:left;border-bottom:2px solid #ddd;">입력 조건</th>
      <th style="padding:0.6rem 0.9rem;text-align:center;border-bottom:2px solid #ddd;">비전 R@1</th>
      <th style="padding:0.6rem 0.9rem;text-align:center;border-bottom:2px solid #ddd;">오디오 R@1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">MLLM (비전만)</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;">6.0%</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;">4.0%</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">+ 오디오 파형 추가</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;">6.0%</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;">6.0% (+2%)</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"><strong>+ ASR (자막) 추가</strong></td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;"><strong>26.0%</strong></td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;"><strong>26.0%</strong></td>
    </tr>
  </tbody>
</table>

<div class="callout">
  <strong>핵심 발견:</strong> 오디오 파형(+2%)보다 ASR 텍스트(+20%)가 훨씬 효과적이다. 현재 MLLM들은 음성의 의미론적 내용을 raw 오디오에서 직접 추출하는 것보다, 텍스트로 변환된 내용을 처리할 때 훨씬 잘 작동한다.
</div>

**방식 2 — 세그먼트 캡션 기반 검색**

비디오를 10초 단위로 나누고, 각 세그먼트 캡션과 쿼리의 임베딩 유사도로 랭킹을 매긴다.

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.87rem;">
  <thead>
    <tr style="background:#f5f5f0;">
      <th style="padding:0.6rem 0.9rem;text-align:left;border-bottom:2px solid #ddd;">모델</th>
      <th style="padding:0.6rem 0.9rem;text-align:center;border-bottom:2px solid #ddd;">비전 R@1</th>
      <th style="padding:0.6rem 0.9rem;text-align:center;border-bottom:2px solid #ddd;">비전 R@10</th>
      <th style="padding:0.6rem 0.9rem;text-align:center;border-bottom:2px solid #ddd;">오디오 R@10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">LanguageBind</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;">30.5%</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;">62.7%</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;">37.0%</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">Qwen2.5-Omni (자막 없음)</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;">—</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;">37.6%</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;">27.5%</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"><strong>Qwen2.5-Omni + 자막</strong></td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;">—</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;"><strong>38.8%</strong></td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;"><span style="background:#d1fae5;padding:2px 6px;border-radius:3px;font-weight:bold;">48.2%</span></td>
    </tr>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">CLAP (오디오 전용)</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;">—</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;">—</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;">24.2%</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">CLIP</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;">14.9%</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;">54.3%</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;">34.4%</td>
    </tr>
  </tbody>
</table>

놀라운 발견이 있다. 오디오 쿼리에 대해 **오디오 전용 모델(CLAP, 24.2%)보다 비전 임베딩 모델(LanguageBind, 37.0%)이 더 높은 성능**을 보인다. 이유는 간단하다 — 오디오 관련 쿼리 중 상당수가 "말하는 사람"의 시각적 모습과도 연관되어 있다. 음성 내용과 시각 컨텍스트가 함께 중요한 것이다.

또한 자막(ASR) 추가로 오디오 R@10이 27.5% → 48.2%로 75% 상대 향상을 보인다.

<div class="ornament">· · ·</div>

## Part 2: LongVALE — 오디오와 비전을 명시적으로 연결하다

### 2.1 핵심 아이디어

LongVALE(Long-term Video Aggregation with Language Events)의 출발점은 근본적 문제 인식이다: **기존 비디오 이해 모델은 비디오를 이미지의 연속으로만 봤다.** 마치 영화를 음소거하고 보는 것과 같다.

음악 영상에서 표정(비전)과 리듬(오디오)은 별개가 아니라 함께 의미를 만든다. LongVALE의 전략은 이 관계를 **명시적으로 모델에 학습시키는 것**이다.

### 2.2 데이터셋: 100K → 8.4K의 엄격한 선별

ACAV-100M의 YouTube 영상 100,000개에서 시작해 8,411개만 선별했다. 선택률 8.4%다.

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.87rem;">
  <thead>
    <tr style="background:#f5f5f0;">
      <th style="padding:0.6rem 0.9rem;text-align:left;border-bottom:2px solid #ddd;">필터 단계</th>
      <th style="padding:0.6rem 0.9rem;text-align:left;border-bottom:2px solid #ddd;">기준</th>
      <th style="padding:0.6rem 0.9rem;text-align:left;border-bottom:2px solid #ddd;">목적</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">1단계</td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">해상도 ≥360p</td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">시각 품질 보장</td></tr>
    <tr><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">2단계</td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">영어 자막 존재</td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">언어 일관성</td></tr>
    <tr><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">3단계</td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">음성 비중 < 95%</td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">오디오 다양성</td></tr>
    <tr><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">4단계</td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">정적 씬 < 80%</td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">동적 정보 확보</td></tr>
    <tr><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"><strong>5단계</strong></td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"><strong>C-MCR 유사도 > 0.25</strong></td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"><strong>오디오-비전 실제 연관성</strong></td></tr>
  </tbody>
</table>

마지막 필터가 핵심이다. C-MCR(Correlation-based Moment Candidate Retrieval)로 각 30초 구간에서 오디오와 비전이 실제로 연관되어 있는지 측정한다. 음악 영상은 통과, 조용한 배경에 내레이션만 있는 영상은 탈락이다.

### 2.3 이벤트 경계 감지: 3단계 계층 구조

비디오는 연속체지만 의미 있는 단위로 나눠야 한다. LongVALE는 **오디오 경계와 비전 경계를 별도로 감지한 후 합치는** 방식을 취한다.

**Level 1 — 비전 이벤트 경계**: PySceneDetect로 시각적 장면 전환 감지. 짧은 씬들은 의미적으로 병합.

**Level 2 — 오디오 이벤트 경계** (논문에서 처음 제시):
- MFCC(음성 스펙트럼 특성)으로 음향 변화 포착
- CLAP 임베딩으로 의미적으로 유사한 오디오 구간 병합

**Level 3 — Omni-Modal 경계** (합성):
```
비전 이벤트 경계 ≠ 오디오 이벤트 경계인 경우가 많다

해결: 비전 이벤트의 종료 시점을
      그 안에 포함된 모든 오디오 이벤트 중
      가장 늦게 끝나는 지점으로 연장한다

→ 오디오 의미 무결성(semantic integrity) 보장
```

정량 평가에서 이 방식이 MRSD(의미적 일관성) 지표상 단일 모달리티 경계보다 우수하다는 것이 확인됐다.

### 2.4 오디오-비전 상관성(AVC): 7가지 패턴

LongVALE의 가장 차별화된 기여다. 오디오와 비전이 "어떤 방식으로 연관되어 있는지"를 8가지 패턴으로 명시적 분류한다.

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.87rem;">
  <thead>
    <tr style="background:#f5f5f0;">
      <th style="padding:0.6rem 0.9rem;text-align:left;border-bottom:2px solid #ddd;">패턴</th>
      <th style="padding:0.6rem 0.9rem;text-align:left;border-bottom:2px solid #ddd;">설명</th>
      <th style="padding:0.6rem 0.9rem;text-align:left;border-bottom:2px solid #ddd;">예시</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;font-weight:bold;">Complementary</td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">오디오가 비전을 설명</td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">인터뷰 영상 — 말하는 사람 화면 + 음성</td></tr>
    <tr><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;font-weight:bold;">Synchronicity</td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">V와 A가 동시에 발생</td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">음악 비트 + 춤 동작이 정확히 동기화</td></tr>
    <tr><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;font-weight:bold;">Enhancement</td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">오디오가 분위기 극대화</td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">자동차 엔진음 + 레이싱 영상 → 속도감 배가</td></tr>
    <tr><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;font-weight:bold;">Scene-Aware</td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">오디오가 맥락 제공</td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">카페 배경음 → 카페 환경 파악</td></tr>
    <tr><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;font-weight:bold;">Causality</td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">A가 V의 원인 또는 결과</td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">선수의 득점(비전) → 관중 환호(오디오)</td></tr>
    <tr><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;font-weight:bold;">Corrective</td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">오디오가 비전의 오해 수정</td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">진지한 표정(비전) + 웃음소리(오디오) → 코미디 쇼</td></tr>
    <tr><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;font-weight:bold;">Temporal Association</td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">순차적 시간 관계</td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">도입부 → 절정 순서로 전개</td></tr>
  </tbody>
</table>

이 7가지 패턴을 Gemini-1.5-Pro로 자동 분류하고, 2,000개 비디오를 인간이 수동 검수(115 human hours)한다. 전체 캡션의 78%가 미세한 시간적 변화를 포착한다.

### 2.5 2단계 학습

모델 학습을 두 단계로 분리한 것이 LongVALE의 방법론적 기여다.

**Stage 1 — Boundary Perception(경계 인식)**: 템플릿 기반으로 자동 생성한 7,240개 Q&A 대화 사용. "이 순간이 이벤트 경계인가"를 판단하는 기초 능력 훈련. LoRA 파인튜닝(Vicuna-7b 기반).

**Stage 2 — Instruction Tuning(지시 학습)**: Gemini-1.5-Pro가 생성한 25,400개 고품질 대화. "가장 긴장감 있는 순간은?" 같은 자연스러운 쿼리에 답변하는 고급 능력 훈련.

두 단계를 분리한 이유: 기초 능력(경계 감지) → 고급 능력(자유 쿼리) 순서로 학습하는 것이 end-to-end 학습보다 최종 성능이 높다는 것을 실험으로 확인했다.

### 2.6 AVC가 가져온 성능 향상

AVC 분석을 추가했을 때의 캡셔닝 성능:

- Dense Video Captioning: CIDEr 3.5 → **7.3 (+108.6%)**
- Scene Captioning: CIDEr 10.4 → **21.1 (+102.9%)**

오디오-비전 관계를 명시적으로 추론하게 하는 것만으로 성능이 2배 이상 오른다.

Zero-shot 평가에서도 강점이 드러난다. AVSD 벤치마크에서:
- LongVALE: 54.8% (훈련 데이터 0.7M)
- 기존 AVicuna: 53.1% (훈련 데이터 1.1M)

**36% 적은 데이터로 더 좋은 성능.** 데이터의 양보다 오디오-비전 상관성이라는 질이 중요하다는 증명이다.

<div class="ornament">· · ·</div>

## Part 3: Momentseeker — VMR이 아직 70%를 틀리는 이유

### 3.1 핵심 질문

LongVALE이 "좋은 데이터를 어떻게 만드는가"에 집중했다면, Momentseeker는 다른 질문을 던진다: **"비디오 모멘트 검색에서 진짜 어려운 게 뭔가?"**

2,500개 비디오, 70,000개 이상 주석으로 구성된 벤치마크를 만들고 15개 이상의 최신 모델(Gemini 2.5, GPT-4V, Qwen2.5VL-72B 등)을 평가했다. 결론부터: **가장 좋은 모델도 R@1 기준 29.6%다. 10개 중 7개는 틀린다.**

### 3.2 계층적 어려움 분류

Momentseeker의 핵심 기여는 **"어떤 종류의 쿼리가 어려운지"** 체계적으로 분류한 것이다.

<div style="margin: 2rem 0;">
<svg viewBox="0 0 640 260" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;display:block;margin:0 auto;font-family:'Source Serif 4',serif;">
  <defs>
    <marker id="arrd" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#555"/>
    </marker>
  </defs>
  <!-- Title -->
  <text x="320" y="22" text-anchor="middle" font-size="13" fill="#374151" font-weight="bold">Momentseeker 계층적 어려움 분류</text>
  <!-- Global -->
  <rect x="20" y="40" width="180" height="80" rx="8" fill="#ffd6d6" stroke="#ef4444" stroke-width="1.5"/>
  <text x="110" y="65" text-anchor="middle" font-size="12" fill="#7f1d1d" font-weight="bold">Global Level</text>
  <text x="110" y="82" text-anchor="middle" font-size="10" fill="#991b1b">인과관계, 공간 추론 등</text>
  <text x="110" y="97" text-anchor="middle" font-size="10" fill="#991b1b">고차원 추론 필요</text>
  <rect x="20" y="128" width="180" height="28" rx="4" fill="#fca5a5"/>
  <text x="110" y="147" text-anchor="middle" font-size="11" fill="#7f1d1d" font-weight="bold">R@1: 10~20% (최저)</text>
  <!-- Event -->
  <rect x="230" y="40" width="180" height="80" rx="8" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="320" y="65" text-anchor="middle" font-size="12" fill="#78350f" font-weight="bold">Event Level</text>
  <text x="320" y="82" text-anchor="middle" font-size="10" fill="#92400e">행동, 이벤트 인식</text>
  <text x="320" y="97" text-anchor="middle" font-size="10" fill="#92400e">정확한 시작점 판정</text>
  <rect x="230" y="128" width="180" height="28" rx="4" fill="#fde68a"/>
  <text x="320" y="147" text-anchor="middle" font-size="11" fill="#78350f" font-weight="bold">R@1: 15~25%</text>
  <!-- Object -->
  <rect x="440" y="40" width="180" height="80" rx="8" fill="#d1fae5" stroke="#10b981" stroke-width="1.5"/>
  <text x="530" y="65" text-anchor="middle" font-size="12" fill="#065f46" font-weight="bold">Object Level</text>
  <text x="530" y="82" text-anchor="middle" font-size="10" fill="#047857">물체, 속성, OCR</text>
  <text x="530" y="97" text-anchor="middle" font-size="10" fill="#047857">상대적으로 낮은 난이도</text>
  <rect x="440" y="128" width="180" height="28" rx="4" fill="#a7f3d0"/>
  <text x="530" y="147" text-anchor="middle" font-size="11" fill="#065f46" font-weight="bold">R@1: 12~35%</text>
  <!-- Examples -->
  <text x="110" y="195" text-anchor="middle" font-size="10" fill="#6b7280" font-style="italic">"누군가 화난 후</text>
  <text x="110" y="210" text-anchor="middle" font-size="10" fill="#6b7280" font-style="italic">화해하는 장면"</text>
  <text x="320" y="195" text-anchor="middle" font-size="10" fill="#6b7280" font-style="italic">"누군가 문을</text>
  <text x="320" y="210" text-anchor="middle" font-size="10" fill="#6b7280" font-style="italic">열기 시작하는 순간"</text>
  <text x="530" y="195" text-anchor="middle" font-size="10" fill="#6b7280" font-style="italic">"빨간 의자가</text>
  <text x="530" y="210" text-anchor="middle" font-size="10" fill="#6b7280" font-style="italic">처음 보이는 순간"</text>
</svg>
</div>

Global Level이 가장 어렵다. "화난 후 화해하는 장면"은 여러 프레임에 걸친 인과관계 추론이 필요하다. "빨간 의자"를 찾는 건 단순 물체 탐지라 상대적으로 쉽다.

### 3.3 두 가지 평가 패러다임

같은 문제를 해결하는 두 방식 — Retrieval과 Generation — 의 성능이 다르다.

**Retrieval (검색)**: 영상을 10초 단위로 나눈 후, 각 청크와 쿼리의 유사도를 측정해 상위 순위를 반환.
- 장점: 여러 후보에서 선택하므로 에러 복구 가능
- 단점: 경계가 고정되어 세밀한 순간을 놓칠 수 있음
- 성능: R@1 30~40%

**Generation (생성)**: 100개 프레임을 MLLM에 입력해 직접 타임스탬프 예측.
- 장점: 이론상 어떤 순간이든 정확히 지정 가능
- 단점: 위치 편향 문제가 심각 (아래 참조)
- 성능: 소형 모델은 낮음, Gemini 2.5만 경쟁력

### 3.4 충격적 발견 1: 위치 편향

생성 기반 모델들이 보이는 편향이다.

**Gemini 2.5 Pro(SOTA)의 타임스탬프 예측 정확도:**

<div style="margin: 1.5rem 0;">
<svg viewBox="0 0 500 180" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:500px;display:block;margin:0 auto;font-family:'Source Serif 4',serif;">
  <text x="250" y="20" text-anchor="middle" font-size="12" fill="#374151" font-weight="bold">영상 위치별 예측 정확도 (Gemini 2.5 Pro)</text>
  <!-- bars -->
  <rect x="60" y="35" width="87" height="105" rx="4" fill="#10b981"/>
  <text x="103" y="28" text-anchor="middle" font-size="11" fill="#065f46" font-weight="bold">35%</text>
  <rect x="167" y="65" width="87" height="75" rx="4" fill="#f59e0b"/>
  <text x="210" y="58" text-anchor="middle" font-size="11" fill="#92400e" font-weight="bold">28%</text>
  <rect x="274" y="90" width="87" height="50" rx="4" fill="#ef4444"/>
  <text x="317" y="83" text-anchor="middle" font-size="11" fill="#7f1d1d" font-weight="bold">22%</text>
  <rect x="381" y="110" width="87" height="30" rx="4" fill="#7f1d1d"/>
  <text x="424" y="103" text-anchor="middle" font-size="11" fill="#7f1d1d" font-weight="bold">18%</text>
  <!-- x axis -->
  <line x1="50" y1="140" x2="480" y2="140" stroke="#d1d5db" stroke-width="1"/>
  <text x="103" y="157" text-anchor="middle" font-size="10" fill="#6b7280">0~25%</text>
  <text x="210" y="157" text-anchor="middle" font-size="10" fill="#6b7280">25~50%</text>
  <text x="317" y="157" text-anchor="middle" font-size="10" fill="#6b7280">50~75%</text>
  <text x="424" y="157" text-anchor="middle" font-size="10" fill="#6b7280">75~100%</text>
  <text x="265" y="175" text-anchor="middle" font-size="10" fill="#9ca3af">영상 내 상대적 위치</text>
</svg>
</div>

영상 초반부(0~25%)에서는 35% 정확도지만, 후반부(75~100%)에서는 18%로 절반 가까이 떨어진다. 모델이 훈련 데이터에서 "중요한 순간은 영상 초반에 많다"는 패턴을 학습해 편향이 생긴 것이다.

### 3.5 충격적 발견 2: 멀티모달 쿼리가 오히려 성능을 낮춘다

직관적으로 이미지나 비디오 클립을 함께 쿼리하면 더 정확할 것 같다. 그런데 현실은 반대다.

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.87rem;">
  <thead>
    <tr style="background:#f5f5f0;">
      <th style="padding:0.6rem 0.9rem;text-align:left;border-bottom:2px solid #ddd;">쿼리 방식</th>
      <th style="padding:0.6rem 0.9rem;text-align:center;border-bottom:2px solid #ddd;">R@1 (Gemini 2.5)</th>
      <th style="padding:0.6rem 0.9rem;text-align:center;border-bottom:2px solid #ddd;">변화</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">TMR — 텍스트만</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;"><strong>29.6%</strong></td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;">기준</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">IMR — 텍스트 + 이미지</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;">24.3%</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;"><span style="color:#ef4444;">-17%</span></td>
    </tr>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">VMR — 텍스트 + 비디오 클립</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;">20.1%</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;"><span style="color:#ef4444;">-32%</span></td>
    </tr>
  </tbody>
</table>

현재 MLLM들이 멀티모달 정보를 통합하는 능력이 부족하다. 추가 정보가 도움이 되는 게 아니라 노이즈처럼 작용한다.

### 3.6 충격적 발견 3: 프레임 수가 성능에 미치는 영향

더 많은 프레임을 보면 당연히 성능이 오른다. 그 규모가 문제다.

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.87rem;">
  <thead>
    <tr style="background:#f5f5f0;">
      <th style="padding:0.6rem 0.9rem;text-align:center;border-bottom:2px solid #ddd;">프레임 수</th>
      <th style="padding:0.6rem 0.9rem;text-align:center;border-bottom:2px solid #ddd;">R@1</th>
      <th style="padding:0.6rem 0.9rem;text-align:center;border-bottom:2px solid #ddd;">향상</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;">96 프레임</td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;">15%</td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;">기준</td></tr>
    <tr><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;">192 프레임</td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;">22%</td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;">+46%</td></tr>
    <tr><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;">768 프레임</td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;">27%</td><td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;text-align:center;"><strong>+80%</strong></td></tr>
  </tbody>
</table>

프레임 수를 8배 늘리면 성능이 80% 오른다. 그러나 연산량도 8배다. MLLM의 컨텍스트 윈도우 제약과 현실적 비용이 한계다.

<div class="ornament">· · ·</div>

## Part 4: 세 논문 비교 분석

### 4.1 데이터 철학의 차이

세 논문은 "좋은 데이터란 무엇인가"에 대해 서로 다른 답을 내놓는다.

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.87rem;">
  <thead>
    <tr style="background:#f5f5f0;">
      <th style="padding:0.6rem 0.9rem;text-align:left;border-bottom:2px solid #ddd;">논문</th>
      <th style="padding:0.6rem 0.9rem;text-align:left;border-bottom:2px solid #ddd;">데이터 철학</th>
      <th style="padding:0.6rem 0.9rem;text-align:left;border-bottom:2px solid #ddd;">핵심 신호</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"><strong>TCVP</strong></td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">사람들이 이미 중요하다고 표시한 것을 사용하라</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">유튜브 댓글 타임스탬프 + 좋아요</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"><strong>LongVALE</strong></td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">오디오-비전이 실제로 연관된 영상만 선별하라</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">C-MCR 상관성 + 115 human hours 검수</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"><strong>Momentseeker</strong></td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">어려운 케이스를 체계적으로 수집하라</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">전문가 주석 + 계층적 난이도 분류</td>
    </tr>
  </tbody>
</table>

### 4.2 쿼리 생성 방식

세 논문의 쿼리 스타일이 근본적으로 다르다.

| | TCVP | LongVALE | Momentseeker |
|---|---|---|---|
| **방식** | 댓글 키워드 추출 → GPT 재구성 | 모달리티별 캡션 → 통합 → 대화 생성 | 전문가 직접 작성 |
| **특성** | 검색 지향, 짧음 (~20단어) | 묘사적, 김 (50단어+) | 다양, 계층별 |
| **예시** | "player scores decisive shot" | "A man is speaking about tearing a dollar bill explaining Banach-Tarski theorem" | "Where was the scale when it first appeared?" |
| **실용성** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

### 4.3 오디오 처리 방식

<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.87rem;">
  <thead>
    <tr style="background:#f5f5f0;">
      <th style="padding:0.6rem 0.9rem;text-align:left;border-bottom:2px solid #ddd;">논문</th>
      <th style="padding:0.6rem 0.9rem;text-align:left;border-bottom:2px solid #ddd;">철학</th>
      <th style="padding:0.6rem 0.9rem;text-align:left;border-bottom:2px solid #ddd;">구현</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"><strong>TCVP</strong></td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">비전과 오디오를 분리 (이진 선택)</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">모달리티 게이팅으로 각 쿼리에 하나만 배정</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"><strong>LongVALE</strong></td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">V+A+S를 통합적으로 처리</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">별도 인코더(BEATs+CLIP) + AVC로 관계 명시화</td>
    </tr>
    <tr>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;"><strong>Momentseeker</strong></td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">멀티모달 쿼리 평가</td>
      <td style="padding:0.55rem 0.9rem;border-bottom:1px solid #eee;">TMR/IMR/VMR 3가지 방식 비교 (TMR이 최고)</td>
    </tr>
  </tbody>
</table>

흥미롭게도 TCVP의 "분리" 접근이 가장 실용적이다. 오디오와 비전을 통합하려는 LongVALE의 접근이 학술적으로 우아하지만, Momentseeker가 증명했듯이 멀티모달 통합은 현재 모델들에게 오히려 성능을 낮춘다.

### 4.4 세 논문이 공통으로 강조하는 것

**1. 데이터 질이 양보다 중요하다**

LongVALE: 36% 적은 데이터로 더 좋은 성능.
TCVP: 타임스탬프 + 좋아요가 자동 품질 신호.
교훈: 1M의 잡음 섞인 데이터보다 10K의 깨끗한 데이터가 낫다.

**2. 모달리티 통합은 단순히 합치는 게 아니다**

LongVALE: 7가지 AVC 패턴 명시 없이는 효과 없음.
Momentseeker: 멀티모달 쿼리가 오히려 성능 하락.
TCVP: 모달리티별 게이팅으로 명확한 할당.
교훈: 단순 결합이 아닌 구조화된 통합이 필요하다.

**3. ASR(자동음성인식)이 오디오 이해의 핵심이다**

TCVP: ASR 추가로 오디오 R@1이 4% → 26%.
LongVALE: Speech 인코더(Whisper)가 별도로 존재.
교훈: 현재 MLLM들은 raw 오디오 파형보다 텍스트로 변환된 내용을 훨씬 잘 처리한다.

**4. 위치 편향은 구조적 문제다**

Momentseeker가 정량화한 것을 TCVP와 LongVALE도 (암묵적으로) 알고 있다. 두 논문 모두 생성 기반보다 검색 기반 접근을 채택한다.

<div class="ornament">· · ·</div>

## Part 5: TCVP 강화를 위한 실험 제안

TCVP는 현재 데이터셋 논문으로서의 위치에 있다. LongVALE와 Momentseeker가 밝힌 인사이트를 활용해 더 강한 논문으로 만들 수 있는 실험들을 정리한다.

### Tier 1: 반드시 추가해야 할 실험

**실험 1 — 필터링 임계값(τ) 어블레이션**

τ=0.3의 선택이 왜 최적인지 정량화한다. τ ∈ {0.1, 0.2, 0.3, 0.4, 0.5}로 필터링한 뒤 댓글 유지율, 쿼리 품질(인간 평가), 모델 성능(R@1)을 측정한다.

| τ | 유지율 | 쿼리 품질 | 모델 R@1 |
|---|---|---|---|
| 0.1 | 95% | 45% | ~18% (노이즈 과다) |
| 0.2 | 92% | 62% | ~25% |
| **0.3** | **85.7%** | **70%** | **~28% (최적)** |
| 0.4 | 70% | 78% | ~26% (너무 엄격) |
| 0.5 | 55% | 82% | ~22% |

의의: 설계 선택의 정당성을 증명하고, 타 도메인 적용 시 가이드를 제시한다.

**실험 2 — 카테고리별 난이도 분석 (Momentseeker 방법론 응용)**

TCVP에는 이미 카테고리 정보가 있다. Momentseeker의 계층적 어려움 분류 방법을 적용한다.

| 카테고리 | 예상 난이도 | 이유 |
|---|---|---|
| 스포츠 | 낮음 | 시각적으로 명확한 순간이 많음 |
| 교육/지식 | 중간 | 오디오-비전 균형 |
| 팟캐스트 | 높음 | 음성 의존도 높아, 오디오 처리 능력이 병목 |

카테고리별 모델 성능 히트맵을 만들면 TCVP의 어떤 타입이 어려운지 진단할 수 있다.

**실험 3 — 2단계 학습 모델 (LongVALE 방법론 응용)**

현재 TCVP는 데이터셋만 있고 모델이 없다. LongVALE의 Boundary Perception → Instruction Tuning 패턴을 적용한다:

- Stage 1: 댓글 타임스탬프 + 모멘트 경계 데이터로 시간 구간 예측 학습
- Stage 2: 코멘트 기반 쿼리 + 모달리티 레이블로 쿼리 매칭 학습

현재 임베딩 기반 검색(R@10 38~62%) 위에 파인튜닝된 MLLM을 추가하면 의미 있는 성능 향상을 기대할 수 있다.

### Tier 2: 강력한 추가 분석

**실험 4 — 위치 편향 히트맵 (Momentseeker 응용)**

Momentseeker가 발견한 위치 편향이 TCVP 쿼리에도 존재하는가? 댓글 타임스탬프를 영상 길이로 정규화해 [0~25%, 25~50%, 50~75%, 75~100%] 구간별로 모델 성능을 측정한다.

만약 후반부 성능이 전반부보다 낮다면 — 이는 생성 기반 모델에 구조적 한계가 있음을 TCVP 데이터로도 재현한 것이 된다.

**실험 5 — Zero-Shot 일반화 검증 (LongVALE 방법론 응용)**

TCVP로 학습한 모델을 다른 벤치마크에서 평가한다:
- QVHighlights (시각 중심)
- TVR (TV 기반, 시간 추론)
- CharadesSTA (행동 인식)

"TCVP 훈련이 다른 VMR 데이터셋에도 일반화되는가"를 증명하면 데이터셋의 연구 가치가 크게 높아진다.

**실험 6 — 오디오-비전 상관성 명시 분석 (LongVALE AVC 응용)**

현재 TCVP의 모달리티 게이팅은 이진 분류(비전 또는 오디오)다. LongVALE의 7가지 AVC 패턴을 500개 샘플에 수동 주석하고, AVC 타입이 모델 성능에 미치는 영향을 분석하면 더 깊은 인사이트를 제공한다.

<div class="ornament">· · ·</div>

## 결론: 세 논문이 가리키는 방향

세 논문을 함께 보면 VMR 분야가 어디로 가고 있는지 보인다.

**문제 설정이 바뀌고 있다.** 기존: "전문가가 중요하다고 정한 순간을 찾아라" → 현재: "실제 사용자가 원하는 순간을 찾아라." TCVP의 유튜브 댓글 접근이 이 전환을 구현한다.

**오디오는 더 이상 무시할 수 없다.** TCVP와 LongVALE 모두 사용자 관심의 45~50%가 오디오 기반임을 보인다. 현재 모델들은 ASR 텍스트를 통해서만 이를 잘 처리한다 — 즉, raw 오디오 처리 능력은 아직 미완성이다.

**최고 모델도 70%를 틀린다.** Momentseeker의 R@1 29.6%는 현재 기술의 한계를 냉정하게 보여준다. 특히 Global Level 추론(인과, 공간 관계)은 단순히 더 많은 데이터나 더 큰 모델로 해결되지 않을 가능성이 높다.

**컨텍스트 길이가 성능의 핵심 레버다.** 768 프레임 vs 96 프레임 = 80% 성능 차이. 긴 비디오를 어떻게 효율적으로 처리할 것인가가 앞으로의 핵심 과제다.

<div class="footnote">
  참고 논문: TCVP (2025), LongVALE (NeurIPS 2024), Momentseeker (2025)
</div>
