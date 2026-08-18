# StreamKV & Video KV Compression

## 핵심 아이디어
실시간 스트리밍 비디오 또는 초장거리 컨텍스트를 처리하는 Large Multimodal Model에서 매 프레임 누적되는 KV Cache의 메모리 폭증을 방지하기 위해, 주의 집중도(Attention Score)와 시간적 지역성(Temporal Locality)에 기반하여 중요한 Key-Value 토큰만 동적으로 유지하고 나머지를 압축/퇴출(Eviction)하는 기법입니다.

---

## 수식 및 알고리즘

### 1. KV 토큰 중요도 평가 (Attention Budget)
시점 $t$에서 각 과거 토큰 $i$에 대한 누적 어텐션 가중치:
$$S_i = \sum_{\tau=t-W}^{t} \sum_{h=1}^{H} A_{h, \tau, i}$$

### 2. 동적 KV Eviction 전략
고정된 버짓 $K_{\text{budget}}$ 내에서 최적의 토큰 집합 $\mathcal{S}^*$ 유지:
$$\mathcal{S}^* = \text{TopK}(S_i, K_{\text{budget}} - K_{\text{recent}}) \cup \mathcal{S}_{\text{recent}}$$
- $\mathcal{S}_{\text{recent}}$: 최근 $W$개의 프레임 토큰 (지역성 보장)
- $\text{TopK}$: 글로벌하게 높은 중요도를 유지하는 앵커 토큰(Sink Tokens)

### 3. 메모리 복잡도
$$O(T \cdot d) \longrightarrow O(K_{\text{budget}} \cdot d) \quad (\text{시간 } T \text{에 독립적인 상수 메모리})$$

---

## 직관적 설명
영화를 보면서 모든 1초 1초의 장면을 머릿속에 다 외우려고 하면 뇌 용량이 초과됩니다. 대신 '줄거리의 결정적 복선과 주요 장면(앵커)'과 '방금 전 5초 동안 일어난 일(최근)'만 선별해서 기억하며 영화를 끝까지 쾌적하게 감상하는 지혜입니다.

---

## 연결 개념
- [[kv_cache]] : 기본 Transformer KV Cache 구조의 직접적 압축 확장
- [[long_video_understanding]] : 초장거리 비디오 스트리밍 처리의 핵심 인프라
- [[cache]] : 시스템의 LRU/LFU 캐시 교체 정책과의 수학적 연결

---

## 참고
- StreamKV: Towards Streaming Long-Context Video LLMs (ICLR 2025)
- StreamingLLM: Efficient Streaming Language Models with Attention Sinks
