# Transformer

카테고리: Architecture  
자신감: ⭐⭐⭐⭐ (심화)  
마지막 복습: 2026-08-09


## 한 문장 요약

"Attention Is All You Need" — RNN 없이 Self-Attention만으로 시퀀스를 처리하는 아키텍처.


## 핵심 구조

```
Input → [Embedding + Positional Encoding]
      → [Multi-Head Self-Attention]
      → [Add & Norm]
      → [Feed-Forward Network]
      → [Add & Norm]
      → (N번 반복)
      → Output
```


## Attention 메커니즘

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- Q (Query): 현재 위치의 representation
- K (Key): 다른 위치와 비교할 representation  
- V (Value): 실제 가져올 정보
- $\sqrt{d_k}$: gradient vanishing 방지 스케일링

### Multi-Head Attention

여러 관점에서 동시에 attention:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(head_1, ..., head_h)W^O$$

$$head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$


## 왜 RNN보다 좋은가?

| | RNN | Transformer |
|---|---|---|
| 병렬화 | ❌ (순차) | ✅ (동시) |
| Long-range dependency | 취약 | 강력 (전역 attention) |
| 학습 속도 | 느림 | 빠름 |
| 메모리 | O(n) | O(n²) |


## Positional Encoding

RNN과 달리 순서 정보가 없어서 직접 주입:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$


## 파생 모델들

- Encoder only: BERT (양방향 이해)
- Decoder only: GPT (단방향 생성)
- Encoder-Decoder: T5, BART (번역, 요약)
- Vision: ViT (이미지 패치 → 토큰)


## 체크리스트

- [x] Q, K, V 역할 설명
- [x] Attention score 계산 및 스케일링 이유
- [x] Multi-head의 의미
- [x] Positional encoding 필요한 이유
- [x] Encoder-only vs Decoder-only 차이
- [ ] Flash Attention 원리 설명
- [ ] Grouped Query Attention (GQA) 설명
