# KV Cache (Key-Value Cache)

## 핵심 아이디어
LLM의 자기회귀(Autoregressive) 생성 과정에서 이미 계산된 이전 토큰들의 Key와 Value 벡터를 GPU VRAM에 캐싱하여, 중복 행렬 곱셈 연산을 제거하고 추론 시간 복잡도를 $O(N^2)$에서 $O(N)$으로 낮추는 핵심 최적화 기법입니다.

---

## 수식

### 1. 일반 Attention 연산
길이 $t$의 입력 시퀀스에 대해 Attention 출력은 다음과 같습니다:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

### 2. Autoregressive 디코딩 시 새로운 토큰 $t+1$ 생성
새로운 토큰 생성 시에는 현재 토큰의 Query $q_{t+1}$ 하나만 계산하고, 이전 Key/Value에 새로운 $k_{t+1}, v_{t+1}$을 이어 붙입니다:

$$K_{\text{new}} = [K_{\text{prev}} \,;\, k_{t+1}], \quad V_{\text{new}} = [V_{\text{prev}} \,;\, v_{t+1}]$$

$$\text{Attention}(q_{t+1}, K_{\text{new}}, V_{\text{new}}) = \text{softmax}\left(\frac{q_{t+1} K_{\text{new}}^T}{\sqrt{d_k}}\right) V_{\text{new}}$$

### 3. 메모리 사용량 공식
$$\text{Memory}_{\text{KV}} = 2 \times B \times L \times H \times S \times D \times \text{bytes per element}$$
* $B$: 배치 크기, $L$: 레이어 수, $H$: KV 헤드 수, $S$: 시퀀스 길이, $D$: 헤드 차원

---

## 직관적 설명
책을 한 문장씩 이어 쓸 때, 지금까지 쓴 모든 문장을 처음부터 다시 소리 내어 읽으며 다음 단어를 고민하는 대신, '지금까지 읽은 내용의 요점 카드(Key-Value)'를 책상 위에 올려두고 새로 추가된 한 단어만 카드에 꽂아 넣으며 빠르게 다음 문장을 써 내려가는 방식입니다.

---

## 연결 개념
- [[transformer]] : Transformer 디코더 추론 시 필수적인 메모리 구조
- [[streamkv]] : 긴 컨텍스트 및 비디오 처리 시 폭증하는 KV Cache를 압축/선택하는 확장 기법
- [[pipeline]] : GPU 연산 파이프라인 및 메모리 대역폭(Memory Bandwidth) 최적화
- [[cache]] : 컴퓨터 시스템 구조의 메모리 계층 및 캐싱 원리

---

## 참고
- vLLM: Efficient Memory Management for Large Language Models with PagedAttention (SOSP 2023)
- Fast Inference from Transformers via KV Caching
