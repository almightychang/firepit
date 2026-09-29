# Q8_0 GEMV (safe cuTile)

첫 커널. design.md 9절 열린 질문 4번("safe cuTile 양자화 GEMV가 GB10 floor의 몇 %인가")에 답한다.

## 결정

- activation은 f32. ds4와 같고, activation 바이트는 가중치 대비 무시할 수준이다.
- batch 차원을 처음부터 포함한다. `x: [B, K]`, `y: [B, N]`. 가중치 tile을 한 번 읽어 세션 B개가 공유한다.
- `unsafe`, `unchecked_accesses`, raw pointer를 쓰지 않는다. 필요해지면 멈추고 공백으로 기록한다.
- cuTile은 `=0.4.0`에 고정한다.

## layout

GGUF Q8_0 block은 `{ d: f16, qs: [i8; 32] }` 34 B다. 로드할 때 SoA로 재배치한다.

```
qs : Tensor<i8>  [N, K/32, 32]
d  : Tensor<f16> [N, K/32]
```

32개 block을 축 하나로 드러낸다. tile 타입에 `BK / 32` 같은 const 산술이 필요 없어진다(stable Rust와 cutile 매크로 모두 거부한다). 바이트 수는 원소당 1.0625 B로 같다. GGUF block을 그대로 읽는 버전은 공백 후보 3번 확인용으로 나중에 따로 시험한다.

## 커널 v1

```
grid = N / BN, block j가 y[:, j·BN ..]를 소유 (partition [B, BN])

x는 [B, K/32, 32]로 본다.

acc[B, BN] = 0
for kc in 0..(K/32)/NB:
    q    = qs[j·BN.., kc·NB.., :]  [BN, NB, 32] i8 → f32
    s    = d [j·BN.., kc·NB..]     [BN, NB] f16 → f32
    xb   = x [:, kc·NB.., :]       [B, NB, 32]
    part = Σ_last(q · xb)          [B, BN, NB]      (broadcast [B,1,NB,32] × [1,BN,NB,32])
    acc += Σ_last(part · s)        [B, BN]
y.store(acc)
```

`B`, `BN`, `NB`(chunk당 block 수)는 const generic이다. 값은 bench로 고른다.

## host API

```rust
pub struct Q8Matrix { qs: Arc<Tensor<i8>>, scales: Arc<Tensor<f16>>, rows: usize, cols: usize }

impl HostQ8Matrix {
    pub fn from_gguf_q8_0(blocks: &[u8], rows: usize, cols: usize) -> Result<Self>;
}

pub fn q8_gemv(w: &Q8Matrix, x: &Tensor<f32>, y: Partition<Tensor<f32>>) -> impl DeviceOp;
```

가중치는 공유 참조, 출력은 소유권 이동이다. step graph와 CUDA Graph에 넣을 수 있도록 `DeviceOp`을 반환한다.

## 검증

- 정확성: CPU f64 reference와 상대 오차 비교. B ∈ {1, 2, 4, 8}. 무작위 가중치와 실제 GGUF 텐서.
- 성능: CUDA event 중앙값, JIT 제외. 실효 GB/s = `N·K·1.0625 / t`.
- 기준: GB10 floor 200 GB/s, candle `fast_mmvq` 88~96%, ds4 `matvec_q8` 76%.
- shape: 2560 × 10240 (GDN qkv), 6144 × 2560 (GDN out), 2560 × 248320 (output head). 표기는 K × N.

## v1 측정 (RTX 5080, peak 960 GB/s)

B = 1은 peak의 57~88%다. B = 8은 B = 1보다 1.5~1.8배 느리다. 가중치 read는 B와 무관하므로, 중간 tile `[B, BN, NB, 32]`가 B배로 커지는 것이 원인으로 추정된다.

## 커널 v2

v1과의 차이는 두 가지다.

1. 세션 루프: block 안에서 세션을 하나씩 돈다. 중간 tile이 `[BN, NB, 32]`로 B와 무관해진다. 가중치 tile은 루프 밖에서 한 번 load한다.
2. int8 activation: x를 32개 단위로 int8 양자화한다(`xq`, `dx`). i8 → i32 확장 → 곱 → 합으로 적으면 tileiras가 `IDP.4A`(dp4a)로 내린다(5080에서 SASS로 확인). candle `fast_mmvq`, llama.cpp `mmvq`와 같은 방식이다.

```
quantize_x:  x [B, K] f32  →  xq [B, K/32, 32] i8,  dx [B, K/32] f32     (block마다 max|x| / 127)

q8_gemv_v2:
grid = N / BN, block j가 y[:, j·BN ..]를 소유 (partition [B, BN])
y 안에서 행 단위 partition [1, BN]으로 세션 b의 결과를 쓴다 (v0.4.0 safe partition_mut)

for b in 0..B:
    acc[BN] = 0
    for kc in 0..(K/32)/NB:
        q    = qs[j·BN.., kc·NB.., :]   [BN, NB, 32] i8 → i32
        s    = d [j·BN.., kc·NB..]      [BN, NB] f16 → f32
        xb   = xq[b, kc·NB.., :]        [NB, 32] i8 → i32 → broadcast [BN, NB, 32]
        dxb  = dx[b, kc·NB..]           [NB] → broadcast [BN, NB]
        dot  = Σ_last(q · xb)           [BN, NB] i32        ← dp4a
        acc += Σ_last(f32(dot) · s · dxb)  [BN]
    y_rows[b].store(acc)
```

가중치 tile을 세션 루프 밖으로 빼는 것은 k 루프와 순서가 충돌한다. 먼저 위 순서(세션 바깥, k 안쪽)로 짜고, 같은 가중치를 B번 읽는 비용이 L2에서 흡수되는지 측정한다. 흡수되지 않으면 k 바깥, 세션 안쪽으로 바꾸고 acc를 세션별로 둔다.

### 비교 변형

추정을 가르기 위해 세 가지를 같은 bench로 잰다.

| 변형 | 세션 루프 | activation | 검증하는 것 |
|---|---|---|---|
| v1 | 없음 (broadcast) | f32 | 기준 |
| v1b | 있음 | f32 | 중간 tile 크기가 B 의존성의 원인인가 |
| v2 | 있음 | int8 (dp4a) | dp4a가 추가로 얼마나 버는가 |

### 정밀도

- 정확성 테스트는 같은 양자화를 적용한 CPU reference와 비교한다. 정수 내적이므로 오차는 f32 누산 수준이어야 한다.
- 양자화 자체의 오차는 f32 x 기준 CPU reference와 비교해 따로 보고한다.

## 다음 후보

- split-K: N이 작은 shape에서 block 수가 모자라면.
- 큰 B(prefill): `mma`(tensor core). N축 8에서는 tileiras가 IMMA를 내지 않았다(5080, 64×64×64에서는 IMMA 확인).
