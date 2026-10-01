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

커널의 `B`, `BN`, `NB`(chunk당 block 수)는 컴파일 특수화 값이다. host API는 입력 view의 B와 tile 설정을 받아 전달한다. 기본 tile은 `DEFAULT_GEMV_TILE`(16, 4)이다.

## host API

```rust
pub struct Q8Matrix { qs: Arc<Tensor<i8>>, scales: Arc<Tensor<f16>>, rows: usize, cols: usize }

impl HostQ8Matrix {
    pub fn from_gguf_q8_0(blocks: &[u8], rows: usize, cols: usize) -> Result<Self>;
}

pub fn gemv(w: &Weight, x: &TensorView<f32>, y: Partition<&mut Tensor<f32>>, tile: GemvTile) -> Gemv;
```

가중치는 공유 참조, 출력은 소유권 이동이다. 반환된 `Gemv`는 `submit(&mut sink)`로 실행한다. `Sink` 구현이 eager 실행과 CUDA Graph 캡처를 해석하고, `Gemv::submit`이 가중치 형식을 한 번만 선택한다.

## 검증

- 정확성: CPU f64 reference와 상대 오차 비교. B ∈ {1, 2, 4, 8}. 무작위 가중치와 실제 GGUF 텐서.
- 성능: CUDA event 중앙값, JIT 제외. 실효 GB/s = `N·K·1.0625 / t`.
- 기준: GB10 floor 200 GB/s, candle `fast_mmvq` 88~96%, ds4 `matvec_q8` 76%.
- shape: 2560 × 10240 (GDN qkv), 6144 × 2560 (GDN out), 2560 × 248320 (output head). 표기는 K × N.

## v1 측정 (RTX 5080, peak 960 GB/s)

B = 1은 peak의 57~88%다. B = 8은 B = 1보다 1.5~1.8배 느리다. 가중치 read는 B와 무관하므로, 중간 tile `[B, BN, NB, 32]`가 B배로 커지는 것이 원인으로 추정된다.

## v2 탐색 결과 (채택 안 함)

v1b는 세션을 커널 안에서 순회해 중간 tile을 줄였고, v2는 activation을 int8로 양자화해 dp4a를 사용했다. 둘 다 B가 커질 때 가중치를 세션마다 다시 읽어 v1보다 느렸다. 양자화 자체도 추가 비용과 오차를 냈다. 측정값과 당시 구현은 [RTX 5080 측정](../research/measurements/q8-gemv-5080.md), 커밋 `f611715`에 남아 있다.

## 다음 후보

- split-K: N이 작은 shape에서 block 수가 모자라면.
- 큰 B(prefill): `mma`(tensor core). N축 8에서는 tileiras가 IMMA를 내지 않았다(5080, 64×64×64에서는 IMMA 확인).
