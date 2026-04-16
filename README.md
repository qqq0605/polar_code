# Polar Code BSC Project

這是一個用 Python 撰寫的完整 polar code 專案，支援：

- `block length N = 2^k` 的任意長度
- 任意 `message length K`，只要 `1 <= K <= N`
- 通道模型為 `BSC (Binary Symmetric Channel)`
- `Bhattacharyya bound` 做 bit-channel construction
- `SC (Successive Cancellation)` decoder
- CLI、測試與可直接重現的模擬流程

## 專案結構

```text
.
├── README.md
├── pyproject.toml
├── src
│   └── polar_code
│       ├── __init__.py
│       ├── __main__.py
│       ├── channel.py
│       ├── cli.py
│       ├── codec.py
│       ├── construction.py
│       └── utils.py
└── tests
    └── test_codec.py
```

## 安裝與執行

如果要安裝成 CLI：

```bash
python3 -m pip install -e .
```

也可以不安裝，直接用 module 執行：

```bash
PYTHONPATH=src python3 -m polar_code --help
```

## CLI 用法

### 1. 編碼

```bash
PYTHONPATH=src python3 -m polar_code encode \
  --block-length 8 \
  --message-length 4 \
  --crossover-probability 0.11 \
  --message 1011
```

### 2. 解碼

```bash
PYTHONPATH=src python3 -m polar_code decode \
  --block-length 8 \
  --message-length 4 \
  --crossover-probability 0.11 \
  --received 11010010
```

### 3. 模擬

```bash
PYTHONPATH=src python3 -m polar_code simulate \
  --block-length 128 \
  --message-length 64 \
  --crossover-probability 0.05 \
  --trials 1000 \
  --seed 7
```

### 4. 產生 BER/FER 圖

```bash
PYTHONPATH=src python3 -m polar_code plot \
  --output-dir outputs \
  --trials 2000 \
  --seed 7 \
  --crossover-probability 0.05
```

會輸出四個檔案：

- `outputs/ber_fer_vs_n.svg`
- `outputs/ber_fer_vs_n.csv`
- `outputs/ber_fer_vs_code_rate.svg`
- `outputs/ber_fer_vs_code_rate.csv`

### 5. 搜尋滿足 FER 條件的最大 message length

```bash
PYTHONPATH=src python3 -m polar_code find-message-length \
  --block-length 65536 \
  --crossover-probability 0.01 \
  --trials 100 \
  --target-frame-error-rate 0.01 \
  --seed 20260416
```

這個指令會自動找出最大的 `K`，使得模擬得到的 frame error rate 不超過指定門檻。

## Python API 範例

```python
from polar_code import PolarCode, bsc_transmit

codec = PolarCode(block_length=8, message_length=4, crossover_probability=0.11)
message = [1, 0, 1, 1]
codeword = codec.encode(message)
received = bsc_transmit(codeword, 0.11)
decoded = codec.decode(received)

print(codec.info_set)
print(codeword)
print(decoded.estimated_message)
```

## 設計說明

- Construction:
  預設使用 Monte Carlo Bhattacharyya construction。對 all-zero transmission
  取樣，沿著 SC LLR recursion 估計每個 synthetic bit-channel 的
  `Z_i = E[exp(-L_i / 2)]`，再挑選 `K` 個最小的通道作為 information bits。
- Encoder:
  使用 Arikan 的 recursive polar transform。
- Decoder:
  使用 SC decoder，輸入為 BSC 對應的 LLR：
  `log((1-p)/p)` 或其相反號。
- Frozen bits:
  預設全部固定為 `0`。

## 測試

```bash
python3 -m unittest discover -s tests -v
```
