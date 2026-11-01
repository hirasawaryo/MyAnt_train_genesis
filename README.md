# Antロボット脚形状のベイズ最適化（Genesis 版）

4本脚ロボット「Ant」の**脚の長さを変えると、歩行はどこまで速くなるのか**を調べるプログラムです。

ある脚の長さで歩き方を強化学習し、その成績を見て次に試す脚の長さを決めます。この手順を繰り返して最適な形状を探します。

- **強化学習**: 物理シミュレータ [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) 上で、[rsl_rl](https://github.com/leggedrobotics/rsl_rl) の PPO を使って歩行を学習します。
- **形状探索**: 学習は1回に数分かかるため、少ない試行回数で最適解を探せるベイズ最適化を使います。実装は [BoTorch](https://botorch.org/) です。
- 重力は地球重力（9.81 m/s²）です。

Isaac Lab で作った元のプロジェクト（MyAnt_train）を Genesis に移植したものです。報酬、観測、PPO の設定は元のプロジェクトに合わせています。

## 探索する変数

Ant の脚は1本につき2つの部品からなります。2つの部品の長さを探索変数にし、4本の脚に共通の値を使います。

| 変数 | 意味 | 探索範囲 | デフォルト形状 |
|---|---|---|---|
| x1 | 脚（leg）: 胴体の付け根から関節までの長さ [m] | 0.1〜2.0 | 0.2828 |
| x2 | 足（foot）: 関節から足先までの長さ [m] | 0.3〜2.0 | 0.5657 |

1つの形状を次の手順で評価します。

1. 指定した長さの Ant モデル（MJCF）を生成する
2. PPO で歩行を学習する（4096 環境を並列、2500 イテレーション）
3. 学習中の `Mean reward`（平均報酬）の最大値を、その形状の成績とする

ベイズ最適化の設定は次のとおりです。

- 目的関数: `y = -最大平均報酬`。BoTorch では最小化問題として扱うため、符号を反転します。
- 観測ノイズ: 学習ログの最後の 100 イテレーションの `Mean reward` の分散を、`train_Yvar` として与えます。
- サロゲートモデル: ガウス過程（`SingleTaskGP`）
- 獲得関数: LCB（β=2.0）

## 動作環境

- Linux と NVIDIA GPU（CUDA）。4096 環境の学習で GPU メモリを数 GB 使います。
- Python 3.11
- 所要時間の目安: 1 形状あたり約 7〜8 分。30 回の評価で約 4 時間です。

## セットアップ

```bash
git clone <このリポジトリのURL>
cd MyAnt_train_genesis

python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`bayes_opt.py` は内部で `.venv/bin/python` を使って `train.py` と `play.py` を呼びます。そのため、仮想環境は必ずリポジトリ直下に `.venv` という名前で作ってください。[uv](https://docs.astral.sh/uv/) を使う場合は、次のコマンドでも同じ環境ができます。

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## 使い方

コマンドはすべてリポジトリ直下で実行してください。出力先が相対パスになっているためです。

### 1. まず動作確認（1 形状だけ学習する）

```bash
# 学習（--max_iterations を小さくすると数十秒で終わります）
.venv/bin/python train.py --leg_length 0.954 --foot_length 0.563 --max_iterations 100 --log_dir runs/test

# 歩行動画を保存（8 秒）
.venv/bin/python play.py --leg_length 0.954 --foot_length 0.563 \
  --checkpoint runs/test/model_final.pt --video videos/test.mp4

# 学習した方策の評価（256 環境 × 15 秒、結果を JSON に保存）
.venv/bin/python evaluate.py --leg_length 0.954 --foot_length 0.563 \
  --checkpoint runs/test/model_final.pt --out eval/test.json
```

`train.py` は学習ログに `Mean reward` を出力し、最後に `<log_dir>/model_final.pt` を保存します。

### 2. ベイズ最適化を実行する

最初の 5 点はデフォルト形状と探索範囲の四隅にします。その後、LCB で 25 点を提案し、計 30 回評価します。

```bash
.venv/bin/python -u bayes_opt.py run-loop --observations Earth \
  --lower-bounds 0.1 0.3 --upper-bounds 2.0 2.0 \
  --n-initial 5 --initial-design specified \
  --initial-point 0.2828427255153656 0.5656854510307312 \
  --initial-point 0.1 0.3 --initial-point 0.1 2.0 --initial-point 2.0 0.3 --initial-point 2.0 2.0 \
  --beta 2.0 --max-evals 30
```

`--initial-design random` を指定すると、初期点を Sobol 列でランダムに選びます。この場合 `--initial-point` は不要です。

### 3. 途中で止めた場合・続きを探索する場合

過去の観測 CSV を `--prior-observations` で事前データとして渡すと、その続きから探索できます。

```bash
.venv/bin/python -u bayes_opt.py run-loop --observations Earth_continued \
  --prior-observations result/Earth_<日時>.csv \
  --lower-bounds 0.1 0.3 --upper-bounds 2.0 2.0 --n-initial 5 --beta 2.0 --max-evals 10
```

観測済みデータの中の最良点は、次のコマンドで表示できます。

```bash
.venv/bin/python bayes_opt.py best --observations result/Earth_<日時>.csv
```

## 出力ファイル

ベイズ最適化を実行すると、次のファイルが作られます。これらは `.gitignore` によってリポジトリには含めません。

| パス | 内容 |
|---|---|
| `result/Earth_<日時>.csv` | 観測データ（Number, Time, x1, x2, y, y_var）。`y` は最大平均報酬の符号を反転した値 |
| `BO_graph/<日時>/acquisition_*.png` | 各ステップの 2 次元探索グラフ（観測点、事後平均、事後標準偏差、獲得関数。★は次の候補） |
| `videos/stepNN_leg*_foot*.mp4` | 各形状の歩行動画（8 秒） |
| `runs/stepNN_<日時>/` | 各評価の学習済みモデル（`model_final.pt`）と TensorBoard ログ |
| `log_temp/log_<日時>_N.txt` | 各評価の学習ログ。目的値の `Mean reward` はここから読み取ります |

学習曲線は TensorBoard で確認できます。

```bash
.venv/bin/tensorboard --logdir runs
```

## ファイル構成

| ファイル | 役割 |
|---|---|
| `bayes_opt.py` | ベイズ最適化の本体（CLI）。`black_box_function()` が形状ごとに `train.py` と `play.py` を実行し、学習ログから目的値を読み取ります |
| `ant_model.py` | 脚と足の長さを指定して Ant の MJCF（XML）を生成します。胴体は半径 0.25 の球、脚は半径 0.08 のカプセルです |
| `ant_env.py` | Genesis 上の並列学習環境です。報酬、観測（36 次元）、行動（8 関節のトルク）、終了条件を定義します |
| `train.py` | PPO による歩行の学習（rsl_rl） |
| `play.py` | 学習済み方策で歩行させ、ロボットを追うカメラで mp4 を保存します |
| `evaluate.py` | 学習済み方策の評価。初期関節角に ±0.1 rad のノイズを加えた 256 環境で 15 秒歩かせ、平均リターン、前進距離、速度、転倒率を JSON に出力します |
| `requirements.txt` | 依存ライブラリ |

### 主なパラメータ

| 場所 | 項目 | 値 |
|---|---|---|
| `bayes_opt.py` | `NUM_ENVS` / `MAX_ITERATIONS` | 4096 / 2500（1 形状あたりの学習量） |
| `bayes_opt.py` | `VIDEO_LENGTH` | 480 ステップ（8 秒） |
| `ant_env.py` | 制御周期 / エピソード長 | 60 Hz（物理 120 Hz、decimation 2） / 15 秒 |
| `ant_env.py` | 転倒判定 | 胴体の高さが 0.25 m 未満 |
| `train.py` | PPO | 方策 [32, 32] elu、学習率 1e-3（adaptive）、1 環境あたり 16 ステップ収集 |

## 結果の例

30 回の評価で見つかった最良形状は、**脚 0.954 m・足 0.563 m** でした。デフォルト形状に比べて脚が約 3.4 倍長くなっています。

最良形状とデフォルト形状を乱数シード 3 通りで学習し直し、`evaluate.py` で評価した平均値は次のとおりです。

| | デフォルト（脚 0.283 / 足 0.566） | 最良形状（脚 0.954 / 足 0.563） |
|---|---|---|
| 学習中の最大平均報酬 | 10,619 | 12,240（+15%） |
| 歩行速度 | 5.92 m/s | 6.65 m/s（+12%） |
| 15 秒間の前進距離 | 88.7 m | 99.6 m |
| 転倒率 | 5.9% | 1.3% |

Genesis の物理演算と乱数の違いがあるため、同じコマンドでも結果の数値は少し変わることがあります。
