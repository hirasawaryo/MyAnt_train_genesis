# MyAnt_train_genesis

`~/MyAnt_train`（Isaac Lab + BoTorch）で行っていた「Ant ロボットの脚形状のベイズ最適化」を、
物理シミュレータ **Genesis** で再実装したプロジェクトです。重力は地球（-9.81 m/s²）のみ。

- x1 = 脚（leg）の長さ [m]、探索範囲 0.1〜2.0（デフォルト 0.2828）
- x2 = 足（foot）の長さ [m]、探索範囲 0.3〜2.0（デフォルト 0.5657）
- 目的関数: rsl_rl PPO 学習中の `Mean reward` の最大値（最小化のため `y = -MaxMeanReward`）
- 観測分散: 学習ログ最後の100イテレーションの `Mean reward` の分散（`y_var`, BoTorch の `train_Yvar`）
- 獲得関数: LCB (β=2.0)、ガウス過程 `SingleTaskGP`

## ファイル構成

| ファイル | 役割 | MyAnt_train での対応 |
|---|---|---|
| `ant_model.py` | 脚・足の長さを指定して Ant の MJCF を生成（nv_ant 準拠: 球胴体 r=0.25, カプセル r=0.08, 密度5, ギア15） | `MyAnt.usda` の行番号書き換え |
| `ant_env.py` | Genesis 版の環境。Isaac Lab `LocomotionEnv` + `MyantTrainEnvCfg` と同じ報酬・観測(36次元)・行動(8次元トルク)・終了条件 | `tasks/direct/myant_train/` |
| `train.py` | rsl_rl PPO 学習（`rsl_rl_ppo_cfg.py` と同じハイパーパラメータ） | `scripts/rsl_rl/train.py` |
| `play.py` | 学習済み方策で歩行動画（mp4）を保存 | `scripts/rsl_rl/play.py --video` |
| `evaluate.py` | 学習済み方策の決定論的評価（256環境、初期関節角に ±0.1 rad ノイズ、15 s） | なし（追加） |
| `bayes_opt.py` | BoTorch ベイズ最適化 CLI。`BoTorch_Project/src/.../cli.py` をコピーし `black_box_function()` を Genesis 用に置換 | `BoTorch_Project/bayes_opt.py` |
| `analyze.py` | レポート用の集計グラフ生成 | なし（追加） |

## 出力

| パス | 内容 |
|---|---|
| `result/Earth_*.csv` | BO 観測データ（Number, Time, x1, x2, y, y_var） |
| `BO_graph/<日時>/acquisition_*.png` | 各ステップの 2 次元探索グラフ（観測点 / 事後平均 / 事後標準偏差 / 獲得関数＋次点★） |
| `videos/stepNN_leg*_foot*.mp4` | 探索中の各形状の歩行動画（8 秒） |
| `runs/stepNN_*/` | 各評価の学習済みモデル・TensorBoard ログ |
| `log_temp/` | 各評価の学習ログ（`Mean reward` をここから読む） |
| `eval/bo_stepNN.json` | 各評価の方策の決定論的評価結果 |
| `verify/` | 最良形状とデフォルト形状の再学習（seed 1〜3） |
| `final/` | 最終提案の歩行動画（15 秒）と比較動画 |
| `report/` | レポートと図 |

## 実行方法

```bash
# 環境（作成済み）
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch genesis-world rsl-rl-lib==3.0.1 botorch gpytorch matplotlib tensorboard imageio imageio-ffmpeg

# ベイズ最適化（今回の実行コマンド）
.venv/bin/python bayes_opt.py run-loop --observations Earth \
  --lower-bounds 0.1 0.3 --upper-bounds 2.0 2.0 \
  --n-initial 5 --initial-design specified \
  --initial-point 0.2828427255153656 0.5656854510307312 \
  --initial-point 0.1 0.3 --initial-point 0.1 2.0 --initial-point 2.0 0.3 --initial-point 2.0 2.0 \
  --beta 2.0 --max-evals 30

# 続きから（過去CSVを事前データに）
.venv/bin/python bayes_opt.py run-loop --observations Earth_continued \
  --prior-observations result/Earth_20261101235702.csv \
  --lower-bounds 0.1 0.3 --upper-bounds 2.0 2.0 --n-initial 5 --beta 2.0 --max-evals 10

# 単体の学習・動画・評価
.venv/bin/python train.py --leg_length 0.954 --foot_length 0.563 --log_dir runs/manual
.venv/bin/python play.py --leg_length 0.954 --foot_length 0.563 --checkpoint runs/manual/model_final.pt --video final/manual.mp4
.venv/bin/python evaluate.py --leg_length 0.954 --foot_length 0.563 --checkpoint runs/manual/model_final.pt --out eval/manual.json
```

## MyAnt_train からの変更点

- 学習量: 1 評価あたり 2500 イテレーション × 4096 環境（約 7 分）。Isaac Lab 版は 30000 × 512。
- 初期姿勢の高さ: Isaac Lab 版は 3.0 m から落下。こちらは足が地面に触れない高さ `max(0.75, 0.707·x2 + 0.25)` から開始。
- 地面の摩擦: Isaac Lab 版は 10.0（average 合成）。Genesis では 1.0。
- `y_var` の計算: MyAnt_train は `MeanRewards[:-100]`（最後の100件**以外**）になっていたため、コメント通り `MeanRewards[-100:]` に修正。
- GPU メモリは llama-server と共有（空き約 5 GB）のため、学習は 1 本ずつ逐次実行。
