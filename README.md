# Tai lieu file va vai tro trong du an ATCS

Tai lieu nay map nhanh: file nao lam gi, file nao plot KPI, file nao so sanh KPI RL vs Fixed-Time.

## 1) File plot KPI va so sanh KPI (ban dang hoi)

| Muc dich | File chinh | Ham/chuc nang plot | Dau ra |
|---|---|---|---|
| Plot KPI tong hop RL vs Fixed-Time (3 KPI: delay, queue, saturation norm) cho tung scenario | `ATCS/compare_kpi.py` | `plot_comparison(...)` | `checkpoints/compare_kpi/{scenario}_compare_kpi3_satnorm.png` |
| Plot KPI tong hop RL vs Fixed-Time cho nhieu scenario tren cung 1 hinh | `ATCS/compare_kpi.py` | `plot_multi_scenario_3kpi(...)` | `checkpoints/compare_kpi/kpi3_compare_{...}_rl_vs_fixed_satnorm.png` |
| Xuat bang tong hop KPI improve (%) | `ATCS/compare_kpi.py` | `save_summary_csv(...)` | `checkpoints/compare_kpi/kpi3_satnorm_summary.csv` |
| So sanh chi tiet theo node/lane (CSV) | `ATCS/compare_kpi.py` | `save_detailed_comparison_csvs(...)` | `checkpoints/compare_kpi/detailed/{scenario}/*_comparison.csv` |
| Plot so sanh KPI theo tung node (tung nut giao thong) | `ATCS/compare_node_kpi.py` | `plot_comparison_per_node(...)` | `checkpoints/compare_kpi/per_node/{scenario}_node_{Jx}_compare_kpi.png` |
| CSV tong hop theo node | `ATCS/compare_node_kpi.py` | `save_summary_csv(...)` | `checkpoints/compare_kpi/per_node/node_kpi_summary.csv` |
| Plot reward trong qua trinh train | `ATCS/train.py` | Ve va `plt.savefig(plot_file)` moi episode | `checkpoints/{scenario}/{scenario}_reward_plot.png` |
| Plot critic loss theo cac scenario | `ATCS/plot_critic_loss.py` | script top-level (khong dong goi ham) | `checkpoints/critic_loss_comparison.png` |

Luu y:
- `ATCS/compare_kpi.py` la file chinh de "plot so sanh KPI".
- `ATCS/compare_node_kpi.py` la file chinh de "plot so sanh KPI theo tung node".
- KPI goc duoc tinh trong `ATCS/atcs/kpi_engine.py`, sau do duoc day ra obs/reward trong `ATCS/atcs/environment.py`.

## 2) Luong du lieu KPI trong he thong

1. `ATCS/atcs/environment.py` lay du lieu xe theo lane tu SUMO/TraCI.
2. `ATCS/atcs/kpi_engine.py` cap nhat queue accumulation + tinh KPI lane (`control_delay`, `degree_of_saturation`, `queue_length`).
3. Environment build:
- Observation tensor cho actor.
- Reward tensor (lower is better, trainer xu ly thanh scalar reward co trong so).
4. Script danh gia/so sanh (`evaluate.py`, `compare_kpi.py`, `compare_node_kpi.py`) doc obs/reward de tinh trung binh va ve bieu do.

## 3) Vai tro cac thu muc va file chinh

### Thu muc `ATCS/` (code chinh)

| File/Thu muc | Vai tro |
|---|---|
| `ATCS/train.py` | Entry-point train ACAC, ghi training log CSV, luu checkpoint, plot reward theo episode. |
| `ATCS/evaluate.py` | Evaluate 1 checkpoint tren 1 scenario, in metric tong hop (reward, delay, queue, saturation norm). |
| `ATCS/compare_kpi.py` | Evaluate RL vs Fixed-Time tren danh sach scenario (tu `scenario_config.json`), plot 3 KPI, xuat summary CSV va detailed CSV. |
| `ATCS/compare_node_kpi.py` | So sanh RL vs Fixed-Time theo tung node, ve tung bieu do theo node, xuat summary node CSV. |
| `ATCS/plot_critic_loss.py` | Doc training log CSV de ve critic loss across scenarios. |
| `ATCS/benchmark_inference.py` | Benchmark thoi gian suy luan cua time encoder, actor, critic, full pipeline. |
| `ATCS/run_test.py` | Script test nhanh/noi bo (duong dan hard-code), khong phai luong chay production. |
| `ATCS/verify_acac.py`, `ATCS/ppo_pendulum.py` | Script verify/toy RL (pendulum), khong phai luong KPI giao thong chinh. |
| `ATCS/tools/build_evaluate_scenarios.py` | Tool tao bo scenario evaluate (route/sumocfg/phase file) tu network SUMO. |
| `ATCS/examples/smoke_env.py` | Smoke test moi truong. |
| `ATCS/examples/fixed_time_baseline.py` | Chay baseline fixed-time de test env. |

### Thu muc `ATCS/atcs/` (mo phong + KPI engine)

| File | Vai tro |
|---|---|
| `ATCS/atcs/environment.py` | Môi truong RL (reset/step), quan ly phase TLS, goi KPI engine, tao obs/reward/info. |
| `ATCS/atcs/kpi_engine.py` | Logic tinh KPI theo lane (delay, saturation, queue), tich luy queue/inflow/outflow theo step. |
| `ATCS/atcs/sumo_parser.py` | Parse `.sumocfg`, `.net.xml`, chuong trinh den TLS, pha green/yellow/red. |
| `ATCS/atcs/config_loader.py` | Load `kpi_config.json` thanh dataclass type-safe. |
| `ATCS/atcs/__init__.py` | Package marker/export. |

### Thu muc `ATCS/acac/` (model RL)

| File | Vai tro |
|---|---|
| `ATCS/acac/networks.py` | Dinh nghia network: positional encoding, history encoder (GRU), actor, centralized critic. |
| `ATCS/acac/trainer.py` | Rollout, tinh advantage, update PPO-like objective, evaluate_model, save/load checkpoint. |
| `ATCS/acac/buffers.py` | Buffer bat dong bo/dong bo cho actor va critic trajectory. |
| `ATCS/acac/config_loader.py` | Load `model_config.json` va `scenario_config.json`. |
| `ATCS/acac/__init__.py` | Re-export thanh phan chinh de import gon. |

### Thu muc `ATCS/config/` (cau hinh)

| File | Vai tro |
|---|---|
| `ATCS/config/model_config.json` | Hyper-parameter model/training ACAC va reward weights. |
| `ATCS/config/scenario_config.json` | Registry scenario evaluate/compare (name, sumocfg, checkpoint, tags). |
| `ATCS/config/kpi_config.json` | Chuan KPI: simulation settings, cong thuc, constants, reward design, LOS. |

### Thu muc du lieu va artifact

| Thu muc | Vai tro |
|---|---|
| `SimulationData/` | Bo map SUMO, route, sumocfg cho train/evaluate/sample. |
| `checkpoints/` | Artifact chinh: checkpoint `.pt`, training log `.csv`, cac file plot KPI/reward. |
| `ATCS/checkpoints/` | Artifact cu/bo sung (moc lich su). Script hien tai uu tien `checkpoints/` o repo root. |
| `ITS-mess_branch/` | Nhanh code cu/tham khao (khong phai module chinh cua pipeline ATCS hien tai). |

## 4) Command nhanh de dung dung file

### So sanh KPI tong hop (RL vs Fixed-Time)
```bash
python ATCS/compare_kpi.py --steps 600
```

### So sanh KPI theo node
```bash
python ATCS/compare_node_kpi.py --steps 600
```

### Ve critic loss tu log train
```bash
python ATCS/plot_critic_loss.py
```

### Train (co plot reward)
```bash
python ATCS/train.py --sumocfg SimulationData/Evaluate/Normal/2Intersection/config.sumocfg --episodes 200 --steps 600
```

## 5) Neu ban can tim "file nao chua KPI nao"

- Delay/Queue/Saturation duoc tinh tai: `ATCS/atcs/kpi_engine.py`.
- KPI tong hop toan mang + cai tien % RL vs Fixed: `ATCS/compare_kpi.py`.
- KPI theo node (J1, J2, ...): `ATCS/compare_node_kpi.py`.
- Critic loss (khong phai KPI giao thong): `ATCS/plot_critic_loss.py`.

