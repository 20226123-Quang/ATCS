# ATCS Architecture (New Baseline)

## Muc tieu
- Tach kien truc moi de train RL tu file `.sumocfg`.
- Chuan hoa KPI theo TCCS 24:2018/TCDBVN trong 1 file config duy nhat.
- Co `TrafficEnvironment` voi `reset()` va `step(action)` dung format ban yeu cau.

## Cau truc thu muc
```text
ATCS/
├─ config/
│  └─ kpi_config.json            # Toan bo KPI, cong thuc, nguong LOS, hang so
├─ atcs/
│  ├─ __init__.py
│  ├─ config_loader.py           # Load + type-safe config
│  ├─ sumo_parser.py             # Parse .sumocfg/.net.xml, chuong trinh den
│  ├─ kpi_engine.py              # Input-Output accumulation + KPI formulas
│  └─ environment.py             # TrafficEnvironment (reset/step)
└─ examples/
   └─ smoke_env.py               # Chay nhanh de test env
```

## Interface moi truong
```python
from atcs.environment import TrafficEnvironment

env = TrafficEnvironment(sumocfg_path=".../config.sumocfg")
observation, reward, done, information = env.reset()
action = {"J1": 5.0, "J3": 3.0}  # extend green theo giay
observation, reward, done, information = env.step(action)
```

### Dinh dang output
- `observation`: tensor shape `[n_intersections, n_lanes_max, 5]`
  - Feature lane-level: `[control_delay_d, v_over_c, queue_length_L, remaining_cycle_time, current_phase]`
- `reward`: tensor shape `[n_intersections, n_lanes_max, 2]`
  - Feature lane-level: `[control_delay_d, queue_length_L]`
- `done`: `bool`
- `information`:
  - `min_green`
  - `max_green`
  - `cycle_length`
  - `delta_t`
  - `intersection_require_action`

## KPI va cong thuc
Tat ca dat tai:
- `ATCS/config/kpi_config.json`

Bao gom:
- `d = d1 * PF + d2 + d3`
- `v/c = V / C_cap`
- `C_cap = s * (g/C) * N`
- `L = Q_avg * S_L`
- `Q(t) = max(0, Q(t-1) + V_in(t) - V_out(t))`
- bang LOS A-F

## Smoke test
```bash
cd ATCS
python -m examples.smoke_env --sumocfg ../ITS-mess_branch/SimulationData/SampleData/Crowded/config.sumocfg --steps 3
```

## Ghi chu
- Env su dung TraCI va dieu khien den theo logic phase runtime.
- RL trainer PPO co the doc truc tiep tensor output de tao scalar loss/reward tuy theo chien luoc huan luyen.
