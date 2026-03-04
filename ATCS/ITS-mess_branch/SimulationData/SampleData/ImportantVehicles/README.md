# Leading-Car Traffic Light Override (SUMO + TraCI)

This controller forces intersections to show green only for the leading (closest) approaching vehicle's incoming lane, with all other signals red, for a fixed duration (default 100 seconds). After the hold period, signals resume their normal fixed-time program.

## Prerequisites
- SUMO installed and available in PATH.
- `SUMO_HOME` environment variable set to the SUMO installation directory, or adjust the script to locate `tools`.
- A SUMO network and configuration `.sumocfg` file.

## Run
From this folder:

```
python leading_car_tl.py -c <path-to-your.sumocfg> --gui --hold 100 --range 50
```

- `--gui` runs SUMO-GUI (omit for CLI).
- `--hold` is the override duration in seconds (default 100).
- `--range` is the detection distance to the stop line in meters (default 50).

## How it works
- At each simulation step, for each TLS, the script finds the closest vehicle on any incoming lane.
- If the vehicle is within `--range` of the stop line, the TLS is overridden: only movements from that vehicle's incoming lane are set to green (`G`), others red (`r`).
- The override is held for `--hold` seconds, then released, letting the normal fixed-time program continue.

## Notes
- The mapping from lanes to signal indices uses `traci.trafficlight.getControlledLinks`. All movements (left/straight/right) from the selected incoming lane will be allowed.
- Ensure your TLS are defined with proper controlled links so the mapping is correct.
- If no controlled link for the chosen lane is found, the script skips overriding.
