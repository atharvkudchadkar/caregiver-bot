# Navigation verification

The evaluator checks the actual robot position, balance state and non-floor
contacts independently of the navigator. No evaluator geometry or contact data
is supplied to navigation. Reports include camera images, learned maps,
trajectories and terminal explanations.

| Scenario | Simulated time | Actual destination reached | Non-floor contacts | Falls |
|---|---:|---|---:|---:|
| Fresh learning, bedroom to bathroom, including inside scan | 57.5 s | Yes | 0 | 0 |
| Saved bathroom map, shifted bedroom starting position | 25.2 s | Yes | 0 | 0 |
| Restart inside bathroom, return to remembered bedroom | 41.1 s | Yes | 0 | 0 |
| Fresh exploration to kitchen | 156.9 s | Yes | 0 | 0 |
| Saved learned routes to kitchen, including inside scan | 74.3 s | Yes | 0 | 0 |

Recorded reports:

- [Fresh bathroom](verification/bathroom_final/report.json)
- [Shifted-start bathroom](verification/bathroom_reloaded/report.json)
- [Return from bathroom](verification/return_from_bathroom/report.json)
- [Fresh kitchen](verification/kitchen_fresh/report.json)
- [Kitchen using traveled memory](verification/kitchen_traveled_memory/report.json)
- [Doorway signs as rendered](verification/doorway_signs.png)

Timings depend on starting position, existing memory and whether an arrival scan
is needed. These are bounded scenario checks, not proof for arbitrary buildings.

The regression suite covers memory persistence, relocalization at a changed
start, continuing navigation without visible codes, English recognition
independent of code ID, rejecting obscured INSIDE/ENTRANCE text, camera
articulation, live obstacle checks, unknown-space clearance, and revisiting
learned routes without treating old motion as permission to cross new obstacles.

```powershell
python -m unittest discover -s tests -v
python verify_navigation.py --room kitchen --seconds 180 --memory memory/robot_map.npz --output verification/my_run
```

The evaluator exits nonzero if the destination, balance or contact checks fail.
Earlier diagnostic runs exposed false sign-side recognition and stale-map route
gaps; the resulting fixes have dedicated regression tests. The default
`memory/robot_map.npz` contains observations and odometry history learned during
simulation runs, not room coordinates supplied by the world builder. Use an
unused memory filename to start learning from an empty map.
