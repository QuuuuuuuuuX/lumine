# Lumine agent evaluation

- started: 2026-10-09 15:37:03
- brain: `scripted` (configured: `none` / `deepseek-flash`)
- controller: `rules` (pixels observation)
- episodes: 14
- **overall success: 14.3%**
- wall clock: 348s

## By category

| category | episodes | success | mean s | model calls | model seconds | faults |
|---|---|---|---|---|---|---|
| boss | 2 | 0% | 31.8 | 35.0 | 0.0 | 0 |
| combat | 2 | 0% | 26.9 | 29.0 | 0.0 | 0 |
| gui | 2 | 0% | 19.4 | 43.0 | 0.0 | 0 |
| icl | 2 | 0% | 37.0 | 39.0 | 0.0 | 0 |
| mission | 2 | 0% | 29.5 | 77.5 | 0.0 | 0 |
| npc | 2 | 100% | 0.5 | 2.0 | 0.0 | 0 |
| puzzle | 2 | 0% | 28.5 | 74.0 | 0.0 | 0 |

## By region (generalisation)

| region | episodes | success | note |
|---|---|---|---|
| liyue | 7 | 14% | **held out / out of distribution** |
| mondstadt | 7 | 14% | in distribution |

## Episodes

| task | category | region | seed | result | reason | s | calls | faults |
|---|---|---|---|---|---|---|---|---|
| `combat_defeat_and_chest` | combat | mondstadt | 0 | FAIL | timed out [mean 0.22 Hz over 120s] | 32.3 | 27 | 0 |
| `boss_hypostasis_electro` | boss | mondstadt | 0 | FAIL | timed out [mean 0.23 Hz over 240s] | 57.3 | 56 | 0 |
| `puzzle_anemoculus_wind` | puzzle | mondstadt | 0 | FAIL | timed out [mean 0.49 Hz over 150s] | 30.5 | 74 | 0 |
| `npc_talk_grace` | npc | mondstadt | 0 | PASS | task complete [2 model calls in 2.1s] | 0.5 | 2 | 0 |
| `gui_cook_sweet_madame` | gui | mondstadt | 0 | FAIL | timed out [mean 0.48 Hz over 90s] | 22.2 | 43 | 0 |
| `icl_climb_pillar` | icl | mondstadt | 0 | FAIL | timed out [mean 0.22 Hz over 180s] | 42.1 | 39 | 0 |
| `mission_mondstadt_act1` | mission | mondstadt | 0 | FAIL | player defeated [9 model calls in 19.8s] | 4.6 | 9 | 0 |
| `combat_defeat_and_chest` | combat | liyue | 0 | FAIL | timed out [mean 0.26 Hz over 120s] | 21.4 | 31 | 0 |
| `boss_hypostasis_electro` | boss | liyue | 0 | FAIL | player defeated [mean 0.40 Hz over 35s] | 6.3 | 14 | 0 |
| `puzzle_anemoculus_wind` | puzzle | liyue | 0 | FAIL | timed out [mean 0.49 Hz over 150s] | 26.5 | 74 | 0 |
| `npc_talk_grace` | npc | liyue | 0 | PASS | task complete [2 model calls in 2.1s] | 0.4 | 2 | 0 |
| `gui_cook_sweet_madame` | gui | liyue | 0 | FAIL | timed out [mean 0.48 Hz over 90s] | 16.7 | 43 | 0 |
| `icl_climb_pillar` | icl | liyue | 0 | FAIL | timed out [mean 0.22 Hz over 180s] | 31.9 | 39 | 0 |
| `mission_mondstadt_act1` | mission | liyue | 0 | FAIL | timed out [mean 0.49 Hz over 300s] | 54.4 | 146 | 0 |

## Reading this report

The `model seconds` column is wall-clock time the agent spent blocked on (or
concurrently waiting for) its vision-language model. It, not the success rate,
is the number that decides whether the architecture is deployable at 30 Hz:
the controller keeps emitting keys throughout, which is exactly why the brain
is a planner over skills rather than a source of keystrokes.
