# NEATGD

**NEATGD** teaches itself to play Geometry Dash levels using
[NEAT](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies)
(NeuroEvolution of Augmenting Topologies); neural networks that evolve their
own structure and weights from scratch.

Every genome is judged by *actually playing* the level: each one takes over
your player for a fast-forwarded attempt, its network reading the terrain
ahead and deciding when to jump, the exact same engine a human plays
against, so what a genome scores in training is what it replays. Children
inherit their parent's proven inputs up to just before where the parent
died, then explore from that frontier, so progress is never lost to a bad
mutation. The furthest, cleanest runs breed the next generation until
something learns to beat the level.

## How to use

1. Open a level and hit **pause**.
2. Click the auto button in the top left corner of the pause menu.
3. Tune the training settings (each has an info button explaining it) and
   press **Train**.
4. Watch the population evolve. When training finishes, the best genome plays
   the level back at normal speed.

Click the button again at any time to cancel.

## Tips

- NEAT improves through **generations**, so a moderate population (~150-500)
  with a high generation cap usually learns faster than a huge one.
- Turn on **Hide graphics** to spend your hardware on physics instead of
  rendering. It lets you push the **Speed** value higher.

---

Made by **Itzar**.
