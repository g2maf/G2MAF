# Release verification — 2026-09-30

A separate fresh Python 3.8 environment installed requirements.txt and passed pip check and the critic gradient smoke test. The MPE critic trainer completed two updates on a real Spread Medium subset. Evaluation loaded the critic and a short-trained CoFlow policy, completing one MPE episode without guidance and one with a 0.01 guidance step. Guided evaluation was repeated successfully in the fresh environment.

These are installation and short functional checks, not full-budget retraining or reproduction of paper scores. They do not certify every map, data split, historical checkpoint, optional rendering path, or inherited prototype. Simulator binaries/maps and datasets remain external requirements. Training seeds and evaluation seeds are distinct.
