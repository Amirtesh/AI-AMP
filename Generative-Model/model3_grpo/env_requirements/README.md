# Environment Setup Notes

Two separate conda environments are required due to a sklearn pickle
version incompatibility:

- **RLGEN**: main training environment (GRPO, Model2, HemoPI2 hybrid
  scoring). Requires sklearn ~1.3.x. Used for grpo_train.py and all
  reward-function scoring except ToxinPred3.

- **toxin_hemo**: ToxinPred3 scoring environment. Requires sklearn ~1.2.2,
  since toxinpred3's bundled model pickle (toxinpred3.0_model.pkl) was
  serialized under an older sklearn Tree format and raises
  `ValueError: node array from the pickle has an incompatible dtype`
  under sklearn >=1.3. HemoPI2's ESM-based Hybrid2 scoring path has no
  such conflict and runs fine under RLGEN's sklearn 1.3.1 directly.

ToxinPred3 is served via a persistent local scoring server
(toxinpred3_server.py, run in the toxin_hemo env) so that grpo_train.py
(running in RLGEN) can call it over HTTP without paying per-call model
reload cost.

To reproduce: create both conda envs, `pip install -r requirements_RLGEN.txt`
and `pip install -r requirements_toxin_hemo.txt` respectively, matching
Python versions noted in notes_*.txt.
