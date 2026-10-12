# Why two workflow-modeling artifacts instead of one

The task asks for a Simulink workflow simulation of the telemedicine
screening pipeline (acquisition rate, bandwidth, processing throughput,
review capacity) to optimize resource allocation for a 100,000+
patient/year district program. Two complementary tools are provided:

## 1. `capacityPlanningAnalysis.m` — analytical queueing model
- Runs in plain MATLAB, no Simulink license required.
- Uses M/M/1 and M/M/c (Erlang-C) queueing formulas to directly compute
  the *minimum number of servers* (upload channels, AI compute workers,
  reviewing ophthalmologists) needed to keep utilization under a target
  (default 75%) at a given patient volume.
- Produces an immediate sensitivity table (reviewer count vs. review time
  vs. referral rate) — the single most useful number for a district health
  officer planning staffing.
- This is the right tool for **answering "how many of X do we need"**
  quickly and reproducibly.

## 2. `buildScreeningWorkflowModel.m` — Simulink block diagram
- Must be run inside MATLAB (creates `ScreeningWorkflow.slx` using the
  Simulink API — `.slx` is a binary format that only MATLAB can write).
- Gives a **visual, editable, presentable** block diagram of the same
  pipeline: arrival → upload/bandwidth → AI processing → referral split →
  human review → outcomes, using Transport Delay + Saturation blocks as a
  fast continuous-time approximation of queueing behavior.
- Intended as the starting point for a district planning meeting or a
  more detailed model — swap the simplified blocks for actual **SimEvents**
  blocks (Entity Generator, Queue, Server, Entity Sink) if SimEvents is
  licensed, to get a true stochastic discrete-event simulation with
  distributions (e.g. Poisson arrivals, lognormal review times) instead of
  the deterministic rate approximation used here.
- This is the right tool for **exploring "what if" scenarios visually**
  and communicating the pipeline structure to non-programmers.

## Recommended workflow
1. Run `capacityPlanningAnalysis.m` first to get sizing numbers and
   understand which stage (upload, AI, or review) is the bottleneck for
   your target volume.
2. Feed those numbers into `buildScreeningWorkflowModel.m` (pass the
   struct in, or edit the defaults) to get a visual model matching the
   analytical sizing, for stakeholder review and further refinement.
3. If your organization has a SimEvents license and needs to model
   burstiness (patients don't arrive at a perfectly steady rate — clinic
   hours, seasonal camps, etc.), extend the `.slx` model with SimEvents
   blocks using the analytical model's numbers as a sanity check on the
   simulation's steady-state behavior.
