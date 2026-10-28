# Simple continuous replay navigation

User explicitly cancels the corridor abort and unnecessary runtime gates; native task completion is the only acceptance criterion. This supersedes prior strict-corridor/stability/contact-veto designs for the new simple execution path.

Implementation: retain original far scene state and real SONIC history, use existing velocity/yaw route control without corridor/stale-observation/stage-timeout aborts, begin a4s quintic measured-posture→replay transition on goal arrival, run source reference and fingers by time, then hold1s. No calibration prerequisite, measured-foot IK/handoff screening, posture/contact readiness clocks or contact/fall/obstacle-force early termination in this path. One75s simulator budget bounds computation; nonfinite state is a simulator failure. Physical dynamics/PD/hand commands/native predicate unchanged. Continue recording every physical step and SONIC token; contacts/tracking are measurements and not success vetoes. No teleport or decoder override.

Use existing experiment preparation with a navigation argument; explicitly freeze actual original initialization, history, interfaces, removed gates, code/runtime and native acceptance. Standalone entry augmentation.replay_navigation starts physics with the same prepared adapter, no images/API. Old frozen evidence stays immutable.
