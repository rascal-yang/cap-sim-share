"""Controlled runner with complete measured action-prefix feedback in LIVE."""
import controlled_run
from repro.feedback_session import FeedbackSession

if __name__ == '__main__':
    controlled_run.RobustSession = FeedbackSession
    controlled_run.main()
