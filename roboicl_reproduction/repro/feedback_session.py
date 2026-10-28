"""Carry confirmed command feedback through the existing final-chunk receipt."""
from copy import deepcopy
from dataclasses import replace

from .robust_session import RobustSession
from .session import StepResult


class FeedbackSession(RobustSession):
    def submit(self, call, execute):
        # Local to this submission, including invalid calls and zero-step retries.
        prefix = []
        action_index = 0

        def capture(command):
            nonlocal action_index
            result = execute(command)
            if not isinstance(result, StepResult):
                return result  # The original Session rejects invalid evidence.
            feedback = deepcopy(result.feedback) if result.feedback is not None else {}
            if not isinstance(feedback, dict):
                raise ValueError('Command feedback must be an object or None')
            # Never recursively embed an accumulator in its own entries.
            feedback.pop('executed_prefix_feedback', None)
            if result.executed is True:
                prefix.append({'action_index': action_index, 'observation_step': self.step+1,
                    'elapsed_s': result.elapsed_s, 'terminal': result.terminal,
                    'interrupt': result.interrupt, 'feedback': deepcopy(feedback)})
            action_index += 1
            feedback['executed_prefix_feedback'] = deepcopy(prefix)
            return replace(result, feedback=feedback)

        # Validation, rejection budgets, suffix discard, and unknown-execution
        # poisoning remain owned by Session. Exceptions do not fabricate rows.
        return super().submit(call, capture)
