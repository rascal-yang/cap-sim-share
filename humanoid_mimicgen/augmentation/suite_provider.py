"""Quality-suite-only credential failover; each request remains separately audited."""
import json
import fcntl
from pathlib import Path
import os
from .provider import ResponsesProvider


class FailoverProvider:
    def __init__(self, primary, secondary, *additional):
        self.providers = (primary, secondary, *additional)
        for other in self.providers[1:]:
            for field in ('model', 'reasoning_effort'):
                if primary.config.get(field) != other.config.get(field):
                    raise ValueError('Failover must preserve model and reasoning effort')
        self.slots = tuple(('primary','secondary','tertiary')[i] if i<3 else f'provider_{i+1}' for i in range(len(self.providers)))
        self.config = primary.config
        self.active = 0

    def attempt_providers(self):
        for index in range(self.active, len(self.providers)):
            self.active = index
            for _ in range(6):  # Initial request plus five retries, then next credential.
                yield self.slots[index], self.providers[index]


class AlternatingProvider(FailoverProvider):
    """Allocate alternating credentials atomically across all task processes."""
    def __init__(self, primary, secondary, routing_path, *additional):
        super().__init__(primary, secondary, *additional)
        self.routing_path = Path(routing_path)
        self.routing_sequence = None

    def attempt_providers(self):
        for _ in range(6):  # Initial request plus five retries, alternating every time.
            self.routing_path.parent.mkdir(parents=True, exist_ok=True)
            with self.routing_path.open('a+') as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                handle.seek(0)
                text = handle.read()
                sequence = json.loads(text)['next_sequence'] if text else 0
                if type(sequence) is not int or sequence < 0:
                    raise ValueError('Invalid persisted API routing sequence')
                handle.seek(0)
                handle.truncate()
                json.dump({'next_sequence': sequence + 1}, handle)
                handle.flush()
                os.fsync(handle.fileno())
            self.routing_sequence = sequence
            index = sequence % len(self.providers)
            yield self.slots[index], self.providers[index]


def configured_provider(config, routing_path=None):
    primary = ResponsesProvider(config)
    secondary = os.environ.get('QUALITY_SUITE_FALLBACK_CONFIG')
    if not secondary:
        return primary
    second = ResponsesProvider(json.loads(secondary))
    extra = [ResponsesProvider(c) for c in json.loads(os.environ.get('QUALITY_SUITE_ADDITIONAL_CONFIGS','[]'))]
    mode = os.environ.get('QUALITY_SUITE_API_ROUTING', 'alternate')
    if mode == 'failover':
        return FailoverProvider(primary, second, *extra)
    if mode != 'alternate' or routing_path is None:
        raise ValueError('Alternating API routing requires a shared study routing path')
    return AlternatingProvider(primary, second, routing_path, *extra)
