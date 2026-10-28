"""One audited request per call, rotating credentials even when a request fails."""
import fcntl
import json
import os
from pathlib import Path

from .contracts import require
from .provider import ResponsesProvider
from ..workflow.contracts import digest, canonical


class RotatingResponsesProvider(ResponsesProvider):
    def __init__(self, config, slots, routing_path):
        super().__init__(config)
        require(1 <= len(slots) <= 32 and len({n for n, _ in slots}) == len(slots), 'Unique bounded API pool')
        for label, provider in slots:
            require(isinstance(label, str) and bool(label), 'API pool label')
            for key in ('model', 'reasoning_effort', 'protocol'):
                require(provider.config.get(key) == self.config.get(key), 'API rotation must preserve model/effort/protocol')
        self.slots = list(slots)
        self.real_transport = all(isinstance(p, ResponsesProvider) for _, p in slots)
        self.routing_path = Path(routing_path)
        self.signature = digest([(n, p.config['model'], p.config['base_url']) for n, p in slots])

    def respond(self, items, tools, instructions):
        self.routing_path.parent.mkdir(parents=True, exist_ok=True)
        with self.routing_path.with_suffix('.lock').open('a+') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            existed = self.routing_path.exists()
            existing = self.routing_path.read_text() if existed else ''
            require(bool(existing) or not existed, 'Empty existing API routing record', 'integrity_error')
            state = json.loads(existing) if existing else dict(next_sequence=0, pool_signature=self.signature)
            require(state.get('pool_signature') == self.signature, 'API pool changed within session', 'integrity_error')
            sequence = state['next_sequence']
            require(type(sequence) is int and sequence >= 0, 'API sequence')
            label, provider = self.slots[sequence % len(self.slots)]
            self.last_provider_label = label
            self.last_provider_sequence = sequence
            self.last_provider_endpoint = provider.config['base_url']
            temporary = self.routing_path.with_suffix('.tmp')
            with temporary.open('wb') as output:
                output.write(canonical(dict(next_sequence=sequence+1, pool_signature=self.signature,
                                           last_label=label, last_sequence=sequence)))
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(self.routing_path)
            directory = os.open(self.routing_path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        provider.config['timeout_s'] = min(provider.config['timeout_s'], self.config['timeout_s'])
        try:
            return provider.respond(items, tools, instructions)
        finally:
            self.last_transport_runtime = getattr(provider, 'last_transport_runtime', None)

    @classmethod
    def from_profiles(cls, config, profiles, routing_path):
        return cls(config, [(p['label'], ResponsesProvider(p['config'])) for p in profiles], routing_path)
