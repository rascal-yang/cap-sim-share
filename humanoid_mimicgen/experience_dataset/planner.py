"""Opt-in adapter for existing plan/reflect providers; never alters actions."""
import copy
import json
from .retrieval import query_experience


class ExperiencePlanner:
    def __init__(self, delegate, dataset, snapshot, context):
        self.delegate = delegate; self.dataset = dataset; self.snapshot = snapshot
        self.context = copy.deepcopy(context)
        self.snapshot_hash = dataset.load_snapshot(snapshot)['sha256']

    def _request(self, request):
        if self.dataset.load_snapshot(self.snapshot)['sha256'] != self.snapshot_hash:
            raise ValueError('Planner snapshot changed')
        q = dict(self.context, snapshot=self.snapshot)
        q.setdefault('text', json.dumps(request.get('observation', {}), ensure_ascii=False))
        enriched = copy.deepcopy(request)
        enriched['experience_context'] = query_experience(self.dataset, q)
        return enriched

    def plan(self, request):
        return self.delegate.plan(self._request(request))

    def reflect(self, request):
        return self.delegate.reflect(self._request(request))
