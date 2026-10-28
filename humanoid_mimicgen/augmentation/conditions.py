"""Explicit, repeatable planar conditions; no implicit robot/object coupling."""
import numpy as np
from .contracts import require, validate_shape


def vector(value, length):
    require(isinstance(value, list) and len(value) == length, 'Vector dimensions')
    require(all(type(x) in (int, float) and np.isfinite(x) for x in value), 'Finite numeric vector')
    return value


def sample_conditions(job):
    validate_shape(job)
    c = job['conditions']
    require(isinstance(c, dict), 'Conditions object')
    if c.get('mode') == 'explicit':
        require(set(c) == {'mode', 'cases'}, 'Explicit condition fields')
        rows = c['cases']
        require(isinstance(rows, list) and 0 < len(rows) <= job['budgets']['max_cases'], 'Case budget exceeded')
        seen = set()
        for row in rows:
            require(isinstance(row, dict) and set(row) == {'case_id', 'robot_delta_xyyaw', 'object_delta_xy'}, 'Case fields')
            import re
            name = row['case_id']
            require(isinstance(name, str) and re.fullmatch('[A-Za-z0-9_-]{1,128}', name) and name not in seen, 'Unique case ID')
            seen.add(name)
            vector(row['robot_delta_xyyaw'], 3); vector(row['object_delta_xy'], 2)
        import copy
        return copy.deepcopy(rows)
    require(c.get('mode') == 'uniform' and set(c) == {'mode', 'seed', 'count', 'robot_delta_xyyaw', 'object_delta_xy'}, 'Sampling fields')
    require(type(c['seed']) is int and c['seed'] >= 0, 'Nonnegative seed required')
    require(type(c['count']) is int and 0 < c['count'] <= job['budgets']['max_cases'], 'Case budget exceeded')
    bounds = []
    for key, length in (('robot_delta_xyyaw', 3), ('object_delta_xy', 2)):
        require(isinstance(c[key], list) and len(c[key]) == length, 'Range dimensions')
        for pair in c[key]:
            vector(pair, 2)
            require(pair[0] <= pair[1], 'Reversed range')
            bounds.append(pair)
    rng = np.random.Generator(np.random.PCG64(c['seed']))
    values = rng.uniform(np.array(bounds)[:, 0], np.array(bounds)[:, 1], (c['count'], 5))
    return [{'case_id': f'case_{i:06d}', 'robot_delta_xyyaw': v[:3].tolist(),
             'object_delta_xy': v[3:].tolist()} for i, v in enumerate(values)]
