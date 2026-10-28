"""Stdlib-only HTTP worker; credentials are inherited, never serialized."""
import json
import platform
import ssl
import sys

from .provider import ResponsesProvider
from ..workflow.contracts import WorkflowError, canonical


def main():
    runtime = dict(python=platform.python_version(), executable=sys.executable, ssl=ssl.OPENSSL_VERSION)
    try:
        raw = sys.stdin.buffer.read(32*1024*1024+1)
        if len(raw) > 32*1024*1024:
            raise WorkflowError('invalid_input', 'Transport input limit')
        value = json.loads(raw)
        if 'transport_python' in value['config']:
            raise WorkflowError('invalid_input', 'Recursive transport forbidden')
        response = ResponsesProvider(value['config']).respond(value['input'], value['tools'], value['instructions'])
        result = dict(ok=True, response=response, transport_runtime=runtime)
    except WorkflowError as exc:
        result = dict(ok=False, reason_code=exc.code, detail=exc.detail, transport_runtime=runtime)
    except Exception as exc:
        result = dict(ok=False, reason_code='api_unavailable', detail=type(exc).__name__, transport_runtime=runtime)
    sys.stdout.buffer.write(canonical(result))


if __name__ == '__main__':
    main()
