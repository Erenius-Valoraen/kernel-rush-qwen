import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('client', Path(__file__).resolve().parents[1] / 'agent/client.py')
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


class ClientTests(unittest.TestCase):
    def test_submission_to_terminal_run_uses_api_envelopes(self):
        api = client.Dryft('https://example.test', 'test-token')
        with patch.object(api, '_send', side_effect=[
            {'submission': {'id': 'submission-1'}},
            {'run': {'id': 'run-1', 'state': 'queued'}, 'replayed': False},
            {'run': {'id': 'run-1', 'state': 'succeeded', 'result': {'score': 100}}},
        ]):
            submission = api.submit(b'archive')
            run = api.start_run(submission)
            result = api.wait(run['id'], timeout=1)
        self.assertEqual(submission, 'submission-1')
        self.assertEqual(result['state'], 'succeeded')
        self.assertEqual(result['result']['score'], 100)

    def test_first_log_request_includes_sequence_zero(self):
        api = client.Dryft('https://example.test', 'test-token')
        with patch.object(api, '_send', return_value={'items': []}) as send:
            api.logs('run-1')
        self.assertIn('after=-1', send.call_args.args[1])


if __name__ == '__main__':
    unittest.main()
