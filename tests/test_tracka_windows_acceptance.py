"""Acceptance-command tests: no pip install, research images or hardware used."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/windows/accept_perilipin2_tracka.py'
SPEC = importlib.util.spec_from_file_location('accept_tracka', SOURCE)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.bundle = self.base / 'input'
        self.bundle.mkdir()
        self.environment = self.base / 'analysis_env'
        self.output = self.base / 'result'
        self.packages = ('numpy', 'pandas', 'scipy', 'scikit-image', 'Pillow', 'tifffile')
        self.pins = [name + '==1.2.3' for name in self.packages]
        (self.bundle / 'reference_requirements.txt').write_text('\n'.join(self.pins) + '\n')
        self.manifest = {'purpose': 'frozen_two_field_reference_replay',
                         'fingerprint_sha256': 'a' * 64,
                         'files': {'reference_requirements.txt': {'bytes': 123, 'sha256': 'fake'}},
                         'reference_versions': dict(zip(self.packages, ['1.2.3'] * 6))}
        self.core = types.SimpleNamespace(
            ENVIRONMENT=self.packages,
            verify_bundle=mock.Mock(return_value=(self.bundle, self.manifest)),
            contained=lambda root, name: root / name,
            file_record=lambda path: {'bytes': 123, 'sha256': 'fake'},
            new_output=lambda out, protected: (Path(out), Path(str(out) + '.BUILDING')),
            load_json=lambda path: json.loads(path.read_text()),
        )
        self.args = types.SimpleNamespace(bundle=self.bundle, environment=self.environment,
            out=self.output, create_environment=False, check_only=False)
        self.patch = mock.patch.object(module, 'adapter', return_value=self.core)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def ready_venv(self):
        self.environment.mkdir()
        python = self.environment / ('Scripts/python.exe' if module.os.name == 'nt' else 'bin/python')
        python.parent.mkdir()
        python.write_bytes(b'test-only')
        (self.environment / 'pyvenv.cfg').write_text('test-only')
        return python

    def successful_command(self, command):
        if 'run' in [str(x) for x in command]:
            self.output.mkdir()
            (self.output / 'TRACKA_EXECUTION.json').write_text(
                json.dumps({'status': 'TRACKA_REFERENCE_REPLAY_PASS'}))
            (self.output / 'REFERENCE_COMPARISON.json').write_text(json.dumps({'passed': True}))

    def test_check_only_never_creates_environment_or_runs_commands(self):
        self.args.check_only = True
        with mock.patch.object(module, 'checked_call') as call:
            self.assertEqual(module.run(self.args), 0)
        call.assert_not_called()
        self.assertFalse(self.environment.exists())
        self.assertFalse(self.output.exists())
        self.core.verify_bundle.assert_called_once()

    def test_bad_bundle_stops_before_environment_setup(self):
        self.args.create_environment = True
        self.core.verify_bundle.side_effect = ValueError('bundle mismatch')
        with mock.patch.object(module, 'checked_call') as call:
            with self.assertRaisesRegex(ValueError, 'bundle mismatch'):
                module.run(self.args)
        call.assert_not_called()
        self.assertFalse(self.environment.exists())

    def test_install_requires_explicit_permission(self):
        with mock.patch.object(module, 'checked_call') as call:
            with self.assertRaisesRegex(ValueError, '--create-environment'):
                module.run(self.args)
        call.assert_not_called()
        self.assertFalse(self.environment.exists())

    def test_requirements_options_or_urls_rejected(self):
        path = self.bundle / 'reference_requirements.txt'
        for extra in ('--index-url https://invalid.test', 'numpy @ https://invalid.test', 'numpy==2.0'):
            path.write_text('\n'.join(self.pins[:-1] + [extra]) + '\n')
            with self.assertRaisesRegex(ValueError, 'six recorded package pins'):
                module.package_pins(self.core, self.bundle, self.manifest)

    def test_existing_env_is_reused_without_installing_even_with_flag(self):
        self.ready_venv()
        self.args.create_environment = True
        with mock.patch.object(module, 'check_environment') as check:
            with mock.patch.object(module, 'checked_call', side_effect=self.successful_command) as call:
                self.assertEqual(module.run(self.args), 0)
        check.assert_called_once()
        self.assertEqual(len(call.call_args_list), 2)
        self.assertFalse(any('install' in x.args[0] for x in call.call_args_list))
        self.assertIn('run', call.call_args_list[0].args[0])
        self.assertIn('verify', call.call_args_list[1].args[0])

    def test_new_install_targets_only_new_environment(self):
        self.args.create_environment = True
        with mock.patch.object(module, 'check_environment'):
            with mock.patch.object(module, 'checked_call', side_effect=self.successful_command) as call:
                self.assertEqual(module.run(self.args), 0)
        commands = [entry.args[0] for entry in call.call_args_list]
        self.assertEqual(len(commands), 4)
        self.assertIn('venv', commands[0])
        self.assertIn('install', commands[1])
        self.assertTrue(Path(commands[1][0]).is_relative_to(self.environment))
        self.assertIn('--only-binary=:all:', commands[1])
        self.assertEqual(commands[1][-6:], self.pins)

    def test_environment_mismatch_not_repaired_and_no_analysis(self):
        self.ready_venv()
        with mock.patch.object(module, 'check_environment', side_effect=ValueError('versions differ')):
            with mock.patch.object(module, 'checked_call') as call:
                with self.assertRaisesRegex(ValueError, 'versions differ'):
                    module.run(self.args)
        call.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_worker_failure_not_followed_by_verification(self):
        self.ready_venv()
        with mock.patch.object(module, 'check_environment'):
            with mock.patch.object(module, 'checked_call', side_effect=RuntimeError('measurement mismatch')) as call:
                with self.assertRaisesRegex(RuntimeError, 'measurement mismatch'):
                    module.run(self.args)
        self.assertEqual(call.call_count, 1)

    def test_missing_required_manifest_entry_rejected(self):
        self.manifest['files'].clear()
        with self.assertRaisesRegex(ValueError, 'not covered'):
            module.package_pins(self.core, self.bundle, self.manifest)

    def test_overlapping_environment_is_rejected(self):
        self.args.environment = self.bundle / 'nested'
        self.args.create_environment = True
        with self.assertRaisesRegex(ValueError, 'separate'):
            module.run(self.args)
        self.assertFalse(self.args.environment.exists())


if __name__ == '__main__':
    unittest.main()
