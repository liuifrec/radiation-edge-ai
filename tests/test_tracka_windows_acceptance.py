"""Acceptance-command tests: no pip install, research images or hardware used."""
from __future__ import annotations

import importlib.util
import json
import sys
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
        self.analysis_base = self.base / 'python311.exe'
        self.analysis_base.write_bytes(b'not executed: mock interpreter')
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
            out=self.output, create_environment=False, check_only=False, base_python=None)
        self.patch = mock.patch.object(module, 'adapter', return_value=self.core)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def interpreter(self, version=(3, 11, 9), bits=64, implementation='cpython'):
        return {'version': list(version), 'bits': bits, 'implementation': implementation,
                'executable': str(self.analysis_base), 'prefix': str(self.base / 'base-install'),
                'base_prefix': str(self.base / 'base-install')}

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
            with mock.patch.object(module, 'probe_analysis_python') as probe:
                self.assertEqual(module.run(self.args), 0)
        call.assert_not_called()
        probe.assert_not_called()
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
        self.args.base_python = self.analysis_base
        with mock.patch.object(module, 'check_environment') as check:
            with mock.patch.object(module, 'probe_analysis_python') as probe:
                with mock.patch.object(module, 'checked_call', side_effect=self.successful_command) as call:
                    self.assertEqual(module.run(self.args), 0)
        check.assert_called_once()
        probe.assert_not_called()  # Existing worker is checked; base selection is creation-only.
        self.assertEqual(len(call.call_args_list), 2)
        self.assertFalse(any('install' in x.args[0] for x in call.call_args_list))
        self.assertIn('run', call.call_args_list[0].args[0])
        self.assertIn('verify', call.call_args_list[1].args[0])

    def test_new_install_targets_only_new_environment_and_uses_explicit_base(self):
        self.args.create_environment = True
        self.args.base_python = self.analysis_base
        with mock.patch.object(module, 'probe_analysis_python',
                               return_value=(self.analysis_base, self.interpreter())) as probe:
            with mock.patch.object(module, 'check_environment'):
                with mock.patch.object(module, 'checked_call', side_effect=self.successful_command) as call:
                    self.assertEqual(module.run(self.args), 0)
        probe.assert_called_once_with(self.analysis_base)
        commands = [entry.args[0] for entry in call.call_args_list]
        self.assertEqual(len(commands), 4)
        self.assertEqual(commands[0][0], self.analysis_base)
        self.assertIn('venv', commands[0])
        self.assertIn('install', commands[1])
        self.assertTrue(Path(commands[1][0]).is_relative_to(self.environment))
        self.assertIn('--only-binary=:all:', commands[1])
        self.assertEqual(commands[1][-6:], self.pins)
        self.assertNotEqual(commands[1][0], self.analysis_base)
        self.assertEqual(self.analysis_base.read_bytes(), b'not executed: mock interpreter')

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

    def test_python39_fails_before_any_environment_creation_or_install(self):
        self.args.create_environment = True
        self.args.base_python = self.analysis_base
        result = types.SimpleNamespace(returncode=0,
            stdout=json.dumps(self.interpreter(version=(3, 9, 13))), stderr='')
        with mock.patch.object(module.subprocess, 'run', return_value=result):
            with mock.patch.object(module, 'checked_call') as call:
                with self.assertRaisesRegex(ValueError, 'Incompatible analysis Python.*3.9.13'):
                    module.run(self.args)
        call.assert_not_called()
        self.assertFalse(self.environment.exists())
        self.assertFalse(self.output.exists())

    def test_compatible_base_versions_are_read_only_bounded_probes(self):
        for minor in (11, 12, 13, 14):
            result = types.SimpleNamespace(returncode=0,
                stdout=json.dumps(self.interpreter(version=(3, minor, 1))), stderr='')
            with self.subTest(minor=minor), mock.patch.object(
                    module.subprocess, 'run', return_value=result) as run:
                path, info = module.probe_analysis_python(self.analysis_base)
                self.assertEqual(path, self.analysis_base)
                self.assertEqual(info['version'][1], minor)
                self.assertFalse(run.call_args.kwargs['shell'])
                self.assertEqual(run.call_args.kwargs['timeout'], 30)
                self.assertIn('-I', run.call_args.args[0])
                self.assertNotIn('install', run.call_args.args[0])
        self.assertFalse(self.environment.exists())

    def test_unsupported_versions_architecture_implementation_are_rejected(self):
        records = [self.interpreter(version=(3, 10, 11)),
                   self.interpreter(version=(3, 15, 0)),
                   self.interpreter(bits=32), self.interpreter(implementation='pypy')]
        for info in records:
            result = types.SimpleNamespace(returncode=0, stdout=json.dumps(info), stderr='')
            with self.subTest(info=info), mock.patch.object(module.subprocess, 'run', return_value=result):
                with self.assertRaisesRegex(ValueError, '64-bit CPython 3.11-3.14'):
                    module.probe_analysis_python(self.analysis_base)

    def test_missing_base_does_not_start_any_process(self):
        with mock.patch.object(module.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'py -0p'):
                module.probe_analysis_python(self.base / 'missing.exe')
        run.assert_not_called()

    def test_selected_base_prefix_is_protected(self):
        self.args.create_environment = True
        self.args.base_python = self.analysis_base
        self.args.environment = self.base / 'base-install'
        with mock.patch.object(module, 'probe_analysis_python',
                               return_value=(self.analysis_base, self.interpreter())):
            with mock.patch.object(module, 'checked_call') as call:
                with self.assertRaisesRegex(ValueError, 'separate'):
                    module.run(self.args)
        call.assert_not_called()
        self.assertFalse(self.args.environment.exists())

    def test_existing_partial_environment_never_repaired_by_selecting_new_base(self):
        self.environment.mkdir()
        evidence = self.environment / 'failed-setup.txt'
        evidence.write_bytes(b'preserve')
        self.args.create_environment = True
        self.args.base_python = self.analysis_base
        with mock.patch.object(module, 'checked_call') as call:
            with self.assertRaisesRegex(ValueError, 'left untouched'):
                module.run(self.args)
        call.assert_not_called()
        self.assertEqual(evidence.read_bytes(), b'preserve')
        self.assertEqual(list(self.environment.iterdir()), [evidence])

    def test_existing_worker_must_be_venv_before_importing_packages(self):
        python = self.ready_venv()
        info = self.interpreter()
        with mock.patch.object(module, 'probe_analysis_python', return_value=(python, info)):
            with mock.patch.object(module.subprocess, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'intended isolated venv'):
                    module.check_environment(python, self.environment, self.pins)
        run.assert_not_called()

    def test_default_base_is_probed_not_silently_assumed_compatible(self):
        self.args.create_environment = True
        with mock.patch.object(module, 'probe_analysis_python',
                               side_effect=ValueError('incompatible launcher')) as probe:
            with mock.patch.object(module, 'checked_call') as call:
                with self.assertRaisesRegex(ValueError, 'incompatible launcher'):
                    module.run(self.args)
        probe.assert_called_once_with(sys.executable)
        call.assert_not_called()
        self.assertFalse(self.environment.exists())


if __name__ == '__main__':
    unittest.main()
