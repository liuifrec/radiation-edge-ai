"""Verify the copied reference bundle and run the existing Track A analysis.

--create-environment explicitly permits installing the six recorded analysis
packages into a NEW venv only. --base-python selects its interpreter separately
from the launcher; it is checked before creating any directory or installing.
Existing environments are checked, never repaired or upgraded. No Docker,
KL720, model training or changes to assay rules occur.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ADAPTER = REPO / 'src/radiation_edge_ai/perilipin2_tracka.py'
DEFAULT_ROOT = Path(r'D:\radiation-edge-ai-data')
MIN_ANALYSIS_PYTHON = (3, 11)
MAX_ANALYSIS_PYTHON = (3, 14)


def adapter():
    spec = importlib.util.spec_from_file_location('_tracka_acceptance_core', ADAPTER)
    if spec is None or spec.loader is None:
        raise ValueError('Track A analysis adapter is missing from this worktree')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def separate(a, b):
    """Neither path may contain the other (also protects existing runtimes)."""
    if a == b or a in b.parents or b in a.parents:
        raise ValueError('Keep input, output, repository and environment separate: ' + str(a))


def package_pins(core, root, manifest):
    name = 'reference_requirements.txt'
    record = manifest['files'].get(name)
    path = core.contained(root, name)
    if record is None or core.file_record(path) != record:
        raise ValueError('Reference requirements are not covered by the bundle manifest')
    expected = []
    for package in core.ENVIRONMENT:
        version = manifest['reference_versions'].get(package)
        if not isinstance(version, str) or not re.fullmatch(r'[0-9][A-Za-z0-9.!+_-]*', version):
            raise ValueError('Invalid recorded package version: ' + package)
        expected.append(package + '==' + version)
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    if len(lines) != len(expected) or sorted(lines) != sorted(expected):
        raise ValueError('Requirements must contain exactly the six recorded package pins')
    return expected


def checked_call(command):
    """Inherit the terminal; UTF-8 is explicit for every Python invocation."""
    result = subprocess.run([str(x) for x in command], check=False, shell=False)
    if result.returncode:
        raise RuntimeError('Command failed (exit {}). Preserve diagnostics; no retry was made.'.format(
            result.returncode))


def probe_analysis_python(python):
    """Read-only interpreter preflight; never discover/download/install Python.

    NumPy 2.4 supports CPython 3.11-3.14. This Windows analysis route requires
    a 64-bit build. Passing this check does not guarantee all pinned wheels
    are present on the user's package index: pip still checks that normally.
    """
    path = Path(python).expanduser().resolve()
    if not path.is_file():
        raise ValueError('Analysis base Python does not exist: {}. Select an installed '
                         '64-bit CPython 3.11-3.14 executable with --base-python; '
                         'on Windows, py -0p lists installed interpreters.'.format(path))
    probe = (
        'import json,struct,sys; '
        'print(json.dumps({"executable":sys.executable,"version":list(sys.version_info[:3]),'
        '"implementation":sys.implementation.name,"bits":struct.calcsize("P")*8,'
        '"prefix":sys.prefix,"base_prefix":sys.base_prefix}))'
    )
    result = subprocess.run(
        [str(path), '-I', '-X', 'utf8', '-c', probe],
        capture_output=True, encoding='utf-8', errors='backslashreplace',
        check=False, shell=False, timeout=30)
    if result.returncode:
        raise ValueError('Could not inspect analysis Python; no environment setup attempted.\n'
                         + result.stderr)
    actual = json.loads(result.stdout)
    version = actual.get('version') if isinstance(actual, dict) else None
    if (not isinstance(version, list) or len(version) != 3
            or any(type(part) is not int or part < 0 for part in version)):
        raise ValueError('Invalid version metadata from analysis Python')
    for key in ('prefix', 'base_prefix', 'executable'):
        if not isinstance(actual.get(key), str) or not actual[key]:
            raise ValueError('Missing interpreter metadata: ' + key)
    if (actual.get('implementation') != 'cpython' or actual.get('bits') != 64
            or not MIN_ANALYSIS_PYTHON <= tuple(version[:2]) <= MAX_ANALYSIS_PYTHON):
        raise ValueError(
            'Incompatible analysis Python: {} {} ({}-bit) at {}. '
            'This reference stack (including NumPy 2.4.x) requires 64-bit CPython '
            '3.11-3.14. Use --base-python with a compatible installed executable '
            'and --environment with a NEW folder. No package pins were changed; '
            'existing environments were not modified.'.format(
                actual.get('implementation'), '.'.join(map(str, version)),
                actual.get('bits'), path))
    return path, actual


def check_environment(python, environment, expected):
    _, interpreter = probe_analysis_python(python)
    if (Path(interpreter['prefix']).resolve() != environment
            or Path(interpreter['prefix']).resolve() == Path(interpreter['base_prefix']).resolve()):
        raise ValueError('Selected analysis interpreter is not the intended isolated venv')
    probe = (
        'import importlib,importlib.metadata,json,sys; '
        'names=json.loads(sys.argv[1]); '
        '[importlib.import_module(n) for n in '
        '("numpy","pandas","scipy","skimage","PIL","tifffile")]; '
        'print(json.dumps({"prefix":sys.prefix,"base_prefix":sys.base_prefix,'
        '"versions":{n:importlib.metadata.version(n) for n in names}}))'
    )
    requested = dict(pin.split('==', 1) for pin in expected)
    result = subprocess.run(
        [str(python), '-X', 'utf8', '-c', probe, json.dumps(list(requested))],
        capture_output=True, encoding='utf-8', errors='backslashreplace',
        check=False, shell=False, timeout=120)
    if result.returncode:
        raise ValueError('Scientific environment is incomplete; it was not modified.\n' + result.stderr)
    actual = json.loads(result.stdout)
    if Path(actual['prefix']).resolve() != environment or actual['prefix'] == actual['base_prefix']:
        raise ValueError('Selected analysis interpreter is not the intended isolated venv')
    if actual['versions'] != requested:
        raise ValueError('Analysis package versions differ from the reference. No upgrade was attempted.\n'
                         + json.dumps({'expected': requested, 'actual': actual['versions']}, indent=2))
    checked_call([python, '-X', 'utf8', '-m', 'pip', 'check'])


def run(args):
    core = adapter()
    root, manifest = core.verify_bundle(args.bundle)
    if manifest.get('purpose') != 'frozen_two_field_reference_replay':
        raise ValueError('Expected the two-field Track A reference bundle')
    pins = package_pins(core, root, manifest)
    print('TRACKA_COPIED_BUNDLE_VERIFIED:', root, flush=True)
    print('Bundle fingerprint:', manifest['fingerprint_sha256'], flush=True)
    if args.check_only:
        print('CHECK ONLY: no environment created, packages installed or analysis performed.')
        return 0

    environment = Path(args.environment).expanduser().resolve()
    output_name = args.out or str(DEFAULT_ROOT / 'scratch' / (
        'Perilipin2_TrackA_windows_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_')
        + uuid.uuid4().hex[:8]))
    output, _ = core.new_output(output_name, (root, environment))
    for protected in (root, REPO.resolve(), Path(sys.prefix).resolve(), output):
        separate(environment, protected)
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if environment.exists():
        if not (environment / 'pyvenv.cfg').is_file() or not python.is_file():
            raise ValueError('Existing analysis folder is not a complete venv; it was left untouched')
    else:
        if not args.create_environment:
            raise ValueError('Analysis venv is absent. Use --create-environment to permit first-time setup')
        # The launcher/SDK Python may be older than the scientific stack.
        # Validate the explicitly selected base BEFORE mkdir, venv or pip.
        base, info = probe_analysis_python(getattr(args, 'base_python', None) or sys.executable)
        for protected in (Path(info['prefix']).resolve(), Path(info['base_prefix']).resolve()):
            separate(environment, protected)
        print('TRACKA_ANALYSIS_BASE_VERIFIED: {} | CPython {} | {}-bit'.format(
            base, '.'.join(map(str, info['version'])), info['bits']), flush=True)
        environment.mkdir(parents=True, exist_ok=False)
        print('Creating NEW analysis environment:', environment, flush=True)
        checked_call([base, '-X', 'utf8', '-m', 'venv', environment])
        checked_call([python, '-X', 'utf8', '-m', 'pip', 'install',
                      '--only-binary=:all:', '--disable-pip-version-check', *pins])

    check_environment(python, environment, pins)
    core.verify_bundle(root)
    print('TRACKA_SCIENTIFIC_ENVIRONMENT_VERIFIED:', python, flush=True)
    print('Running real RGB measurements with saved nuclear masks. CPU only.', flush=True)
    print('New output:', output, flush=True)
    checked_call([python, '-X', 'utf8', '-u', ADAPTER,
                  'run', '--bundle', root, '--out', output])
    checked_call([python, '-X', 'utf8', '-u', ADAPTER, 'verify', output])
    receipt = core.load_json(output / 'TRACKA_EXECUTION.json')
    comparison = core.load_json(output / 'REFERENCE_COMPARISON.json')
    if receipt.get('status') != 'TRACKA_REFERENCE_REPLAY_PASS' or comparison.get('passed') is not True:
        raise ValueError('Reference comparison did not pass; keep the output folder for diagnosis')
    print('TRACKA_WINDOWS_REFERENCE_ACCEPTANCE_PASS', flush=True)
    print('Measurement receipt:', output / 'TRACKA_EXECUTION.json')
    print('Exact-comparison report:', output / 'REFERENCE_COMPARISON.json')
    print('Open QC viewer:', output / 'index.html')
    print('No fresh nuclei segmentation, KL720 inference or biological validation claimed.')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=DEFAULT_ROOT / 'references' /
                        'Perilipin2_TrackA_portable_20261111_v1')
    parser.add_argument('--environment', type=Path, default=DEFAULT_ROOT / 'envs/perilipin2-tracka')
    parser.add_argument('--base-python', type=Path,
                        help='installed 64-bit CPython 3.11-3.14 for NEW venv creation; '
                             'defaults to the launcher interpreter after a compatibility check')
    parser.add_argument('--out', type=Path, help='new result folder; otherwise a unique scratch path')
    parser.add_argument('--create-environment', action='store_true',
                        help='permit installation only when creating a new isolated analysis venv')
    parser.add_argument('--check-only', action='store_true', help='verify copied bundle without changes')
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print('TRACKA_ACCEPTANCE_STOP: {}: {}'.format(type(exc).__name__, exc),
              file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
