#!/usr/bin/env python3
"""Compare the screened ONNX and generate a KL720 BIE for a trained NASA fold.

The host and vendor modes use the same pinned manifests and array reader.
Keep this module runnable as a standalone file in the vendor Python 3.7 image.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
import traceback
from collections import Counter, namedtuple
from datetime import datetime, timezone
from pathlib import Path

IMAGE_ID = 'sha256:2207d99c9f78deef90647a3da3047ec3f7942ec689385a586fee476f77e89041'
FOLD = 'fold_female_role'
ARCH = 'resnet18'
MODEL_ID = 32783
EXPECTED_SOURCE_SHA = '277ba0acd6c9e17200948d8f979cabe4fc56069b0e3885c7854d244667a706a2'
EXPECTED_EXPORT_SHA = 'fc12dcf5be78d45b2c221257f4e1f70b5017982d16bec7af17bfe886fce07af5'
EXPECTED_TRAIN_SHA = 'add7951f471ddf644a92fe3d457c83a5253cc5f9342a82e2fd3df25771ced451'
EXPECTED_SCREEN_SHA = 'dd2a540fe22a78edb7972a23b2635c60da7aff5d2fec235f2c196e1086172b76'
V1_POLICY_SHA = '20a685eba6502e5cb10d4506f39f8b155d6b29548b2df10915fe9dd5cd470db9'
FIXED_SIM_API = 'sys_flow.inference.inference_dynasty_fx'
FIXED_SIM_SHA = '111ff72391b1b6918e72a20e89c233b621b3846f6a54f3155629d394d547f2d2'

Case = namedtuple('Case', 'architecture model_id source export training screen')
MOBILENET_FEMALE = Case(
    'mobilenet_v2', 32781, '2f18b2330ca3289ab422e1ff47b526b4bfa6a46fc4bd7354346609715f575e45',
    '40b89d4b349dfa4060e4fd62a31cf27647b6fb04c1bdef93b4f7b84165f32257',
    '861e93ef0a128fdab8f81cb2d992a3fce8833aa92fdc09494c9008ebad4fe5b5',
    '9bab4fa38ef118a2da7cfbaa9f7fa710eb75f02c132c0e3d87be945c53dad983')


def trained_case(args):
    architecture = getattr(args, 'architecture', ARCH)
    if architecture == MOBILENET_FEMALE.architecture:
        return MOBILENET_FEMALE
    require(architecture == ARCH, 'Unrecognized trained PTQ architecture')
    return Case(ARCH, MODEL_ID, EXPECTED_SOURCE_SHA, EXPECTED_EXPORT_SHA,
                EXPECTED_TRAIN_SHA, EXPECTED_SCREEN_SHA)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as inp:
        for block in iter(lambda: inp.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    path = Path(path)
    return {'sha256': sha256(path), 'bytes': path.stat().st_size}


def load_json(path):
    with Path(path).open('r', encoding='utf-8-sig') as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError('Expected JSON object: {}'.format(path))
    return result


def save_json(path, data):
    target = Path(path)
    tmp = target.with_name(target.name + '.partial')
    with tmp.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    os.replace(str(tmp), str(target))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def calibration_path(root, name):
    """Read both frozen Windows paths and portable paths written by newer exports."""
    root = Path(root).resolve()
    name = name.replace('\\', '/')
    if not re.fullmatch(r'calibration_nchw/[0-9]{4}\.npy', name):
        raise ValueError('Invalid calibration array path: ' + name)
    candidate = (root / name).resolve()
    if candidate != root / name:
        raise ValueError('Calibration path escapes tensor directory: ' + name)
    return candidate


def checked_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return checked_csv(stream)


def checked_csv(stream):
    reader = csv.DictReader(stream)
    columns = reader.fieldnames or []
    require(columns and all(columns) and len(columns) == len(set(columns)),
            'Missing or duplicate CSV column')
    rows = list(reader)
    require(all(None not in row and None not in row.values() for row in rows),
            'Unexpected CSV row width')
    return rows


def read_calibration_array(path, expected_sha256):
    import numpy as np

    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha256,
            'Calibration .npy SHA256 mismatch: {}'.format(path.name))
    return np.load(io.BytesIO(raw), allow_pickle=False)


def preflight(args):
    import numpy as np

    case = trained_case(args)
    graph = Path(args.onnx)
    optimized = Path(args.optimized)
    screen_path = Path(args.screen)
    export_path = Path(args.export)
    manifest_path = Path(args.manifest)
    require(sha256(graph) == case.source, 'Source trained ONNX SHA256 changed')
    require(sha256(export_path) == case.export, 'Strict export receipt SHA256 changed')
    require(sha256(screen_path) == case.screen, 'Successful vendor screen receipt SHA256 changed')

    export = load_json(export_path)
    screen = load_json(screen_path)
    require(export.get('architecture') == case.architecture and export.get('fold') == FOLD,
            'Wrong trained architecture/fold in export')
    require(export.get('parity_pass') is True and export.get('calibration_arrays') == 192,
            'Original strict FP32 parity is incomplete')
    require(export.get('parity_atol') == 1e-5 and export.get('parity_rtol') == 1e-4,
            'Original FP32 parity thresholds changed')
    require(export.get('dataset_role') == 'development_train_ptq_calibration',
            'Calibration data role is not development train PTQ')
    require(export['training_run'].get('sha256') == case.training,
            'Training run identity changed')
    require(export['onnx'].get('sha256') == case.source,
            'Export ONNX identity changed')
    require(export['calibration'].get('sha256') == sha256(manifest_path),
            'Calibration manifest identity changed')
    require(export.get('contract', {}).get('input', {}).get('deployment_shape') == [1, 3, 256, 256]
            and export['contract']['input'].get('channels') == ['FITC_53BP1', 'DAPI', 'zeros'],
            'Input science contract changed')
    require(export.get('contract', {}).get('output', {}).get('semantics') ==
            'latent_continuous_per_nucleus_53bp1_burden', 'Output science contract changed')

    require(screen.get('purpose') == 'trained_fp32_exact_graph_feasibility_only'
            and screen.get('architecture') == case.architecture and screen.get('fold') == FOLD,
            'Unrecognized or wrong trained vendor screen')
    require(screen.get('trained') is True and screen.get('original_fp32_parity_pass') is True,
            'Screen did not verify trained strict-parity export')
    require(screen.get('support_status') == 'supported'
            and screen.get('vendor_evaluation_completed') is True,
            'Fresh successful target-720 IP evaluation is absent')
    require(screen.get('source_onnx_sha256') == case.source
            and screen.get('export_receipt_sha256') == case.export,
            'Vendor screen graph/receipt identity disagrees')
    require(screen.get('optimized_onnx_sha256') == sha256(optimized),
            'Optimized vendor-evaluated graph differs from screen')
    require(screen.get('toolchain_image_id') == IMAGE_ID
            and args.image_id == IMAGE_ID, 'Pinned Kneron toolchain image changed')
    require(screen.get('model_id') == case.model_id
            and screen.get('platform') == '720', 'Unexpected model ID or device platform')
    require(screen.get('ptq_performed') is False and screen.get('nef_compiled') is False,
            'Prior screen unexpectedly claims quantization or compilation')

    train_sources = set(export['source_split']['train_sources'])
    test_sources = set(export['source_split']['test_sources'])
    require(len(train_sources) == 4 and len(test_sources) == 2 and not train_sources & test_sources
            and export['source_split']['train_bags'] == 48
            and export['source_split']['test_bags'] == 24,
            'Unexpected frozen source-held-out grouping')
    rows = checked_rows(manifest_path)
    require(len(rows) == 192, 'PTQ requires exactly 192 training-source calibration arrays')
    seen_id, seen_array, sample_counts, source_counts = set(), set(), Counter(), Counter()
    file_ids = set()
    arrays = []
    calibration_root = Path(args.calibration_root).resolve()
    for index, row in enumerate(rows):
        require(row.get('dataset_role') == 'development_train_ptq_calibration'
                and row.get('calibration_fold') == FOLD
                and row.get(FOLD) == 'train'
                and row.get('source_name') in train_sources
                and row.get('source_name') not in test_sources,
                'Array {} violates train-only provenance'.format(index))
        key = row['sample_id']
        relative = row['array_file']
        require(key.strip() and row['sample_name'].strip(), 'Missing training nucleus or bag identity')
        require(key not in seen_id, 'Duplicate nucleus ID')
        candidate = calibration_path(calibration_root, relative)
        require(candidate not in seen_array, 'Duplicate calibration array filename')
        seen_id.add(key)
        seen_array.add(candidate)
        require(candidate.is_file(), 'Missing calibration array: {}'.format(relative))
        info = candidate.stat()
        if info.st_ino:
            file_id = (info.st_dev, info.st_ino)
            require(file_id not in file_ids, 'Duplicate calibration file identity')
            file_ids.add(file_id)
        arr = read_calibration_array(candidate, row['array_sha256'])
        require(arr.shape == (1, 3, 256, 256) and arr.dtype == np.float32,
                'PTQ input does not match native NCHW float32 science contract')
        require(bool(np.isfinite(arr).all()), 'Nonfinite PTQ input')
        require(float(arr.min()) >= -1e-7 and float(arr.max()) <= 1.0000001,
                'PTQ input normalized pixel range changed')
        require(bool(np.all(arr[:, 2, :, :] == 0.0)), 'Third input channel is not zero')
        arrays.append(arr)
        sample_counts[row['sample_name']] += 1
        source_counts[row['source_name']] += 1
    require(len(sample_counts) == 48 and all(c == 4 for c in sample_counts.values()),
            'Expected 4 PTQ nuclei from each of 48 training bags')
    require(set(source_counts) == train_sources and all(c == 48 for c in source_counts.values()),
            'Expected exactly 48 calibration nuclei from each train source')
    return {
        'rows': rows, 'arrays': arrays,
        'input_record': {
            'source_onnx': identity(graph),
            'optimized_onnx': identity(optimized),
            'screen': identity(screen_path),
            'export': identity(export_path),
            'calibration_manifest': identity(manifest_path),
            'calibration_arrays': len(arrays),
            'train_sources': sorted(train_sources),
            'test_sources': sorted(test_sources),
            'train_bags': 48,
            'nuclei_per_train_bag': 4,
            'array_layout': 'NCHW',
            'array_shape': [1, 3, 256, 256],
            'array_dtype': 'float32',
            'input_name': 'input',
        },
    }


def parity(args):
    import numpy as np
    import onnxruntime as ort

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / 'fp32_optimized_parity.csv'
    report = {
        'purpose': 'trained_original_to_vendor_optimized_fp32_parity',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'FAILED',
        'atol': 1e-5, 'rtol': 1e-4, 'pass_count': 0, 'fail_count': 0,
        'inputs': 0, 'max_absolute_error': None, 'error': None,
    }
    try:
        check = preflight(args)
        report.update(source_sha256=sha256(args.onnx), optimized_sha256=sha256(args.optimized),
                      manifest_sha256=sha256(args.manifest), inputs=len(check['rows']))
        sessions = [ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
                    for path in (args.onnx, args.optimized)]
        for session in sessions:
            require([(x.name, x.shape, x.type) for x in session.get_inputs()] ==
                    [('input', [1, 3, 256, 256], 'tensor(float)')], 'Changed FP32 input contract')
            require([(x.name, x.shape, x.type) for x in session.get_outputs()] ==
                    [('burden', [1], 'tensor(float)')], 'Changed FP32 output contract')
        fields = ('sample_id', 'source_name', 'sample_name', 'source_fp32', 'optimized_fp32',
                  'absolute_error', 'parity_pass')
        with csv_path.open('x', newline='', encoding='utf-8') as stream:
            table = csv.DictWriter(stream, fieldnames=fields)
            table.writeheader()
            for row, array in zip(check['rows'], check['arrays']):
                original, optimized = [s.run(['burden'], {'input': array})[0] for s in sessions]
                require(all(x.shape == (1,) and x.dtype == np.float32 and np.isfinite(x).all()
                            for x in (original, optimized)), 'Invalid FP32 burden output')
                # Preserve operand order and tolerances used in the original parity checker.
                passed = bool(np.allclose(original, optimized, atol=1e-5, rtol=1e-4))
                error = float(np.abs(original - optimized).max())
                table.writerow(dict(sample_id=row['sample_id'], source_name=row['source_name'],
                                    sample_name=row['sample_name'], source_fp32=float(original[0]),
                                    optimized_fp32=float(optimized[0]), absolute_error=error,
                                    parity_pass=passed))
                report['pass_count' if passed else 'fail_count'] += 1
                report['max_absolute_error'] = max(report['max_absolute_error'] or 0.0, error)
        report['predictions_sha256'] = sha256(csv_path)
        report['status'] = 'PASS' if report['pass_count'] == 192 else 'PARITY_FAILED'
    except Exception as exc:
        report['error'] = '{}: {}'.format(type(exc).__name__, exc)
        traceback.print_exc()
    finally:
        save_json(output / 'fp32_optimized_parity.json', report)
        print('Original vs optimized FP32: {} ({} pass, {} fail, max error {})'.format(
            report['status'], report['pass_count'], report['fail_count'],
            report['max_absolute_error']), flush=True)
    return 0 if report['status'] == 'PASS' else 1


def verify_parity_receipt(path, args):
    """Bind vendor analysis to the complete host parity table and its exact inputs."""
    import numpy as np

    report = load_json(path)
    expected = {'purpose': 'trained_original_to_vendor_optimized_fp32_parity',
                'status': 'PASS', 'inputs': 192, 'pass_count': 192, 'fail_count': 0,
                'atol': 1e-5, 'rtol': 1e-4, 'source_sha256': sha256(args.onnx),
                'optimized_sha256': sha256(args.optimized), 'manifest_sha256': sha256(args.manifest),
                'predictions_sha256': sha256(Path(path).with_name('fp32_optimized_parity.csv'))}
    for key, value in expected.items():
        require(report.get(key) == value, 'Invalid optimized FP32 parity receipt: ' + key)
    inputs = checked_rows(args.manifest)
    rows = checked_rows(Path(path).with_name('fp32_optimized_parity.csv'))
    require(len(rows) == len(inputs) == 192, 'Incomplete optimized FP32 parity table')
    maximum = 0.0
    for row, sample in zip(rows, inputs):
        require(all(row[key] == sample[key] for key in ('sample_id', 'sample_name', 'source_name')),
                'Optimized FP32 parity row identity mismatch')
        original, optimized = [np.float32(row[key]) for key in ('source_fp32', 'optimized_fp32')]
        error = float(abs(original - optimized))
        require(np.isfinite([original, optimized]).all() and row['parity_pass'] == 'True'
                and np.allclose(original, optimized, atol=1e-5, rtol=1e-4)
                and error == float(row['absolute_error']), 'Invalid optimized FP32 parity row')
        maximum = max(maximum, error)
    require(maximum == report.get('max_absolute_error'), 'Inconsistent FP32 maximum error')


def vendor_model(path):
    import onnx
    model = onnx.load(str(path), load_external_data=False)
    require(not any(t.external_data for t in model.graph.initializer),
            'Optimized ONNX uses external tensor data')
    onnx.checker.check_model(model)
    return model


def run(args):
    case = trained_case(args)
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / 'ptq_receipt.json'
    report = {
        'schema_version': 1,
        'purpose': 'verified_development_train_ptq_pilot',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'architecture': case.architecture, 'fold': FOLD, 'model_id': case.model_id,
        'model_version': '0001', 'platform': '720',
        'toolchain_image_id': args.image_id,
        'python_executable': sys.executable,
        'status': 'STARTED', 'ptq_performed': False, 'bie_generated': False,
        'bie_simulator_performed': False, 'nef_compiled': False,
        'physical_device_used': False, 'biological_accuracy_evaluated': False,
        'sources': 'development_train_only',
        'input': None, 'bie': None, 'error': None,
    }
    save_json(report_path, report)
    started = time.monotonic()
    try:
        import ktc
        verify_parity_receipt(args.parity_receipt, args)
        check = preflight(args)
        report['input'] = check['input_record']
        report['optimized_fp32_parity_receipt_sha256'] = sha256(args.parity_receipt)
        report['status'] = 'PREFLIGHT_PASS'
        save_json(report_path, report)
        print('TRAINED PTQ INPUT VALIDATION: PASS (192 NCHW arrays / 48 train bags)', flush=True)
        km = ktc.ModelConfig(case.model_id, '0001', '720', onnx_model=vendor_model(args.optimized))
        # Same training-only, raw, normalized arrays used in the strict export;
        # vendor default analysis mode as in the pinned toolchain documentation.
        print('Starting Kneron ModelConfig.analysis; no NEF compile.', flush=True)
        report['ptq_performed'] = True
        report['status'] = 'ANALYSIS_STARTED'
        save_json(report_path, report)
        result_path = km.analysis({'input': check['arrays']}, threads=2)
        if not result_path:
            raise RuntimeError('Vendor analysis returned no BIE path')
        src = Path(str(result_path)).resolve()
        mount_root = Path('/data1').resolve()
        require(src.is_file(), 'Returned BIE file does not exist: {}'.format(src))
        require(mount_root == src.parent or mount_root in src.parents,
                'Vendor BIE was written outside the isolated /data1 mount')
        require(src.suffix.lower() == '.bie', 'Vendor result is not a .bie file')
        require(src.stat().st_size > 0, 'Vendor created an empty BIE')
        dest = output / 'quantized_model.bie'
        if src != dest:
            shutil.copyfile(str(src), str(dest))
        report['bie'] = dict(identity(dest), path=str(dest))
        report['bie_generated'] = True
        # Validate persisted input digests after vendor analysis; Docker binds are RO.
        for path, ref in ((args.onnx, report['input']['source_onnx']),
                          (args.optimized, report['input']['optimized_onnx']),
                          (args.screen, report['input']['screen']),
                          (args.export, report['input']['export']),
                          (args.manifest, report['input']['calibration_manifest'])):
            require(sha256(path) == ref['sha256'], 'Input identity changed during PTQ')
        report['status'] = 'PTQ_GENERATED_UNVALIDATED'
        print('PTQ BIE GENERATED (unvalidated): {}'.format(dest), flush=True)
        print('BIE SHA256: {}'.format(report['bie']['sha256']), flush=True)
    except (Exception, SystemExit) as exc:
        report['status'] = 'FAILED'
        report['error'] = '{}: {}'.format(type(exc).__name__, exc)
        traceback.print_exc()
        print('TRAINED PTQ PILOT FAILED: {}'.format(report['error']), flush=True)
    finally:
        report['duration_seconds'] = time.monotonic() - started
        report['completed_utc'] = datetime.now(timezone.utc).isoformat()
        save_json(report_path, report)
        print('PTQ receipt: {}'.format(report_path), flush=True)
    return 0 if report['status'] == 'PTQ_GENERATED_UNVALIDATED' else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['parity', 'quantize'])
    p.add_argument('--architecture', choices=[ARCH, MOBILENET_FEMALE.architecture], default=ARCH)
    p.add_argument('--onnx', required=True)
    p.add_argument('--optimized', required=True)
    p.add_argument('--screen', required=True)
    p.add_argument('--export', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--calibration-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--image-id', required=True)
    p.add_argument('--parity-receipt', type=Path)
    args = p.parse_args()
    if args.stage == 'parity':
        return parity(args)
    if args.parity_receipt is None:
        p.error('quantize requires --parity-receipt')
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
