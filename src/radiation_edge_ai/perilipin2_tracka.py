"""Portable, source-locked Perilipin2 Track A M2 measurement/replay adapter.

Reads real RGB pixels and frozen M1 nuclear/exclusion masks, reruns the unchanged
M2 object measurements and QC, and compares them with frozen M2 tables/arrays.
No learned weights, training, new nuclear segmentation, Docker or KL720 access.
The original source archive is an external, hash-pinned dependency in the bundle;
no private images, masks, scientific tables or model binaries belong in Git.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace

HANDOFF_SHA256 = 'caa052667320eff5784933844dcc25e5877f0e0d9922037953b2a0a5eb58afde'
M2_ZIP_SHA256 = '68d3a682d63eb6497b0105ac0f10199bf3a050487fbb0a1bbc93f63fea13fdaf'
SOURCE_SHA256 = 'b92659b7784233695ebc708da34e975b1cc39195c64d45076a76b585aa96dd9c'
VIEWER_SHA256 = 'a86a75f15acfd7ea5bacaa2d6a91f2cad2c490fc7d7be51621f1151461a4ca6e'
BUNDLE_SCHEMA = 'radedge-perilipin2-tracka-bundle-v1'
RUN_SCHEMA = 'radedge-perilipin2-tracka-run-v1'
VERSION = 'PLIN2-TRACKA-M2.0'
RULES = {'permissive', 'working', 'strict'}
TABLE_KEYS = {
    'T01_field_measurements.tsv': ('image_uid', 'rule'),
    'T02_brown_object_catalog.tsv': ('image_uid', 'rule', 'object_id'),
    'T03_independent_nuclear_audit.tsv': ('image_uid', 'nucleus_id'),
    'T04_random_patch_review_plan.tsv': ('image_uid', 'patch_id'),
    'T05_input_provenance.tsv': ('image_uid',),
}
ARRAYS = ('objects_permissive.tif', 'objects_working.tif', 'objects_strict.tif',
          'hematoxylin_od.tif', 'DAB_od.tif', 'tissue.tif', 'analysis_valid.tif')
RELOCATED_COLUMNS = {'source_path', 'mask_path', 'valid_mask_path'}
ENVIRONMENT = ('numpy', 'pandas', 'scipy', 'scikit-image', 'Pillow', 'tifffile')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      allow_nan=False, separators=(',', ':')).encode('utf-8')


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def load_json(path):
    def reject(value):
        raise ValueError('Non-finite JSON: ' + value)
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object: ' + str(path))
    return value


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, sort_keys=True, ensure_ascii=True, indent=2, allow_nan=False)
        handle.write('\n')


def contained(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError('Missing relative asset path')
    p, w = PurePosixPath(relative), PureWindowsPath(relative)
    if p.is_absolute() or w.is_absolute() or w.drive or '\\' in relative or '..' in p.parts:
        raise ValueError('Unsafe asset path: ' + relative)
    result = root / p
    if result.is_symlink() or not result.resolve().is_relative_to(root.resolve()):
        raise ValueError('Asset escapes root or is a symlink: ' + relative)
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def new_output(path, protected=()):
    output = Path(path).expanduser().resolve()
    repo = Path(__file__).resolve().parents[2]
    for root in (repo, *protected):
        root = Path(root).expanduser().resolve()
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError('Output must be separate from source/repository trees: ' + str(output))
    stage = output.with_name(output.name + '.BUILDING')
    if output.exists() or output.is_symlink() or stage.exists() or stage.is_symlink():
        raise FileExistsError('Preserve existing output; choose a new folder: ' + str(output))
    return output, stage


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError('Cannot load frozen source')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def handoff_members(path):
    if digest(path) != HANDOFF_SHA256:
        raise ValueError('Unrecognized microscopy handoff ZIP; do not substitute new analysis code')
    with zipfile.ZipFile(path) as archive:
        base = 'edgeai_microscopy_handoff/'
        source = archive.read(base + 'reference_packages/Perilipin2_TrackA_M2.zip')
        inspector = archive.read(base + 'tools/inspect_perilipin2_reference.py')
    if hashlib.sha256(source).hexdigest() != M2_ZIP_SHA256:
        raise ValueError('Frozen M2 source archive hash mismatch')
    return source, inspector


def unpack_reference(source_zip, destination):
    """Only known filenames from a byte-identified source archive are extracted."""
    if isinstance(source_zip, (str, Path)):
        source_zip = Path(source_zip).read_bytes()
    if hashlib.sha256(source_zip).hexdigest() != M2_ZIP_SHA256:
        raise ValueError('Frozen M2 source archive hash mismatch')
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(source_zip)) as archive:
        for name in ('run_tracka.py', 'config.json', 'viewer.html', 'METHODS.txt'):
            (destination / name).write_bytes(archive.read('Perilipin2_TrackA_M2/' + name))
    if digest(destination / 'run_tracka.py') != SOURCE_SHA256:
        raise ValueError('Frozen numerical source mismatch')
    if digest(destination / 'viewer.html') != VIEWER_SHA256:
        raise ValueError('Frozen QC viewer mismatch')
    return destination


def file_record(path):
    return {'bytes': Path(path).stat().st_size, 'sha256': digest(path)}


def copy_asset(source, root, relative, records):
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    before = file_record(source)
    shutil.copyfile(source, target)
    if file_record(target) != before or file_record(source) != before:
        raise ValueError('Source changed while packing: ' + str(source))
    records[relative] = before


def prepare(args):
    """Export the exact two-field M1/M2 reference as a relocatable local bundle."""
    roots = [Path(x).expanduser().resolve(strict=True) for x in (args.m1, args.m2, args.images)]
    m1, m2, images = roots
    output, stage = new_output(args.out, roots)
    source, inspector = handoff_members(args.handoff)
    with tempfile.TemporaryDirectory(prefix='tracka-check-') as td:
        helper = Path(td) / 'reference_preflight.py'
        helper.write_bytes(inspector)
        audit = import_file(helper, '_tracka_source_preflight').inspect_reference(m1, m2, images)
    original = load_json(m2 / 'RUN_CONFIG.json')
    provenance = original['provenance']
    if provenance.get('script_sha256') != SOURCE_SHA256 or provenance.get('viewer_sha256') != VIEWER_SHA256:
        raise ValueError('M2 outputs were produced by different code/viewer; preserve and inspect')
    cfg = original['configuration']
    if set(cfg.get('rules', {})) != RULES or cfg.get('working_rule') != 'working':
        raise ValueError('Expected recorded M2 permissive/working/strict rules')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir(exist_ok=False)
    files = {}
    fields = []
    try:
        source_path = stage / 'reference_source.zip'
        source_path.write_bytes(source)
        files['reference_source.zip'] = file_record(source_path)
        by_uid = {f['image_uid']: f for f in audit['fields']}
        for record in provenance['inputs']:
            uid = record['image_uid']
            field = by_uid[uid]
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', uid) or uid in ('.', '..'):
                raise ValueError('Unsafe field identifier')
            assets = {}
            for key, source_file, name in (
                ('rgb', field['source']['path'], 'image' + Path(field['source']['path']).suffix.lower()),
                ('nuclei', field['m1_artifacts']['nuclei.tif']['path'], 'nuclei.tif'),
                ('valid', field['m1_artifacts']['analysis_valid_mask.tif']['path'], 'analysis_valid_mask.tif'),
            ):
                relative = 'inputs/' + uid + '/' + name
                copy_asset(source_file, stage, relative, files)
                assets[key] = relative
            fields.append({'image_uid': uid, 'original_record': record, 'assets': assets})
            for name in ARRAYS:
                relative = 'per_field/' + uid + '/' + name
                if relative not in audit['m2_manifest_assets']:
                    raise ValueError('Reference file manifest omits: ' + relative)
                copy_asset(contained(m2, relative), stage, 'expected/' + relative, files)
        for name in TABLE_KEYS:
            relative = 'tables/' + name
            copy_asset(contained(m2, relative), stage, 'expected/' + relative, files)
        for name in ('RUN_CONFIG.json', 'RUN_COMPLETE.json', 'FILE_MANIFEST.tsv'):
            copy_asset(m2 / name, stage, 'reference/' + name, files)
        requirement_lines = []
        for package in ENVIRONMENT:
            version = provenance['versions'].get(package)
            if not isinstance(version, str) or not re.fullmatch(r'[0-9][A-Za-z0-9.!+_-]*', version):
                raise ValueError('Invalid recorded dependency version: ' + package)
            requirement_lines.append(package + '==' + version)
        (stage / 'reference_requirements.txt').write_text(
            '\n'.join(requirement_lines) + '\n', encoding='utf-8')
        files['reference_requirements.txt'] = file_record(stage / 'reference_requirements.txt')
        write_json(stage / 'SOURCE_AUDIT.json', audit)
        files['SOURCE_AUDIT.json'] = file_record(stage / 'SOURCE_AUDIT.json')
        manifest = {
            'schema': BUNDLE_SCHEMA, 'purpose': 'frozen_two_field_reference_replay',
            'source_dataset_id': audit['dataset_id'], 'configuration': cfg,
            'reference_versions': provenance['versions'], 'fields': fields, 'files': files,
            'reference_source_sha256': SOURCE_SHA256,
            'nuclear_backend': 'saved_M1_proposals_no_new_segmentation',
            'kl720_inference': False, 'biological_validation': False,
        }
        manifest['fingerprint_sha256'] = fingerprint(manifest)
        write_json(stage / 'TRACKA_INPUT_BUNDLE.json', manifest)
        stage.rename(output)
    except Exception as exc:
        if stage.exists():
            write_json(stage / 'PACK_FAILED.json', {'error': str(exc)})
        raise
    print('TRACKA_PORTABLE_REFERENCE_READY:', output)
    print('Two original RGB fields, frozen masks, fixed rules, five reference tables and fourteen arrays.')
    print('COPY THIS ENTIRE FOLDER to Windows. No new measurements or nuclear inference yet.')
    return 0


def bundle_manifest(path):
    p = Path(path).expanduser().resolve(strict=True)
    return p / 'TRACKA_INPUT_BUNDLE.json' if p.is_dir() else p


def verify_bundle(path):
    path = bundle_manifest(path)
    root = path.parent
    manifest = load_json(path)
    expected = manifest.get('fingerprint_sha256')
    payload = {k: v for k, v in manifest.items() if k != 'fingerprint_sha256'}
    if manifest.get('schema') != BUNDLE_SCHEMA or fingerprint(payload) != expected:
        raise ValueError('Invalid bundle identity or modified manifest')
    files = manifest.get('files', {})
    if not isinstance(files, dict) or not files:
        raise ValueError('Empty bundle file manifest')
    for relative, record in files.items():
        if file_record(contained(root, relative)) != record:
            raise ValueError('Bundle artifact was modified: ' + relative)
    if file_record(contained(root, 'reference_source.zip'))['sha256'] != M2_ZIP_SHA256:
        raise ValueError('Wrong frozen M2 code')
    fields = manifest.get('fields', [])
    if len(fields) != 2 or len({f['image_uid'] for f in fields}) != 2:
        raise ValueError('This release accepts exactly the two frozen reference fields')
    pairs = {(f['original_record']['dose_Gy'], f['original_record']['sample_token']) for f in fields}
    if pairs != {('0', '13'), ('3.8', '27')}:
        raise ValueError('Reference field identities changed')
    if set(manifest['configuration']['rules']) != RULES:
        raise ValueError('Reference rule names changed')
    required = {'reference_source.zip'} | {'expected/tables/' + n for n in TABLE_KEYS}
    for field in fields:
        uid = field['image_uid']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', uid) or uid in ('.', '..'):
            raise ValueError('Unsafe image identifier')
        required.update(field['assets'].values())
        required.update('expected/per_field/' + uid + '/' + n for n in ARRAYS)
        for asset, key in (('rgb', 'source_sha256'), ('nuclei', 'mask_sha256'),
                           ('valid', 'valid_mask_sha256')):
            relative = field['assets'][asset]
            if files.get(relative, {}).get('sha256') != field['original_record'][key]:
                raise ValueError('Image/mask provenance mismatch: ' + uid)
    if not required.issubset(files):
        raise ValueError('Required bundle files are missing from its manifest')
    return root, manifest


def normalize_cell(text):
    if text == '' or text.lower() == 'nan':
        return ('missing', None)
    if text in ('True', 'False'):
        return ('bool', text == 'True')
    try:
        number = Decimal(text)
        if number.is_finite():
            return ('number', number)
    except InvalidOperation:
        pass
    return ('text', text)


def read_table(path, keys):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        columns = reader.fieldnames or []
        rows = list(reader)
        if columns == ['status'] and not rows:
            return columns, {}
        if len(columns) != len(set(columns)) or not set(keys).issubset(columns):
            raise ValueError('Missing/duplicate table columns: ' + str(path))
        indexed = {}
        for row in rows:
            if None in row or any(v is None for v in row.values()):
                raise ValueError('Malformed TSV row')
            key = tuple(normalize_cell(row[k]) for k in keys)
            if key in indexed:
                raise ValueError('Duplicate scientific row key in ' + str(path))
            indexed[key] = row
    return columns, indexed


def compare(bundle_root, run_root, manifest):
    import numpy as np
    import tifffile
    report = {'comparison': 'exact_numeric_values_and_decoded_arrays_no_tolerance',
              'tables': {}, 'arrays': {}, 'differences': [], 'difference_count': 0,
              'allowed_path_relocations': 0, 'passed': True}
    def problem(message):
        report['passed'] = False
        report['difference_count'] += 1
        if len(report['differences']) < 40:
            report['differences'].append(message)
    fields = {f['image_uid']: f for f in manifest['fields']}
    for name, keys in TABLE_KEYS.items():
        left = bundle_root / 'expected' / 'tables' / name
        right = run_root / 'tables' / name
        try:
            cols_a, a = read_table(left, keys)
            cols_b, b = read_table(right, keys)
            before = report['difference_count']
            if set(cols_a) != set(cols_b) or set(a) != set(b):
                problem(name + ': columns or row keys changed')
            else:
                for key, row in a.items():
                    for col in cols_a:
                        if name == 'T05_input_provenance.tsv' and col in RELOCATED_COLUMNS:
                            asset = {'source_path': 'rgb', 'mask_path': 'nuclei', 'valid_mask_path': 'valid'}[col]
                            exact = str(contained(bundle_root, fields[row['image_uid']]['assets'][asset]))
                            if b[key][col] != exact:
                                problem(name + ': unexpected relocated path')
                            elif row[col] != b[key][col]:
                                report['allowed_path_relocations'] += 1
                        elif normalize_cell(row[col]) != normalize_cell(b[key][col]):
                            problem(name + ': key=' + repr(key) + ', column=' + col)
            report['tables'][name] = {'reference_rows': len(a), 'actual_rows': len(b),
                                      'passed': report['difference_count'] == before}
        except (ValueError, OSError) as exc:
            problem(name + ': ' + str(exc))
            report['tables'][name] = {'passed': False, 'error': str(exc)}
    for uid in fields:
        for name in ARRAYS:
            relative = 'per_field/' + uid + '/' + name
            try:
                a = tifffile.imread(bundle_root / 'expected' / relative)
                b = tifffile.imread(run_root / relative)
                ok = a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
                report['arrays'][relative] = {'passed': bool(ok), 'dtype': str(b.dtype),
                                              'shape': list(b.shape)}
                if not ok:
                    problem(relative + ': decoded dtype, shape or values differ')
            except (ValueError, OSError) as exc:
                problem(relative + ': ' + str(exc))
                report['arrays'][relative] = {'passed': False, 'error': str(exc)}
    return report


def source_records(root, manifest, core):
    import numpy as np
    import tifffile
    records = []
    for field in manifest['fields']:
        record = copy.deepcopy(field['original_record'])
        for key, asset in (('source_path', 'rgb'), ('mask_path', 'nuclei'), ('valid_mask_path', 'valid')):
            record[key] = str(contained(root, field['assets'][asset]))
        a = core.load_rgb(record['source_path'])
        n = tifffile.imread(record['mask_path'])
        valid = tifffile.imread(record['valid_mask_path'])
        if a.shape[:2] != n.shape or n.shape != valid.shape or n.ndim != 2:
            raise ValueError('RGB/nucleus/exclusion coordinate grids disagree')
        if n.dtype.kind not in 'ui' or np.any(n < 0) or np.any(n > np.iinfo(np.uint32).max):
            raise ValueError('Invalid instance-label dtype/range')
        if not np.isin(valid, [0, 1, 255]).all():
            raise ValueError('Exclusion mask must contain only 0/1/255')
        if [record['height'], record['width']] != list(n.shape):
            raise ValueError('Recorded source dimensions disagree')
        px = record.get('pixel_size_um')
        if px is not None and (not math.isfinite(float(px)) or float(px) <= 0 or
                               not record.get('calibration_source')):
            raise ValueError('Physical calibration requires a positive value AND recorded source')
        records.append(record)
    return records


def versions():
    import importlib.metadata
    return {name: importlib.metadata.version(name) for name in ENVIRONMENT}


def run(args):
    root, manifest = verify_bundle(args.bundle)
    output, stage = new_output(args.out, (root,))
    output.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir(exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HIP_VISIBLE_DEVICES'] = ''
    try:
        with tempfile.TemporaryDirectory(prefix='frozen-tracka-', dir=output.parent) as td:
            source = unpack_reference(root / 'reference_source.zip', Path(td) / 'source')
            core = import_file(source / 'run_tracka.py', '_radedge_tracka_m2')
            cfg = copy.deepcopy(manifest['configuration'])
            core.validate_config(cfg)
            records = source_records(root, manifest, core)
            current_versions = versions()
            identity = {'bundle_sha256': manifest['fingerprint_sha256'],
                        'source_sha256': SOURCE_SHA256, 'adapter_sha256': digest(__file__),
                        'configuration': cfg, 'versions': current_versions}
            dataset = fingerprint(identity)
            core.write_json(stage / 'RUN_CONFIG.json', {
                'version': VERSION, 'dataset_id': dataset, 'configuration': cfg,
                'provenance': {'script_sha256': SOURCE_SHA256, 'viewer_sha256': VIEWER_SHA256,
                               'inputs': records, 'versions': current_versions},
                'status': 'STARTED', 'reference_dataset_id': manifest['source_dataset_id']})
            fields, objects, nuclei, sums, patches = [], [], [], [], []
            for i, record in enumerate(records, 1):
                print('[Track A measurement {}/{}] {}'.format(i, len(records), record['image_uid']), flush=True)
                field, o, n, s, p = core.process_field(
                    record, cfg, stage / 'per_field' / record['image_uid'])
                field['alias'] = 'Field ' + chr(64 + i)
                fields.append(field)
                objects.extend(o)
                nuclei.extend(n)
                sums.extend(s)
                patches.extend(p)
            table_data = (sums, objects, nuclei, patches, records)
            for name, rows in zip(TABLE_KEYS, table_data):
                core.save_table(stage / 'tables' / name, rows,
                                core.OBJ_COLS if name.startswith('T02') else None)
            payload = {'version': VERSION, 'dataset_id': dataset, 'config': cfg, 'fields': fields,
                       'status': 'CALIBRATION_AND_IMAGE_REVIEW_NOT_VALIDATED_OBJECT_COUNTS'}
            serialized = json.dumps(core.clean(payload), ensure_ascii=False, separators=(',', ':'),
                                    allow_nan=False).replace('</', '<\\/')
            template = (source / 'viewer.html').read_text(encoding='utf-8')
            (stage / 'index.html').write_text(template.replace('__DATA_JSON__', serialized), encoding='utf-8')
            shutil.copyfile(source / 'METHODS.txt', stage / 'METHODS_FROZEN_M2.txt')
            core.write_json(stage / 'RUN_COMPLETE.json', {'dataset_id': dataset, 'n_fields': len(fields),
                            'source_hashes_verified': True, 'n_all_rule_components': len(objects)})
            comparison = compare(root, stage, manifest)
            write_json(stage / 'REFERENCE_COMPARISON.json', comparison)
            verify_bundle(root)
            state = 'TRACKA_REFERENCE_REPLAY_PASS' if comparison['passed'] else 'TRACKA_REFERENCE_MISMATCH'
            receipt = {'schema': RUN_SCHEMA, 'status': state, 'dataset_id': dataset,
                       'reference_dataset_id': manifest['source_dataset_id'], **identity,
                       'reference_versions': manifest['reference_versions'],
                       'saved_nuclear_masks_reused': True, 'new_nuclear_segmentation': False,
                       'kl720_inference': False, 'biological_validation': False,
                       'brown_objects_recomputed_from_rgb': True,
                       'n_fields': len(fields), 'n_rules': len(cfg['rules']),
                       'n_tables_compared': len(comparison['tables']),
                       'n_arrays_compared': len(comparison['arrays']),
                       'finished_utc': datetime.now(timezone.utc).isoformat()}
            write_json(stage / 'TRACKA_EXECUTION.json', receipt)
            core.manifest_files(stage)
        stage.rename(output)
    except Exception as exc:
        if stage.exists() and not (stage / 'TRACKA_FAILED.json').exists():
            write_json(stage / 'TRACKA_FAILED.json', {'status': 'FAILED', 'error': str(exc)})
        raise
    print(state)
    print('Tables:', output / 'tables')
    print('Original-image QC/review viewer:', output / 'index.html')
    print('Saved nuclei only; no KL720, new segmentation or biological validation claimed.')
    return 0 if comparison['passed'] else 2


def verify_result(folder):
    root = Path(folder).expanduser().resolve(strict=True)
    receipt = load_json(root / 'TRACKA_EXECUTION.json')
    if receipt.get('schema') != RUN_SCHEMA:
        raise ValueError('Unknown Track A run receipt')
    with (root / 'FILE_MANIFEST.tsv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    seen = set()
    for row in rows:
        rel = row['relative_path']
        if rel in seen or rel == 'FILE_MANIFEST.tsv':
            raise ValueError('Duplicate/self-referencing output manifest')
        seen.add(rel)
        path = contained(root, rel)
        if digest(path) != row['sha256'] or path.stat().st_size != int(row['bytes']):
            raise ValueError('Modified result artifact: ' + rel)
    required = {'TRACKA_EXECUTION.json', 'REFERENCE_COMPARISON.json', 'RUN_CONFIG.json',
                'RUN_COMPLETE.json', 'index.html'} | {'tables/' + n for n in TABLE_KEYS}
    if not required.issubset(seen):
        raise ValueError('Output manifest omits required measurements/evidence')
    comparison = load_json(root / 'REFERENCE_COMPARISON.json')
    if receipt['status'] == 'TRACKA_REFERENCE_REPLAY_PASS' and comparison.get('passed') is not True:
        raise ValueError('Contradictory replay receipt')
    print('TRACKA_OUTPUT_INTEGRITY_PASS:', len(rows), 'files;', receipt['status'])
    return receipt


def score(args):
    if os.name == 'nt' and not sys.flags.utf8_mode:
        raise ValueError('Run scoring with Python -X utf8 to preserve the frozen UTF-8 review reader')
    root, manifest = verify_bundle(args.bundle)
    run_root = Path(args.run).expanduser().resolve(strict=True)
    receipt = verify_result(run_root)
    if receipt['bundle_sha256'] != manifest['fingerprint_sha256']:
        raise ValueError('Scoring bundle differs from this analysis run')
    if not math.isfinite(args.radius) or args.radius <= 0:
        raise ValueError('A positive, explicitly chosen match radius in source pixels is required')
    output, _ = new_output(args.out, (root, run_root))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='tracka-review-', dir=output.parent) as td:
        source = unpack_reference(root / 'reference_source.zip', Path(td) / 'source')
        core = import_file(source / 'run_tracka.py', '_radedge_tracka_m2_score')
        core.evaluate(SimpleNamespace(out=run_root, review=Path(args.review),
                                      evaluation_out=output, match_radius=args.radius))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('prepare', help='export source-verified two-field reference bundle')
    for option in ('handoff', 'm1', 'm2', 'images', 'out'):
        p.add_argument('--' + option, type=Path, required=True)
    p = subs.add_parser('run', help='recompute real brown-object measurements with saved nuclei')
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = subs.add_parser('verify-bundle')
    p.add_argument('bundle', type=Path)
    p = subs.add_parser('verify')
    p.add_argument('output', type=Path)
    p = subs.add_parser('score', help='score only explicitly exhaustive patch annotations')
    for option in ('bundle', 'run', 'review', 'out'):
        p.add_argument('--' + option, type=Path, required=True)
    p.add_argument('--radius', type=float, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            return prepare(args)
        if args.command == 'run':
            return run(args)
        if args.command == 'score':
            return score(args)
        if args.command == 'verify':
            verify_result(args.output)
        else:
            root, manifest = verify_bundle(args.bundle)
            print('TRACKA_BUNDLE_INTEGRITY_PASS:', root, manifest['fingerprint_sha256'])
        return 0
    except Exception as exc:
        print('TRACKA_STOP: {}: {}'.format(type(exc).__name__, exc), file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
