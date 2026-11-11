"""Blind, image-native QC review for the frozen Perilipin2 Track A reference.

PREPARE crops only original RGB pixels from the 24 precommitted random
tissue-grid patches. No detections or model overlays enter the reviewer folder.
Private mapping stays OUTSIDE blind/. SCORE converts explicitly exhaustive,
alias-only reviewer points to the frozen M2 scorer without changing thresholds.
Not a biological accuracy certification, new nuclear inference, or KL720 run.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from radiation_edge_ai import perilipin2_tracka as core

PACK_SCHEMA = 'radedge-tracka-blind-qc-pack-v1'
ANNOTATION_SCHEMA = 'radedge-tracka-blind-points-v1'
SCORE_SCHEMA = 'radedge-tracka-blind-qc-score-v1'
VERSION = 'tracka-blind-review-v1'
REPO = Path(__file__).resolve().parents[2]
REVIEW_MARGIN_POLICY = 'original_frozen_inner_scoring_rectangle'
REQUIRED = ('x', 'y', 'width', 'height', 'margin', 'patch_id', 'image_uid',
            'role', 'selection')
ROLES = {'calibration', 'held_out_validation'}
ANNOTATOR_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Perilipin2 independent blinded patch review</title>
<style>
body{margin:0;background:#f3f5f8;color:#152638;font:15px system-ui,sans-serif}
main{max-width:960px;margin:28px auto;padding:20px;background:white;border-radius:10px}
button,select,input,textarea{font:inherit;padding:8px;border:1px solid #a6b4c3;border-radius:5px}
button{cursor:pointer;background:#eaf2f7}button:hover{background:#d4e7f2}
button:focus-visible,textarea:focus-visible{outline:3px solid #2587b5}
#picture{display:block;width:min(90vw,672px);height:auto;max-width:100%;margin:15px auto;
  border:1px solid #bbc9d2;image-rendering:auto;touch-action:manipulation}
.row{display:flex;gap:9px;flex-wrap:wrap;align-items:center;margin:10px 0}
.notice{padding:13px;background:#fff3cf;border-left:4px solid #a87600;line-height:1.6}
.ok{color:#006a4b;font-weight:bold}textarea{width:95%;min-height:80px}
small{color:#566675}h1{font-size:24px}.wide{width:100%}
</style></head><body><main>
<h1>Independent brown-object patch review</h1>
<div class="notice">
<strong>Detection predictions are hidden.</strong> Mark the center of EVERY independently
identifiable, discrete compact brown DAB-positive object within the yellow scoring rectangle,
including objects a detector could have missed. Do not automatically label rings, merged
structures, diffuse DAB or debris as compact objects. Describe ambiguity in Notes.
A patch with zero objects is a true reviewed negative ONLY after you deliberately mark
the exhaustive checkbox. This is provisional object review, not validated lipid-droplet biology.
Do not open the prediction/QC viewer during independent annotation.
</div>
<div class="row"><label>Reviewer code <input id="reviewer" autocomplete="off" placeholder="optional pseudonym"></label>
<label>Patch <select id="picker"></select></label><button id="previous">Previous</button>
<button id="next">Next</button></div>
<div class="row"><strong id="heading"></strong><span id="progress"></span></div>
<canvas id="picture" width="224" height="224" aria-label="Click original pixels to mark a point"></canvas>
<div class="row"><label><input type="checkbox" id="complete"> This patch is exhaustively reviewed (including zero-object negatives)</label>
<button id="clear">Clear points</button><span id="points"></span></div>
<label class="wide">Ambiguities / review notes<br><textarea id="notes" placeholder="Record unclear rings, clusters, staining, excluded artifacts"></textarea></label>
<div class="row"><button id="export">Export annotated review JSON</button>
<label>Import previous review <input id="import" type="file" accept=".json,application/json"></label></div>
<p class="notice">There is NO autosave. Export JSON before closing or reloading. Re-import it to
continue. All coordinates are crop-local and will be validated against the frozen scoring
rectangle before the study coordinator maps them to the original images. No predictions,
treatment labels, patch roles, or original sample IDs are contained in this reviewer folder.</p>
<p><small id="status"></small></p></main>
<script>
'use strict';
const PACK_DATA = __PACK_DATA__;
const P = PACK_DATA.patches, A = {}, cache = {};
for(const p of P) A[p.alias]={points:[],exhaustive_complete:false,note:''};
let selected=0;
const el=id=>document.getElementById(id), canvas=el('picture'),ctx=canvas.getContext('2d');
const picker=el('picker'); for(let i=0;i<P.length;i++){
  const opt=document.createElement('option');opt.value=String(i);
  opt.textContent='Patch '+String(i+1).padStart(2,'0');picker.appendChild(opt);
}
const current=()=>P[selected], entry=()=>A[current().alias];
function draw(){
 const p=current();let im=cache[p.alias];
 ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#f2f4f8';ctx.fillRect(0,0,canvas.width,canvas.height);
 if(!im){ im=new Image();cache[p.alias]=im;im.onload=()=>{if(current().alias===p.alias)draw()};
   im.onerror=()=>{el('status').textContent='Image missing: '+p.alias+'.png'};
   im.src='crops/'+encodeURIComponent(p.alias)+'.png'; }
 else if(im.complete&&im.naturalWidth>0){ctx.drawImage(im,0,0,p.width,p.height)}
 ctx.strokeStyle='#ffe000';ctx.lineWidth=1;ctx.setLineDash([4,2]);
 ctx.strokeRect(p.margin+.5,p.margin+.5,p.width-2*p.margin,p.height-2*p.margin);ctx.setLineDash([]);
 ctx.strokeStyle='#00f6ff';ctx.lineWidth=1.4;
 for(const q of entry().points){ctx.beginPath();ctx.moveTo(q[0]-3,q[1]);ctx.lineTo(q[0]+3,q[1]);ctx.moveTo(q[0],q[1]-3);ctx.lineTo(q[0],q[1]+3);ctx.stroke()}
 el('points').textContent=String(entry().points.length)+' reference points';
}
function render(){
 const p=current();canvas.width=p.width;canvas.height=p.height;picker.value=String(selected);
 el('heading').textContent='Anonymous patch '+p.alias;
 el('notes').value=entry().note;el('complete').checked=entry().exhaustive_complete;
 const completed=Object.values(A).filter(x=>x.exhaustive_complete===true).length;
 el('progress').textContent='Completed: '+completed+' / '+P.length;
 draw();
}
function change(n){entry().note=el('notes').value;selected=(n+P.length)%P.length;render()}
el('next').onclick=()=>change(selected+1);el('previous').onclick=()=>change(selected-1);
picker.onchange=()=>change(Number(picker.value));
el('complete').onchange=()=>{entry().exhaustive_complete=el('complete').checked;render()};
el('notes').oninput=()=>{entry().note=el('notes').value};
canvas.onclick=evt=>{
 const b=canvas.getBoundingClientRect(),p=current();
 const x=(evt.clientX-b.left)*canvas.width/b.width,y=(evt.clientY-b.top)*canvas.height/b.height;
 if(x<p.margin||y<p.margin||x>=p.width-p.margin||y>=p.height-p.margin)return;
 const pts=entry().points;
 const k=pts.findIndex(z=>Math.hypot(z[0]-x,z[1]-y)<4);
 if(k>=0)pts.splice(k,1);else pts.push([Math.round(x*100)/100,Math.round(y*100)/100]);
 entry().exhaustive_complete=false;el('complete').checked=false;render();
};
el('clear').onclick=()=>{
 if(!confirm('Clear ALL annotations for this patch?'))return;
 entry().points=[];entry().exhaustive_complete=false;render();
};
function download(content,name){
 const url=URL.createObjectURL(new Blob([content],{type:'application/json'}));
 const a=document.createElement('a');a.href=url;a.download=name;a.click();
 setTimeout(()=>URL.revokeObjectURL(url),10000);
}
el('export').onclick=()=>{
 entry().note=el('notes').value;
 const record={schema:PACK_DATA.annotation_schema,review_pack_id:PACK_DATA.review_pack_id,
   reviewer_code:el('reviewer').value.trim(),patches:A};
 download(JSON.stringify(record,null,2),'TrackA_blind_review_'+PACK_DATA.review_pack_id.slice(0,10)+'.json');
 el('status').textContent='Review JSON exported. Keep it alongside the private pack for scoring.';
};
el('import').onchange=async evt=>{
 try{
  const file=evt.target.files[0];if(!file)return;
  const x=JSON.parse(await file.text());
  if(x.schema!==PACK_DATA.annotation_schema||x.review_pack_id!==PACK_DATA.review_pack_id)
    throw Error('Review file belongs to a different pack');
  if(!x.patches||Object.keys(x.patches).some(id=>!(id in A)))throw Error('Unexpected patch ID');
  for(const p of P){
   const v=x.patches[p.alias];if(!v)continue;
   if(typeof v.exhaustive_complete!=='boolean'||!Array.isArray(v.points)||
      v.points.some(q=>!Array.isArray(q)||q.length!==2||
        !q.every(Number.isFinite)||q[0]<p.margin||q[1]<p.margin||
        q[0]>=p.width-p.margin||q[1]>=p.height-p.margin))
       throw Error('Invalid coordinates or review state');
  }
  for(const p of P)if(x.patches[p.alias])A[p.alias]=JSON.parse(JSON.stringify(x.patches[p.alias]));
  el('reviewer').value=x.reviewer_code||'';render();el('status').textContent='Review imported; export JSON after any edits';
 }catch(err){alert('Import refused: '+err.message)}
};
render();
</script></body></html>"""


def stamp():
    return datetime.now(timezone.utc).isoformat()


def package_audit():
    filename = REPO / 'scripts/windows/audit_perilipin2_tracka.py'
    spec = importlib.util.spec_from_file_location('_tracka_qc_original_audit', filename)
    if spec is None or spec.loader is None:
        raise ValueError('Frozen replay audit unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verified_parent(bundle_arg, run_arg):
    bundle, manifest = core.verify_bundle(bundle_arg)
    run = Path(run_arg).expanduser().resolve(strict=True)
    receipt = core.verify_result(run)
    if (receipt.get('bundle_sha256') != manifest['fingerprint_sha256']
            or receipt.get('source_sha256') != core.SOURCE_SHA256
            or receipt.get('configuration') != manifest['configuration']
            or receipt.get('kl720_inference') is not False
            or receipt.get('new_nuclear_segmentation') is not False):
        raise ValueError('Run/reference identity or processing scope mismatched')
    exact = core.compare(bundle, run, manifest)
    if exact != core.load_json(run / 'REFERENCE_COMPARISON.json'):
        raise ValueError('Saved exact comparison cannot be reproduced')
    all_arrays = len(exact['arrays']) == 14 and all(x.get('passed') is True for x in exact['arrays'].values())
    other_tables = all(exact['tables'].get(name, {}).get('passed') is True
                       for name in core.TABLE_KEYS if name != 'T02_brown_object_catalog.tsv')
    if not all_arrays or not other_tables:
        raise ValueError('The reference images/other tables do not pass portability invariants')
    if exact['passed']:
        mode = 'exact_reference_replay'
    else:
        audit = package_audit()
        summary, _ = audit.catalog_audit(
            core, bundle / 'expected/tables/T02_brown_object_catalog.tsv',
            run / 'tables/T02_brown_object_catalog.tsv', manifest['configuration'])
        if not (summary['columns_and_row_keys_exact'] and summary['status_exact']
                and summary['reason_exact'] and summary['only_finite_morphology_values_differ']
                and summary['morphology_threshold_decision_changes'] == 0):
            raise ValueError('Object identities/decisions differ: QC source cannot be accepted')
        mode = 'portability_decision_invariants_exact_mismatch_retained'
    return bundle, manifest, run, receipt, mode


def original_patches(run, manifest):
    path = run / 'tables/T04_random_patch_review_plan.tsv'
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError('Malformed frozen patch plan')
        if not set(REQUIRED).issubset(reader.fieldnames):
            raise ValueError('Missing patch-plan columns')
        rows = list(reader)
    fields = {f['image_uid']: f for f in manifest['fields']}
    if len(rows) != 24 or len({p['patch_id'] for p in rows}) != 24:
        raise ValueError('This reference validation is fixed to exactly 24 random tissue-grid patches')
    result, per_field = [], {key: {role: 0 for role in ROLES} for key in fields}
    for p in rows:
        if p['image_uid'] not in fields or p['role'] not in ROLES:
            raise ValueError('Unknown source image or calibration/heldout role')
        if p['selection'] != 'random_nonoverlapping_tissue_grid_NOT_candidate_selected':
            raise ValueError('Patch plan is not frozen random tissue-grid selection')
        numeric = {}
        for name in ('x', 'y', 'width', 'height', 'margin'):
            raw = p[name]
            if not raw.isdigit():
                raise ValueError('Invalid integer patch coordinate')
            numeric[name] = int(raw)
        h, w = int(fields[p['image_uid']]['original_record']['height']), int(fields[p['image_uid']]['original_record']['width'])
        x, y, pw, ph, m = (numeric[k] for k in ('x','y','width','height','margin'))
        if not (pw == ph == 224 and m == 8 and x>=0 and y>=0 and x+pw<=w and y+ph<=h):
            raise ValueError('Frozen patch geometry or margins changed')
        if not (p['patch_id'].startswith(p['image_uid'] + '_P')):
            raise ValueError('Frozen patch identifier/image mismatch')
        per_field[p['image_uid']][p['role']] += 1
        result.append({**p, **numeric})
    if any(count != 6 for d in per_field.values() for count in d.values()):
        raise ValueError('Expected six calibration and six heldout patches per field')
    return result


def pack_fingerprint(bundle_hash, source_dataset, patches):
    identity = {'version': VERSION, 'bundle': bundle_hash, 'dataset': source_dataset,
                'patches': [dict((k, p[k]) for k in (
                    'patch_id','image_uid','role','x','y','width','height','margin'))
                    for p in sorted(patches, key=lambda p: p['patch_id'])]}
    return core.fingerprint(identity)


def crop_original_pixels(bundle, field, p, dest):
    from PIL import Image
    source = core.contained(bundle, field['assets']['rgb'])
    y, x, w, h = p['y'], p['x'], p['width'], p['height']
    # Same image-mode/frame/EXIF policy as the frozen M2 load_rgb()
    # implementation. Cropping RGB with Pillow retains original pixel values.
    with Image.open(source) as image:
        if getattr(image, 'n_frames', 1) != 1 or image.mode not in ('RGB', 'RGBA'):
            raise ValueError('Expected a single RGB brightfield image')
        if image.getexif().get(274, 1) != 1:
            raise ValueError('EXIF rotation is not silently applied')
        if image.size != (int(field['original_record']['width']),
                          int(field['original_record']['height'])):
            raise ValueError('Input image dimensions changed')
        crop = image.convert('RGB').crop((x, y, x+w, y+h))
        if crop.size != (w, h):
            raise ValueError('Incomplete crop')
        crop.save(dest, format='PNG')
    with Image.open(dest) as check:
        if check.mode != 'RGB' or check.size != (w, h):
            raise ValueError('Crop format/size changed')


def generate_blind_html(pack_id, mapping):
    fields = [{'alias': alias, 'width': item['width'], 'height': item['height'],
               'margin': item['margin']} for alias, item in mapping.items()]
    data = {'review_pack_id': pack_id, 'annotation_schema': ANNOTATION_SCHEMA, 'patches': fields}
    blob = json.dumps(data, ensure_ascii=True, separators=(',', ':')).replace('<', '\\u003c')
    return ANNOTATOR_HTML.replace('__PACK_DATA__', blob)


def pack_file_manifest(root):
    records={}
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name != 'QC_PACK_MANIFEST.json':
            rel=path.relative_to(root).as_posix()
            records[rel]=core.file_record(path)
    return records


def prepare(args):
    bundle, manifest, run, receipt, mode = verified_parent(args.bundle, args.run)
    patches = original_patches(run, manifest)
    pack_id=pack_fingerprint(manifest['fingerprint_sha256'], receipt['dataset_id'], patches)
    out,stage=core.new_output(args.out, (bundle,run))
    stage.mkdir(parents=True,exist_ok=False)
    try:
        randomized=sorted(patches,key=lambda p:p['patch_id'])
        rng=random.Random(int(pack_id[:16],16))
        rng.shuffle(randomized)
        fields={f['image_uid']:f for f in manifest['fields']}
        mappings={}
        crops=stage / 'blind/crops'
        crops.mkdir(parents=True)
        (stage / 'private').mkdir()
        for index,p in enumerate(randomized,1):
            alias='R{:03d}'.format(index)
            field=fields[p['image_uid']]
            dest=crops / (alias+'.png')
            crop_original_pixels(bundle,field,p,dest)
            mappings[alias]={**p,'source_sha256':field['original_record']['source_sha256'],
                             'dose_Gy':field['original_record']['dose_Gy'],
                             'sample_token':field['original_record']['sample_token']}
        (stage / 'blind/annotate.html').write_text(
            generate_blind_html(pack_id,mappings),encoding='utf-8')
        (stage / 'blind/README.txt').write_text(
            'INDEPENDENT REVIEW PACKET. Open annotate.html in your browser. '
            'Review original RGB crops without looking at predictions, treatment metadata '
            'or the private mapping. Mark every compact brown object inside the yellow '
            'rectangle; explicitly mark exhaustive even for true empty patches. '
            'Export review JSON before closing; there is no autosave. '
            'Do not use non-exhaustive patches to estimate recall.\n',encoding='utf-8')
        private={'schema':PACK_SCHEMA,'review_pack_id':pack_id,
                 'source_bundle_fingerprint':manifest['fingerprint_sha256'],
                 'source_dataset_id':receipt['dataset_id'],
                 'source_receipt_sha256':core.digest(run/'TRACKA_EXECUTION.json'),
                 'portability_mode':mode,'n_patches':len(mappings),'patches':mappings}
        core.write_json(stage/'private/REVIEW_MAP.json',private)
        files=pack_file_manifest(stage)
        record={'schema':PACK_SCHEMA,'review_pack_id':pack_id,'status':'BLINDED_QC_PACK_READY',
                'source_bundle_fingerprint':manifest['fingerprint_sha256'],
                'source_dataset_id':receipt['dataset_id'],
                'source_receipt_sha256':private['source_receipt_sha256'],
                'portability_mode':mode,'n_patches':len(mappings),'files':files,
                'no_model_predictions_in_blind_folder':True,
                'no_new_inference':True,'biological_validation':False}
        record['fingerprint_sha256']=core.fingerprint(record)
        core.write_json(stage/'QC_PACK_MANIFEST.json',record)
        core.verify_bundle(bundle)
        core.verify_result(run)
        stage.rename(out)
    except Exception as exc:
        if stage.exists() and not (stage/'QC_PACK_FAILED.json').exists():
            core.write_json(stage/'QC_PACK_FAILED.json',{'error':str(exc)})
        raise
    print('TRACKA_BLIND_QC_PACK_READY:',out)
    print('Reviewer-only folder:',out/'blind')
    print('Private mapping (do not distribute to reviewer):',out/'private')
    print('Open:',out/'blind/annotate.html')
    print('Original randomized scoring roles remain unchanged; annotations NOT yet collected.')
    return 0


def verify_pack(path):
    root=Path(path).expanduser().resolve(strict=True)
    manifest=core.load_json(root/'QC_PACK_MANIFEST.json')
    h=manifest.get('fingerprint_sha256')
    if manifest.get('schema') != PACK_SCHEMA or core.fingerprint({
        k:v for k,v in manifest.items() if k!='fingerprint_sha256'}) !=h:
        raise ValueError('QC pack identity changed')
    recorded=manifest.get('files',{})
    if not isinstance(recorded,dict) or len(recorded)!=27:
        raise ValueError('QC pack must contain all 24 crops, HTML, README, private map')
    if set(recorded)!=set(pack_file_manifest(root)):
        raise ValueError('QC pack file list changed')
    for rel,v in recorded.items():
        if core.file_record(core.contained(root,rel))!=v:
            raise ValueError('QC pack asset changed: '+rel)
    private=core.load_json(root/'private/REVIEW_MAP.json')
    if (private.get('schema') != PACK_SCHEMA or private.get('review_pack_id')!=manifest['review_pack_id']
        or private.get('source_receipt_sha256')!=manifest['source_receipt_sha256']
        or len(private.get('patches',{}))!=24):
        raise ValueError('Private review mapping is invalid')
    if sorted(private['patches'])!=['R{:03d}'.format(i) for i in range(1,25)]:
        raise ValueError('Review alias list changed')
    if len({x['patch_id'] for x in private['patches'].values()})!=24:
        raise ValueError('Original patch IDs are not unique')
    html=(root/'blind/annotate.html').read_text(encoding='utf-8')
    # Prevent accidental leakage of source identifiers or treatment metadata.
    for item in private['patches'].values():
        for prohibited in (item['patch_id'],item['image_uid']):
            if prohibited in html:
                raise ValueError('Unblinding data leaked into reviewer HTML')
    for label in ('dose_Gy','sample_token','held_out_validation','calibration'):
        if label in html:
            raise ValueError('Source/study metadata leaked into reviewer HTML')
    return root,manifest,private


def validate_annotation(pack,review):
    if review.get('schema')!=ANNOTATION_SCHEMA or review.get('review_pack_id')!=pack['review_pack_id']:
        raise ValueError('Annotation belongs to a different review pack')
    patches=review.get('patches')
    if not isinstance(patches,dict) or set(patches)-set(pack['patches']):
        raise ValueError('Unexpected reviewer patch alias')
    reviewer=review.get('reviewer_code','')
    if not isinstance(reviewer,str) or len(reviewer)>128:
        raise ValueError('Invalid reviewer code')
    normalized={}
    complete=0
    for alias,p in pack['patches'].items():
        data=patches.get(alias, {'points':[],'exhaustive_complete':False,'note':''})
        if not isinstance(data,dict) or type(data.get('exhaustive_complete')) is not bool:
            raise ValueError('Exhaustive status must be an explicit Boolean for '+alias)
        notes=data.get('note','')
        if not isinstance(notes,str) or len(notes)>10000:
            raise ValueError('Invalid review note')
        pts=data.get('points')
        if not isinstance(pts,list) or len(pts)>20000:
            raise ValueError('Invalid patch points')
        unique=set()
        abs_points=[]
        for q in pts:
            if (not isinstance(q,list) or len(q)!=2 or
                any(type(v) not in (float,int) or not math.isfinite(v) for v in q)):
                raise ValueError('Reference points must be finite coordinate pairs')
            x,y=q
            if not (p['margin']<=x<p['width']-p['margin']
                    and p['margin']<=y<p['height']-p['margin']):
                raise ValueError('Reference point is outside the frozen inner scoring region')
            if tuple(q) in unique:
                raise ValueError('Duplicate point in patch '+alias)
            unique.add(tuple(q))
            abs_points.append([x+p['x'],y+p['y']])
        if data['exhaustive_complete']:
            complete+=1
        normalized[p['patch_id']]={'points':abs_points,
                                   'exhaustive_complete':data['exhaustive_complete'],
                                   'note':notes}
    if not complete:
        raise ValueError('No explicitly exhaustive patch. No recall can be estimated')
    return normalized,complete,reviewer


def score(args):
    bundle,manifest,run,receipt,mode=verified_parent(args.bundle,args.run)
    root,pack_manifest,private=verify_pack(args.pack)
    if (pack_manifest['source_bundle_fingerprint']!=manifest['fingerprint_sha256']
        or pack_manifest['source_dataset_id']!=receipt['dataset_id']
        or pack_manifest['source_receipt_sha256']!=core.digest(run/'TRACKA_EXECUTION.json')
        or pack_manifest['portability_mode']!=mode):
        raise ValueError('QC review does not belong to this validated source run')
    review_path=Path(args.review).expanduser().resolve(strict=True)
    review=core.load_json(review_path)
    converted,count,reviewer=validate_annotation(private,review)
    output,stage=core.new_output(args.out,(root,bundle,run))
    if output==review_path or output in review_path.parents:
        raise ValueError('Output must not contain review annotations')
    stage.mkdir(parents=True,exist_ok=False)
    try:
        freeze={'dataset_id':receipt['dataset_id'],'objects':{},
                'patches':converted}
        core.write_json(stage/'SCORER_INPUT_FROZEN_M2.json',freeze)
        from types import SimpleNamespace
        core.score(SimpleNamespace(bundle=bundle,run=run,
                                   review=stage/'SCORER_INPUT_FROZEN_M2.json',
                                   radius=args.radius,out=stage/'frozen_evaluation'))
        core.verify_bundle(bundle)
        core.verify_result(run)
        verify_pack(root)
        roles={r:0 for r in ROLES}
        fields={}
        for p in private['patches'].values():
            if converted[p['patch_id']]['exhaustive_complete']:
                roles[p['role']]+=1
                fields.setdefault(p['image_uid'],{r:0 for r in ROLES})[p['role']]+=1
        full=count==24
        outcome={'schema':SCORE_SCHEMA,
                 'status':'REVIEW_COMPLETE_SCORING_NOT_BIOLOGICAL_VALIDATION' if full
                          else 'REVIEW_PARTIAL_SCORING_NOT_BIOLOGICAL_VALIDATION',
                 'review_pack_id':private['review_pack_id'],'review_sha256':core.digest(review_path),
                 'run_receipt_sha256':core.digest(run/'TRACKA_EXECUTION.json'),
                 'dataset_id':receipt['dataset_id'],'complete_patches':count,
                 'total_patches':24,'completed_by_role':roles,'completed_by_field_role':fields,
                 'reviewer_code':reviewer,'match_radius_source_px':args.radius,
                 'original_m2_scorer':True,'thresholds_modified':False,
                 'kl720_inference':False,'biological_validation':False,
                 'interpretation':'Selected random tissue patches only. Do not pool multiple patches as independent animals. Calibration and held-out roles remain distinct.'}
        core.write_json(stage/'QC_REVIEW_SCORING.json',outcome)
        stage.rename(output)
    except Exception as exc:
        if stage.exists() and not (stage/'QC_SCORE_FAILED.json').exists():
            core.write_json(stage/'QC_SCORE_FAILED.json',{'error':str(exc)})
        raise
    print(outcome['status'])
    print('Exhaustive patches:',count,'/ 24; by role:',roles)
    print('Frozen role/rule precision and recall:',output/'frozen_evaluation/T02_role_rule_evaluation.tsv')
    print('Review/scorer provenance:',output/'QC_REVIEW_SCORING.json')
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    for name in ('prepare','score'):
        p=sub.add_parser(name)
        for field in ('bundle','run','out'):
            p.add_argument('--'+field,required=True,type=Path)
        if name=='score':
            p.add_argument('--pack',required=True,type=Path)
            p.add_argument('--review',required=True,type=Path)
            p.add_argument('--radius',required=True,type=float,
                           help='Prespecified one-to-one match radius in source pixels')
    p=sub.add_parser('verify-pack')
    p.add_argument('--pack',required=True,type=Path)
    args=parser.parse_args(argv)
    try:
        if args.action=='prepare':
            return prepare(args)
        if args.action=='score':
            if not math.isfinite(args.radius) or args.radius<=0:
                raise ValueError('Match radius must be positive and finite')
            return score(args)
        root,manifest,_=verify_pack(args.pack)
        print('TRACKA_BLIND_QC_PACK_INTEGRITY_PASS:',root,manifest['review_pack_id'])
        return 0
    except Exception as exc:
        print('TRACKA_QC_STOP: {}: {}'.format(type(exc).__name__,exc),file=sys.stderr,flush=True)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
