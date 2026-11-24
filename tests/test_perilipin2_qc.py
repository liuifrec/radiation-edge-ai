"""Independent Track A QC-pack tests: no microscopy images, devices or model weights.

Synthetic crop writes and frozen-scoring calls are mocked; these tests verify
identity, selection, exclusion margins, reviewer blinding, and provenance.
"""
from __future__ import annotations

import csv
import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from radiation_edge_ai import perilipin2_qc as qc
from radiation_edge_ai import perilipin2_tracka as core


class QCReviewTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base=Path(temp.name).resolve()
        self.bundle=self.base/'bundle'
        self.run=self.base/'run'
        self.bundle.mkdir()
        self.run.mkdir()
        self.pack=self.base/'qc_packet'
        self.score_output=self.base/'scored'
        self.fields={}
        for i in range(2):
            uid='synthetic_field_'+str(i)
            self.fields[uid]={
                'image_uid':uid, 'assets':{'rgb':'inputs/'+uid+'/image.png'},
                'original_record':{
                    'height':1223,'width':2174,'source_sha256':('a' if i==0 else 'b')*64,
                    'dose_Gy':('0' if i==0 else '3.8'),
                    'sample_token':str(13+i*14)}}
        self.manifest={'fingerprint_sha256':'m'*64,
                       'fields':list(self.fields.values())}
        self.receipt={'dataset_id':'d'*64}
        (self.run/'TRACKA_EXECUTION.json').write_text('{"sample":"fixture"}')
        (self.run/'tables').mkdir()
        self.patches=[]
        for field in self.fields:
            for n in range(12):
                p={'patch_id':field+'_P{:02d}'.format(n+1),'image_uid':field,
                   'x':str(224*(n%8)),'y':str(224*(n//8)),
                   'width':'224','height':'224','margin':'8',
                   'role':'calibration' if n%2==0 else 'held_out_validation',
                   'selection':'random_nonoverlapping_tissue_grid_NOT_candidate_selected',
                   'tissue_fraction':'0.95'}
                self.patches.append(p)
        with (self.run/'tables/T04_random_patch_review_plan.tsv').open(
                'w',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(self.patches[0]),delimiter='\t')
            writer.writeheader()
            writer.writerows(self.patches)
        self.parents=mock.patch.object(qc,'verified_parent',
            return_value=(self.bundle,self.manifest,self.run,self.receipt,
                          'portability_decision_invariants_exact_mismatch_retained'))
        self.parents.start()
        self.addCleanup(self.parents.stop)
        v1=mock.patch.object(qc.core,'verify_bundle',
            return_value=(self.bundle,self.manifest))
        v2=mock.patch.object(qc.core,'verify_result',return_value=self.receipt)
        v1.start()
        v2.start()
        self.addCleanup(v1.stop)
        self.addCleanup(v2.stop)

    def make_pack(self):
        def fake_crop(bundle,field,patch,destination):
            destination.write_bytes(('synthetic '+patch['patch_id']).encode())
        with mock.patch.object(qc,'crop_original_pixels',side_effect=fake_crop):
            rc=qc.prepare(SimpleNamespace(bundle=self.bundle,run=self.run,out=self.pack))
        self.assertEqual(rc,0)
        return qc.verify_pack(self.pack)

    def test_frozen_plan_roles_are_preserved_and_not_candidate_selected(self):
        p=qc.original_patches(self.run,self.manifest)
        self.assertEqual(len(p),24)
        self.assertEqual(sum(x['role']=='calibration' for x in p),12)
        self.assertEqual(sum(x['role']=='held_out_validation' for x in p),12)
        self.assertEqual(sum(x['width'] for x in p),24*224)
        p2=qc.original_patches(self.run,self.manifest)
        self.assertEqual(p,p2)

    def test_packet_is_blind_and_integrity_checked(self):
        root, manifest, private=self.make_pack()
        self.assertEqual(manifest['status'],'BLINDED_QC_PACK_READY')
        self.assertEqual(len(private['patches']),24)
        self.assertEqual(len(manifest['files']),27)
        html=(root/'blind/annotate.html').read_text()
        self.assertIn('exhaustively reviewed',html)
        self.assertIn('not validated lipid-droplet',html)
        self.assertIn('crops/',html)
        self.assertIn('review_pack_id',html)
        for p in self.patches:
            self.assertNotIn(p['patch_id'],html)
            self.assertNotIn(p['image_uid'],html)
        self.assertNotIn('held_out_validation',html)
        self.assertNotIn('dose_Gy',html)
        self.assertNotIn('calibration',html)
        self.assertTrue((root/'blind/crops/R001.png').exists())
        self.assertNotIn('image_uid',(root/'blind/README.txt').read_text())

    def test_packet_alias_order_is_reproducible_without_leaking_field(self):
        p=qc.original_patches(self.run,self.manifest)
        id1=qc.pack_fingerprint(self.manifest['fingerprint_sha256'],
                                self.receipt['dataset_id'],p)
        id2=qc.pack_fingerprint(self.manifest['fingerprint_sha256'],
                                self.receipt['dataset_id'],list(reversed(p)))
        self.assertEqual(id1,id2)
        self.assertEqual(len(id1),64)

    def test_tampered_crop_fails_integrity(self):
        self.make_pack()
        path=self.pack/'blind/crops/R002.png'
        path.write_bytes(path.read_bytes()+b'tampered')
        with self.assertRaisesRegex(ValueError,'asset changed'):
            qc.verify_pack(self.pack)

    def test_missing_or_unexpected_crop_fails(self):
        self.make_pack()
        (self.pack/'blind/crops/R004.png').unlink()
        with self.assertRaises(ValueError):
            qc.verify_pack(self.pack)

    def test_role_and_size_mutations_rejected_before_cropping(self):
        p=self.run/'tables/T04_random_patch_review_plan.tsv'
        original=p.read_text()
        try:
            p.write_text(original.replace('held_out_validation','custom_review'))
            with self.assertRaisesRegex(ValueError,'role'):
                qc.original_patches(self.run,self.manifest)
            p.write_text(original.replace('\t224\t224\t8\t','\t224\t225\t8\t',1))
            with self.assertRaises(ValueError):
                qc.original_patches(self.run,self.manifest)
        finally:
            p.write_text(original)

    def test_explicit_negative_is_allowed_but_implicit_is_not_complete(self):
        _,_,private=self.make_pack()
        key=next(iter(private['patches']))
        form={'schema':qc.ANNOTATION_SCHEMA,'review_pack_id':private['review_pack_id'],
              'patches':{key:{'points':[],'exhaustive_complete':True,'note':'No compact brown objects'}}}
        rows,count,reviewer=qc.validate_annotation(private,form)
        self.assertEqual(count,1)
        self.assertFalse(any(v['points'] for v in rows.values()))
        self.assertEqual(len(rows),24)
        self.assertEqual(reviewer,'')
        form['patches'][key]['exhaustive_complete']=False
        with self.assertRaisesRegex(ValueError,'No explicitly exhaustive'):
            qc.validate_annotation(private,form)

    def test_coordinate_mapping_and_inner_margin(self):
        _,_,private=self.make_pack()
        alias=next(iter(private['patches']))
        p=private['patches'][alias]
        form={'schema':qc.ANNOTATION_SCHEMA,'review_pack_id':private['review_pack_id'],
              'reviewer_code':'reviewer-1',
              'patches':{alias:{'points':[[8.25,25.75]],
                                'exhaustive_complete':True,'note':'reviewed'}}}
        mapped,n,code=qc.validate_annotation(private,form)
        point=mapped[p['patch_id']]['points'][0]
        self.assertEqual(point,[p['x']+8.25,p['y']+25.75])
        self.assertEqual(code,'reviewer-1')
        for point in ([7.999,10],[224-8,10],[10,-1],[float('nan'),20]):
            form['patches'][alias]['points']=[point]
            with self.subTest(point=point),self.assertRaises(ValueError):
                qc.validate_annotation(private,form)

    def test_duplicate_and_wrong_dataset_rejected(self):
        _,_,private=self.make_pack()
        alias=next(iter(private['patches']))
        form={'schema':qc.ANNOTATION_SCHEMA,'review_pack_id':private['review_pack_id'],
              'patches':{alias:{'points':[[22,22],[22,22]],
                                'exhaustive_complete':True,'note':''}}}
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            qc.validate_annotation(private,form)
        form['patches'][alias]['points']=[[22,22]]
        form['review_pack_id']='unrelated'
        with self.assertRaisesRegex(ValueError,'different review pack'):
            qc.validate_annotation(private,form)

    def test_annotation_keeps_original_m2_scorer_and_roles(self):
        _,_,private=self.make_pack()
        alias=next(iter(private['patches']))
        form={'schema':qc.ANNOTATION_SCHEMA,'review_pack_id':private['review_pack_id'],
              'reviewer_code':'anon',
              'patches':{alias:{'points':[[12,12]],'exhaustive_complete':True,'note':''}}}
        review=self.base/'review.json'
        core.write_json(review,form)
        calls=[]
        def original_scorer(args):
            calls.append(args)
            args.out.mkdir(parents=True,exist_ok=False)
            (args.out/'T02_role_rule_evaluation.tsv').write_text('role\trule\n')
            scorer_review=core.load_json(args.review)
            self.assertEqual(scorer_review['dataset_id'],self.receipt['dataset_id'])
            self.assertEqual(len(scorer_review['patches']),24)
            self.assertEqual(sum(v['exhaustive_complete']
                                 for v in scorer_review['patches'].values()),1)
        with mock.patch.object(qc.core,'score',side_effect=original_scorer):
            self.assertEqual(qc.score(SimpleNamespace(
                bundle=self.bundle,run=self.run,pack=self.pack,review=review,
                out=self.score_output,radius=6.0)),0)
        self.assertEqual(len(calls),1)
        outcome=core.load_json(self.score_output/'QC_REVIEW_SCORING.json')
        self.assertFalse(outcome['biological_validation'])
        self.assertEqual(outcome['complete_patches'],1)
        self.assertEqual(outcome['status'],'REVIEW_PARTIAL_SCORING_NOT_BIOLOGICAL_VALIDATION')
        self.assertEqual(sum(outcome['completed_by_role'].values()),1)

    def test_zero_annotations_fail_before_scoring_or_new_output(self):
        _,_,private=self.make_pack()
        form={'schema':qc.ANNOTATION_SCHEMA,'review_pack_id':private['review_pack_id'],
              'patches':{}}
        review=self.base/'empty.json'
        core.write_json(review,form)
        with mock.patch.object(qc.core,'score') as call:
            with self.assertRaisesRegex(ValueError,'No explicitly exhaustive'):
                qc.score(SimpleNamespace(bundle=self.bundle,run=self.run,
                       pack=self.pack,review=review,out=self.score_output,radius=6.0))
        call.assert_not_called()
        self.assertFalse(self.score_output.exists())

    @unittest.skipUnless(importlib.util.find_spec('PIL'),
                         'Pillow unavailable for synthetic pixel-level crop test')
    def test_original_rgb_crop_has_identical_pixel_values(self):
        import numpy as np
        from PIL import Image
        source=self.bundle/'inputs/synthetic_field_0/image.png'
        source.parent.mkdir(parents=True)
        y,x=np.indices((260,280))
        rgb=np.stack([(x%255).astype(np.uint8),
                      (y%255).astype(np.uint8),
                      ((x+y)%255).astype(np.uint8)],axis=-1)
        Image.fromarray(rgb).save(source)
        field={'assets':{'rgb':'inputs/synthetic_field_0/image.png'},
               'original_record':{'height':260,'width':280}}
        dest=self.base/'cropped.png'
        qc.crop_original_pixels(self.bundle,field,
                                {'x':13,'y':17,'width':224,'height':224},dest)
        with Image.open(dest) as image:
            result=np.asarray(image)
        self.assertTrue(np.array_equal(result,rgb[17:241,13:237]))
        self.assertEqual(result.shape,(224,224,3))

    @unittest.skipUnless(shutil.which('node'),
                         'Node.js unavailable for static browser-JS syntax test')
    def test_generated_offline_review_javascript_parses(self):
        html=qc.generate_blind_html('a'*64,
             {'R001':{'width':224,'height':224,'margin':8}})
        source=html.split('<script>',1)[1].split('</script>',1)[0]
        code=self.base/'review-browser.js'
        code.write_text(source,encoding='utf-8')
        result=subprocess.run(['node','--check',str(code)],
                              capture_output=True,text=True,check=False)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_review_pack_never_replaces_existing_output(self):
        self.pack.mkdir()
        with mock.patch.object(qc,'crop_original_pixels') as call:
            with self.assertRaises(FileExistsError):
                qc.prepare(SimpleNamespace(bundle=self.bundle,run=self.run,out=self.pack))
        call.assert_not_called()


if __name__=='__main__':
    unittest.main()
