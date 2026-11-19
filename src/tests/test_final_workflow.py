"""Numerical regressions and public-command contracts; no GPU required."""
import ast
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from signed_tf_targets import edit_signed_rows,make_programs,_float32
from baseline_contract import compare_cells,compare_arrays
from cluster_regulons import features,cluster


class SignedEditorTests(unittest.TestCase):
    def setUp(self):
        self.plan=dict(signed_edit_version='float32_v2',version=2,tf='TF',tf_token=30,tf_edit=False,
            tf_auxiliary_mask=1,tf_overexpress_value=5.,perturb_type='overexpress',response_sign=1,
            value_clip=5.,max_length=8,targets=[dict(gene='UP',token=11,bulk_log2FC=1.,mean=math.log(11),sd=1.),
                dict(gene='DOWN',token=22,bulk_log2FC=-1.,mean=math.log(11),sd=1.)])
        self.row=dict(input_ids=[22,11,99],expression_values=[0.,0.,0.],auxiliary_gene_mask=[1,0,1],length=3)

    def test_zero_strength_zero_fold_change_absent_targets_are_exact_identities(self):
        self.assertEqual(edit_signed_rows([self.row],self.plan,0)[0][0],self.row)
        zero=dict(self.plan,targets=[dict(t,bulk_log2FC=0.) for t in self.plan['targets']])
        self.assertEqual(edit_signed_rows([self.row],zero,1)[0][0],self.row)
        absent=dict(self.plan,targets=[dict(self.plan['targets'][0],token=700)])
        self.assertEqual(edit_signed_rows([self.row],absent,1)[0][0],self.row)

    def test_tiny_float32_changes_preserve_clipped_ties_and_masks(self):
        row=dict(self.row,expression_values=[5.,5.,5.])
        self.assertEqual(edit_signed_rows([row],self.plan,1e-12)[0][0],row)

    def test_overexpression_and_inverse_loss_have_opposite_target_directions(self):
        for sign in [1,-1]:
            plan=dict(self.plan,response_sign=sign,perturb_type='overexpress' if sign==1 else 'delete')
            result=edit_signed_rows([self.row],plan,1)[0][0]
            values=dict(zip(result['input_ids'],result['expression_values']))
            self.assertGreater(sign*values[11],0);self.assertLess(sign*values[22],0)
            self.assertEqual(dict(zip(result['input_ids'],result['auxiliary_gene_mask'])),{22:1,11:0,99:1})

    def test_tf_only_controls_insert_or_delete_without_moving_aligned_masks(self):
        plan=dict(self.plan,tf_edit=True,targets=[],max_length=2)
        result=edit_signed_rows([self.row],plan,1)[0][0]
        self.assertEqual(result['input_ids'][0],30);self.assertEqual(result['auxiliary_gene_mask'][0],1)
        self.assertEqual(len(result['input_ids']),2)
        loss=dict(plan,perturb_type='delete')
        self.assertNotIn(30,edit_signed_rows([result],loss,1)[0][0]['input_ids'])

    def test_five_programs_and_legacy_modes_rejected(self):
        self.assertEqual(len(make_programs(self.plan,[.25,.5,1])),5)
        for plan in [dict(self.plan,signed_edit_version='legacy_float64'),dict(self.plan,target_edit='binary_tf_loss')]:
            with self.assertRaises(ValueError):edit_signed_rows([self.row],plan,1)


class BaselineTests(unittest.TestCase):
    def test_cell_order_and_identity_are_required(self):
        compare_cells(['a','b'],['a','b'])
        for ids in [['b','a'],['a','a']]:
            with self.assertRaises(ValueError):compare_cells(ids,['a','b'])

    def test_nonfinite_or_different_predictions_rejected(self):
        compare_arrays([1,2],[1,2],'baseline')
        for values in [[1,3],[1,float('nan')],[1]]:
            with self.assertRaises(ValueError):compare_arrays(values,[1,2],'baseline')


class ClusteringTests(unittest.TestCase):
    def test_common_scale_and_offset_are_preserved(self):
        values=np.array([[1.,2.,3.],[3.,4.,5.],[-1.,-2.,-3.]])
        actual,scale=features(values)
        self.assertAlmostEqual(scale,np.median(np.sqrt(np.mean(values**2,axis=1))))
        np.testing.assert_allclose(actual,np.arcsinh(values/scale))
        self.assertFalse(np.allclose(actual[0],actual[1]))
        np.testing.assert_allclose(actual[0],-actual[2])

    def test_twelve_clusters_and_missing_values(self):
        x=np.random.default_rng(42).normal(size=(30,8))
        _,_,_,labels,order=cluster(x,12)
        self.assertEqual(set(labels),set(range(1,13)))
        self.assertEqual(len(set(order)),30)
        for x in [np.zeros((4,3)),np.array([[1,np.nan],[2,3]])]:
            with self.assertRaises(ValueError):features(x)


class CommandTests(unittest.TestCase):
    def test_default_sweep_is_complete(self):
        import training
        self.assertEqual(len(training.conditions()),22)
        self.assertEqual(len(training.conditions())*len(training.EPOCHS),440)
        self.assertEqual(sum(x['group']=='delight_fm' for x in training.conditions()),6)

    def test_commands_only_use_supported_engine_options(self):
        import screens
        src=Path(__file__).resolve().parents[1]
        def options(filename):
            tree=ast.parse((src/filename).read_text())
            return {arg.value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                and n.func.attr=='add_argument' for arg in n.args if isinstance(arg,ast.Constant) and isinstance(arg.value,str) and arg.value.startswith('--')}
        tasks=[dict(group='delight_fm',kind='regulons',tfs=['ERG'],table='collectri.tsv',output='out'),
            dict(group='delight_fm',kind='markers',genes=['EPCAM'],output='out'),
            dict(group='immune',kind='regulons',tfs=['IRF1'],table='scenic.tsv',output='out')]
        with patch('screens.model_path',return_value=Path('/saved/model')):
            for task in tasks:
                cmd=screens.task_command(task)
                supported=options('perturb_lineage.py' if task['group']=='delight_fm' else 'perturb_age.py')
                flags={str(a) for a in cmd if str(a).startswith('--')}
                self.assertTrue(flags<=supported,flags-supported)
                self.assertIn('--reference-baseline',cmd)
                if task['group']!='delight_fm':self.assertIn('--skip-training',cmd)


if __name__=='__main__':unittest.main()
