"""Verify release program identity, coverage and donor-level effect semantics."""
import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from notebook_workflows import gene_program,regulon_programs,summarize_age,summarize_classifier

class NotebookWorkflowTests(unittest.TestCase):
    def test_aliases_and_signed_regulons_keep_distinct_programs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);stats=root/'stats.csv';mapping=root/'mapping.tsv';table=root/'regulons.tsv'
            pd.DataFrame({'gene':['TF','A','B'],'ensembl_id':['ENSG1','ENSG2','ENSG3'],'token':[1,2,3]}).to_csv(stats,index=False)
            mapping.write_text('gene_symbol\tensembl_id\nALIAS\tENSG2\n')
            self.assertEqual(gene_program(['ALIAS','A','missing'],stats,mapping)['tokens'],[2])
            pd.DataFrame({'eRegulon_name':['TF_direct_+/+','TF_direct_+/-'],'tf':['TF','TF'],'target':['A','B']}).to_csv(table,sep='\t',index=False)
            programs=regulon_programs(table,stats,mapping,source='scenic',include_tf=True)
            self.assertEqual(programs['TF_direct_+/+']['tokens'],[1,2])
            self.assertEqual(programs['TF_direct_+/-']['tokens'],[1,3])
            with self.assertRaises(ValueError):regulon_programs(table,stats,names=['UNKNOWN'])

    def test_age_ranking_preserves_opposite_sign_donors_and_equal_weight(self):
        metadata=pd.DataFrame({'cell_id':['a','b','c'],'donor_id':['D1','D1','D2'],'true_log2_PCW':[4,4,8]})
        result={'prediction_delta':np.array([2.,2.,-2.]),'embedding_l2':np.ones(3)}
        _,donors,summary=summarize_age(result,metadata,np.ones(3,dtype=bool))
        self.assertEqual(summary['mean_delta_log2_PCW'],0)
        self.assertEqual(summary['mean_absolute_donor_delta'],2)
        self.assertEqual(donors.n_cells.tolist(),[2,1])

    def test_annotated_class_effect_uses_each_cells_actual_class(self):
        metadata=pd.DataFrame({'cell_id':['a','b'],'donor_id':['D','D'],'cell_type':['A','B']})
        result={'prediction_delta':np.array([[-.2,.2],[.3,-.3]]),'embedding_l2':np.ones(2)}
        cells,groups=summarize_classifier(result,metadata,[True,False],['A','B'])
        np.testing.assert_allclose(cells.annotated_class_probability_delta,[-.2,-.3])
        self.assertEqual(len(groups),2)

if __name__=='__main__':unittest.main()
