import pickle as pkl
from collections import OrderedDict



l =[OrderedDict([('train', ['ABD_006','ABD_007','ABD_009','ABD_010','ABD_021','ABD_023','ABD_024','ABD_026','ABD_027','ABD_031','ABD_033','ABD_034','ABD_039','ABD_040','ABD_005','ABD_028','ABD_030','ABD_037']),  ('val', ['ABD_001','ABD_002','ABD_003','ABD_004','ABD_008','ABD_022','ABD_025','ABD_029','ABD_032','ABD_035','ABD_036','ABD_038'])])]
l+=[OrderedDict([('train', ['ABD_001','ABD_002','ABD_003','ABD_004','ABD_023','ABD_008','ABD_022','ABD_025','ABD_029','ABD_031','ABD_032','ABD_033','ABD_034','ABD_036','ABD_037','ABD_038','ABD_039','ABD_040']),  ('val', ['ABD_035','ABD_006','ABD_007','ABD_009','ABD_010','ABD_021','ABD_005','ABD_024','ABD_026','ABD_027','ABD_028','ABD_030'])])]
l+=[OrderedDict([('train', ['ABD_002','ABD_003','ABD_004','ABD_005','ABD_006','ABD_009','ABD_021','ABD_022','ABD_023','ABD_024','ABD_026','ABD_027','ABD_028','ABD_029','ABD_030','ABD_032','ABD_036','ABD_038']),  ('val', ['ABD_001','ABD_035','ABD_007','ABD_008','ABD_010','ABD_025','ABD_031','ABD_033','ABD_034','ABD_037','ABD_039','ABD_040'])])]
l+=[OrderedDict([('train', ['ABD_001','ABD_005','ABD_006','ABD_007','ABD_008','ABD_009','ABD_010','ABD_021','ABD_023','ABD_024','ABD_025','ABD_027','ABD_029','ABD_031','ABD_032','ABD_036','ABD_038','ABD_039']),  ('val', ['ABD_002','ABD_035','ABD_003','ABD_004','ABD_022','ABD_026','ABD_028','ABD_030','ABD_033','ABD_034','ABD_037','ABD_040'])])]
l+=[OrderedDict([('train', ['ABD_001','ABD_002','ABD_003','ABD_004','ABD_007','ABD_008','ABD_010','ABD_022','ABD_023','ABD_025','ABD_026','ABD_028','ABD_030','ABD_033','ABD_034','ABD_036','ABD_037','ABD_040']),  ('val', ['ABD_005','ABD_006','ABD_009','ABD_021','ABD_024','ABD_027','ABD_029','ABD_031','ABD_032','ABD_035','ABD_038','ABD_039'])])]

with open("splits_final_bcv.pkl", "wb") as f:
	pkl.dump(l,f)
