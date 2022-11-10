import pickle as pkl
from collections import OrderedDict



l =[OrderedDict([('train', ['bcv_6','bcv_7','bcv_9','bcv_10','bcv_21','bcv_23','bcv_24','bcv_26','bcv_27','bcv_31','bcv_33','bcv_34','bcv_39','bcv_40','bcv_5','bcv_28','bcv_30','bcv_37']),('val', ['bcv_1','bcv_2','bcv_3','bcv_4','bcv_8','bcv_22','bcv_25','bcv_29','bcv_32','bcv_35','bcv_36','bcv_38'])])]
l+=[OrderedDict([('train', ['bcv_1','bcv_2','bcv_3','bcv_4','bcv_23','bcv_8','bcv_22','bcv_25','bcv_29','bcv_31','bcv_32','bcv_33','bcv_34','bcv_36','bcv_37','bcv_38','bcv_39','bcv_40']), ('val', ['bcv_35','bcv_6','bcv_7','bcv_9','bcv_10','bcv_21','bcv_5','bcv_24','bcv_26','bcv_27','bcv_28','bcv_30'])])]
l+=[OrderedDict([('train', ['bcv_2','bcv_3','bcv_4','bcv_5','bcv_6','bcv_9','bcv_21','bcv_22','bcv_23','bcv_24','bcv_26','bcv_27','bcv_28','bcv_29','bcv_30','bcv_32','bcv_36','bcv_38']),  ('val', ['bcv_1','bcv_35','bcv_7','bcv_8','bcv_10','bcv_25','bcv_31','bcv_33','bcv_34','bcv_37','bcv_39','bcv_40'])])]
l+=[OrderedDict([('train', ['bcv_1','bcv_5','bcv_6','bcv_7','bcv_8','bcv_9','bcv_10','bcv_21','bcv_23','bcv_24','bcv_25','bcv_27','bcv_29','bcv_31','bcv_32','bcv_36','bcv_38','bcv_39']),  ('val', ['bcv_2','bcv_35','bcv_3','bcv_4','bcv_22','bcv_26','bcv_28','bcv_30','bcv_33','bcv_34','bcv_37','bcv_40'])])]
l+=[OrderedDict([('train', ['bcv_1','bcv_2','bcv_3','bcv_4','bcv_7','bcv_8','bcv_10','bcv_22','bcv_23','bcv_25','bcv_26','bcv_28','bcv_30','bcv_33','bcv_34','bcv_36','bcv_37','bcv_40']),  ('val', ['bcv_5','bcv_6','bcv_9','bcv_21','bcv_24','bcv_27','bcv_29','bcv_31','bcv_32','bcv_35','bcv_38','bcv_39'])])]

with open("splits_final.pkl", "wb") as f:
	pkl.dump(l,f)
