import pickle as pkl
from collections import OrderedDict

# train : ['10_1048_img', '10_158_img', '10_1141_img', '10_265_img', '10_1309_img', '10_855_img', '10_687_img', '10_225_img', '10_752_img', '10_832_img', '10_948_img', '10_1166_img', '10_73_img', '10_485_img', '10_108_img', '10_1068_img']
# test : ['10_913_img', '10_128_img', '10_652_img', '10_3_img', '10_193_img', '10_400_img', '10_782_img', '10_1209_img']


l =[OrderedDict([('train', ['LVU_1048', 'LVU_0158', 'LVU_1141', 'LVU_0265', 'LVU_1309', 'LVU_0855', 'LVU_0687', 'LVU_0225', 'LVU_0752', 'LVU_0832', 'LVU_0948', 'LVU_1166', 'LVU_0073', 'LVU_0485', 'LVU_0108', 'LVU_1068' ]),  ('val', ['LVU_0913', 'LVU_0128', 'LVU_0652', 'LVU_0003', 'LVU_0193', 'LVU_0400', 'LVU_0782', 'LVU_1209' ])])]

with open("splits_final_livus.pkl", "wb") as f:
	pkl.dump(l,f)
