import os

comp_size = 400
ref_size = 8
min_count = ref_size + 1

batch_size = 64
learning_rate = 1e-4
epochs = 10000
save_period = 200

checkpoint_path = './yeast_idr_model/'
data_path = './sc_idr_alignments/'

map_path = './final_list_idrs.csv'

if checkpoint_path != '' and not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)
