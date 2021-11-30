import os

comp_size = 400                 # Total size of contrastive target set (1 positive + comp_size - 1 negative examples)
ref_size = 8                    # How many sequences to put into the query set?
min_count = ref_size + 1        # Filters minimum number of sequences per homologue set

batch_size = 64
learning_rate = 1e-4
epochs = 10000
save_period = 200

checkpoint_path = './human_idr_model/'
data_path = './hs_idr_alignments/'

if checkpoint_path != '' and not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)
