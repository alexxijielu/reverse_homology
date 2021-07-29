import os
import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder

'''Dataset object stores and formats sequences into one-hot batches for training'''
class Dataset(object):
    def __init__(self, map_path=None, min_count=9):
        self.file_ids = []
        self.file_info = []
        self.enc = LabelEncoder().fit(['G', 'A', 'L', 'M', 'F', 'W', 'K', 'Q', 'E', 'S',
                                       'P', 'V', 'I', 'C', 'Y', 'H', 'R', 'N', 'D', 'T'])
        self.seq_length = 256           # The length of sequences to process into
        self.map_path = map_path        # The IDR to ORF mapping file (used for yeast)
        self.min_count = min_count      # The minimum number of IDRs per homologue family
        self.num_files = 0

    '''Load each fasta file into Dataset object; called by the add_dataset function
    file_id: Unique numerical ID for fasta file
    name: Name of the IDR
    count: How many sequences there are in the homology family
    sequences: Array of sequences
    species_labels: Species of each sequence'''
    def add_file(self, file_id, name, count, sequences, species_labels):
        file_info = {
            "id": file_id,
            "name": name,
            "count": count,
            "sequences": sequences,
            "species_labels": species_labels
        }
        self.file_info.append(file_info)

    '''Given a directory of fasta files, load all fasta files into Dataset object
    root_dir: Path to directory of fasta files
    clip_M: Whether to remove the starting methionine from IDRs at the N-terminal'''
    def add_dataset(self, root_dir, clip_M=False):
        if self.map_path:
            # Open map path (for yeast) - this lets us assign start/end positions for IDRs
            map = open(self.map_path)
            assignments = csv.reader(map, delimiter=',')
            assignments = np.array([row for row in assignments])
            orfs = assignments[1:, 1]
            starts = assignments[1:, 2]

        i = 0
        totals = 0
        counts = []
        # Iterate through fasta files in the directory
        for curr_file_name in os.listdir(root_dir):
            if ".fasta" in curr_file_name:
                print("Adding " + curr_file_name)
                curr_path = root_dir + curr_file_name
                curr_name = curr_file_name.split(".fasta")[0]

                # If clip_M, remove the starting methionine if the IDR is at the N-terminal
                if clip_M and self.map_path:
                    remove_M = False
                    # Human IDRs have their positions included in the naming convention, so just use these
                    if "HUMAN" in curr_name:
                        curr_start = curr_name.split("_")[1].split("to")[0]
                        if curr_start == '1':
                            remove_M = True     # If we can't get the position, just assume it's N-terminal to be safe
                    else:
                        # For yeast IDRs, we have to use a special map file
                        try:
                            curr_start = starts[np.where(orfs == curr_name.split(".")[0])[0][0]]
                            if curr_start == '1':
                                remove_M = True
                        except IndexError:
                            print("Couldn't find ", curr_name, " in the IDR mapping")
                            remove_M = True     # If we can't get the position, just assume it's N-terminal to be safe

                # Open sequences (filtered - remove anything with an 'x" and under/past certain lengths)
                sequences = []
                species_labels = []
                contents = open(curr_path).read().split("\n")
                species = [idx for idx, s in enumerate(contents) if '>' in s]

                for index in species:
                    curr_index = index + 1
                    # Remove the alignment padding token
                    sequence = contents[curr_index].replace("-", "")

                    if clip_M and self.map_path:
                        # Remove the starting M if it passed the check
                        if remove_M:
                            if len(sequence) > 0:
                                if sequence[0] == 'M':
                                    sequence = sequence[1:]

                    if 'X' in sequence or len(sequence) < 5:
                        pass  # filter low-quality or small sequences
                    else:
                        sequences.append(sequence)
                        species_labels.append(contents[index].split('>')[1])

                # Store all of the sequences as an array and load it into the Dataset object
                sequences = np.array(sequences)
                count = len(sequences)
                counts.append(count)
                if count >= self.min_count:
                    self.add_file(
                        file_id=i,
                        name=curr_name,
                        count=count,
                        sequences=sequences,
                        species_labels=species_labels)
                    i += 1
                    totals += count

    '''Given a list of indices, construct a batch
    file_ids: List of indices to include in batch
    ref_size: How many reference sequences to include in query set
    comp_size: How many target sequences to include in target set'''
    def load_list(self, file_ids, ref_size=8, comp_size=400):
        bsize = len(file_ids)

        # Initialize data arrays
        ref_in = np.zeros((bsize, ref_size, self.seq_length, 20))
        comp_in = np.zeros((1, comp_size, self.seq_length, 20))
        ref_mask = np.zeros((bsize, ref_size, self.seq_length, 1))
        comp_mask = np.zeros((1, comp_size, self.seq_length, 1))
        labels = np.zeros((bsize, comp_size))

        # Check to make sure we can actually return a valid sample
        if comp_size > len(self.file_ids):
            raise ValueError("Size of target (comp_size) cannot be larger than the number of files provided")

        # Sample positive examples for each file id in batch
        for i in range(0, bsize):
            curr_file_id = file_ids[i]
            curr_sequences = self.file_info[curr_file_id]['sequences']
            rand_seqs = curr_sequences[np.random.choice(curr_sequences.shape[0], replace=False, size=ref_size + 1)]

            # Encode randomly selected sequences
            enc_seqs = []
            for j in range(0, len(rand_seqs)):
                curr_seq = rand_seqs[j]
                if len(curr_seq) > self.seq_length:
                    curr_seq = curr_seq[:self.seq_length // 2] + curr_seq[len(curr_seq) - self.seq_length // 2:]
                else:
                    while len(curr_seq) < self.seq_length:
                        curr_seq = curr_seq + curr_seq
                    if len(curr_seq) > self.seq_length:
                        curr_seq = curr_seq[:self.seq_length]
                enc_seqs.append(self.enc.transform(list(curr_seq)))
            enc_seqs = np.array(enc_seqs)
            enc_seqs_mask = np.expand_dims(np.ones(enc_seqs.shape), axis=-1)
            enc_seqs_mask[np.where(enc_seqs == 0)] = 0
            enc_seqs_onehot = np.eye(20)[enc_seqs]

            # Populate data container
            ref_in[i, :, :, :] = enc_seqs_onehot[:ref_size, :, :]
            comp_in[:, i, :, :] = enc_seqs_onehot[-1, :, :]
            ref_mask[i, :, :, :] = enc_seqs_mask[:ref_size, :, :]
            comp_mask[:, i, :, :] = enc_seqs_mask[-1, :, :]
            labels[i, i] = 1

        # Randomly retrieve negative samples from all files (minus files in file_ids)
        n_negative = comp_size - bsize
        p_negative = np.ones(len(self.file_ids))
        p_negative[file_ids] = 0
        p_negative = p_negative / np.sum(p_negative)
        negative_files = np.random.choice(self.file_ids, p=p_negative, replace=False, size=n_negative)

        for i in range(0, len(negative_files)):
            curr_file_id = negative_files[i]
            curr_sequences = self.file_info[curr_file_id]['sequences']
            curr_seq = curr_sequences[np.random.choice(curr_sequences.shape[0])]

            if len(curr_seq) > self.seq_length:
                curr_seq = curr_seq[:self.seq_length // 2] + curr_seq[len(curr_seq) - self.seq_length // 2:]
            else:
                while len(curr_seq) < self.seq_length:
                    curr_seq = curr_seq + curr_seq
                if len(curr_seq) > self.seq_length:
                    curr_seq = curr_seq[:self.seq_length]
            enc_seq = self.enc.transform(list(curr_seq))
            enc_seq_mask = np.expand_dims(np.ones(enc_seq.shape), axis=-1)
            enc_seq_mask[np.where(enc_seq == 0)] = 0
            enc_seq_onehot = np.eye(20)[enc_seq]

            # Add negative sample to comparison data containers
            comp_in[0, i + bsize, :, :] = enc_seq_onehot
            comp_mask[0, i + bsize, :, :] = enc_seq_mask

        # Shuffle comparison indices
        rand_idxs = np.random.permutation(comp_size)
        comp_in = comp_in[:, rand_idxs, :, :]
        comp_mask = comp_mask[:, rand_idxs, :, :]
        labels = labels[:, rand_idxs]

        return ref_in, comp_in, ref_mask, comp_mask, labels

    '''Retrieve all sequences as one-hot for a given file ID (used for debugging and post-training feature extraction'''
    def load_all_sequences(self, file_id):
        pad = self.seq_length

        # sample from target image
        sequences = self.file_info[file_id]['sequences']
        name = self.file_info[file_id]['name']

        # pad sequences and transform into one-hot
        transformed_seqs = []
        for i in range(0, len(sequences)):
            curr_seq = sequences[i]
            if len(curr_seq) > self.seq_length:
                curr_seq = curr_seq[:self.seq_length // 2] + curr_seq[len(curr_seq) - self.seq_length // 2:]
            else:
                while len(curr_seq) < self.seq_length:
                    curr_seq = curr_seq + curr_seq
                if len(curr_seq) > self.seq_length:
                    curr_seq = curr_seq[:self.seq_length]
            transformed_seqs.append(self.enc.transform(list(curr_seq)))
        transformed_seqs = np.array(transformed_seqs)

        masks = np.ones(transformed_seqs.shape)
        masks[np.where(transformed_seqs == 0)] = 0

        transformed_seqs = np.eye(20)[transformed_seqs]
        return transformed_seqs, masks, name

    '''Retrieve all sequences as one-hot for a given file ID. Identical to previous function but also retrieves 
    species associated with each sequence'''
    def load_all_sequences_with_species(self, file_id):
        pad = self.seq_length

        # sample from target image
        sequences = self.file_info[file_id]['sequences']
        name = self.file_info[file_id]['name']
        species = self.file_info[file_id]['species_labels']

        # pad sequences and transform into one-hot
        transformed_seqs = []
        for i in range(0, len(sequences)):
            curr_seq = sequences[i]
            if len(curr_seq) > self.seq_length:
                curr_seq = curr_seq[:self.seq_length // 2] + curr_seq[len(curr_seq) - self.seq_length // 2:]
            else:
                while len(curr_seq) < self.seq_length:
                    curr_seq = curr_seq + curr_seq
                if len(curr_seq) > self.seq_length:
                    curr_seq = curr_seq[:self.seq_length]
            transformed_seqs.append(self.enc.transform(list(curr_seq)))
        transformed_seqs = np.array(transformed_seqs)

        masks = np.ones(transformed_seqs.shape)
        masks[np.where(transformed_seqs == 0)] = 0

        transformed_seqs = np.eye(20)[transformed_seqs]
        return transformed_seqs, masks, name, species

    '''Retrieve all sequences as one-hot for a given file ID. Identical to previous function but retrieves 
    length of each sequence'''
    def load_all_sequences_with_lengths(self, file_id):
        pad = self.seq_length

        # sample from target image
        sequences = self.file_info[file_id]['sequences']
        name = self.file_info[file_id]['name']

        # pad sequences and transform into one-hot
        transformed_seqs = []
        lengths = []
        for i in range(0, len(sequences)):
            curr_seq = sequences[i]
            curr_len = len(curr_seq)
            if len(curr_seq) > self.seq_length:
                curr_seq = curr_seq[:self.seq_length // 2] + curr_seq[len(curr_seq) - self.seq_length // 2:]
                curr_len = len(curr_seq)
            else:
                while len(curr_seq) < self.seq_length:
                    curr_seq = curr_seq + curr_seq
                if len(curr_seq) > self.seq_length:
                    curr_seq = curr_seq[:self.seq_length]
            transformed_seqs.append(self.enc.transform(list(curr_seq)))
            lengths.append(curr_len)
        transformed_seqs = np.array(transformed_seqs)

        masks = np.ones(transformed_seqs.shape)
        masks[np.where(transformed_seqs == 0)] = 0

        transformed_seqs = np.eye(20)[transformed_seqs]
        return transformed_seqs, masks, name, lengths

    '''Retrieve all sequences as one-hot for a given file ID. Identical to previous function but retrieves 
    length AND species of each sequence'''
    def load_all_sequences_with_lengths_and_species(self, file_id):
        pad = self.seq_length

        # sample from target image
        sequences = self.file_info[file_id]['sequences']
        name = self.file_info[file_id]['name']
        species = self.file_info[file_id]['species_labels']

        # pad sequences and transform into one-hot
        transformed_seqs = []
        lengths = []
        for i in range(0, len(sequences)):
            curr_seq = sequences[i]
            curr_len = len(curr_seq)
            if len(curr_seq) > self.seq_length:
                curr_seq = curr_seq[:self.seq_length // 2] + curr_seq[len(curr_seq) - self.seq_length // 2:]
                curr_len = len(curr_seq)
            else:
                while len(curr_seq) < self.seq_length:
                    curr_seq = curr_seq + curr_seq
                if len(curr_seq) > self.seq_length:
                    curr_seq = curr_seq[:self.seq_length]
            transformed_seqs.append(self.enc.transform(list(curr_seq)))
            lengths.append(curr_len)
        transformed_seqs = np.array(transformed_seqs)

        masks = np.ones(transformed_seqs.shape)
        masks[np.where(transformed_seqs == 0)] = 0

        transformed_seqs = np.eye(20)[transformed_seqs]
        return transformed_seqs, masks, name, lengths, species

    """Prepares the Dataset class for use - solves any mismatches or overlaps in file_ids"""
    def prepare(self):
        # Build (or rebuild) everything else from the info dicts.
        self.num_files = len(self.file_info)
        self.file_ids = np.arange(self.num_files)