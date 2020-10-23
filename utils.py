import pandas as pd
import numpy as np

from keras.preprocessing.sequence import pad_sequences


def make_seq(seqs, vocab_size, min_len, max_len):
    clip_seq = []
    for i, seq in enumerate(seqs):
        if len(seq) >= min_len and len(seq) <= max_len:
            clip_seq.append(seq)
    clip_seq = np.array(clip_seq)
    padded_seq = pad_sequences(clip_seq, maxlen=max_len, padding='post')
    temp = np.zeros((padded_seq.shape[0], max_len, vocab_size+1))
    for i,s in enumerate(padded_seq):
        for j,t in enumerate(s):
            temp[i,j,t] = 1
    x_train = temp
    
    return x_train

def write_learned_pattern(prob, all_char):
    char_char_and_blank = {}
    char_char_and_blank[0] = 'BLANK'
    for i,c in enumerate(all_char):
        char_char_and_blank[i+1] = c
    prob_df = pd.DataFrame()
    for i in range(len(prob)):
        prob_df[i] = prob[:][i]
    prob_df.index = [char_char_and_blank[i] for i in char_char_and_blank]
    prob_df.to_csv('learned_pattern.csv' )
    return

def _sample_with_temp(preds, temp=1):
    streched = np.log(preds) / temp
    stretched_probs = np.exp(streched) / np.sum(np.exp(streched))
    return np.random.choice(len(streched), p=stretched_probs)

def write_fasta(peptides, AMP_rank = 0):
        fasta_file = 'AMPs_%s.fa' % (AMP_rank)
        fp = open(fasta_file , 'w')
        for i, f in enumerate(peptides):
            fp.write('>%s|%s\n%s\n' %(i,i+1,f))
        fp.close()
        return
            
def make_peptides(temp, prob, all_char, generate_seq = 2000, max_patient = 10000):
    print ('Sampling peptides with Temp = %.1f ......' % temp)
    peptides = {}
    i = 0
    patient = 0
    while patient < max_patient :
        peptide = []
        for t in prob:
            arg = _sample_with_temp(t, temp = temp)
            if arg == 0:
                break  
            amino = all_char[arg-1]
            peptide.append(amino)     
        peptide = ''.join(peptide)
        if len(peptide) > 20 and peptide not in peptides:
            peptides[i] = peptide
            i += 1
        patient += 1
        if i == generate_seq:
            break

    peptides = [peptide for peptide in peptides.values()]
    generated_len = len(peptides)
    pass_rate = generated_len/patient
    print ('Peptdies generation complete! %s unique peptides were generated, generating success rate = %.3f.' % (generated_len, pass_rate))
    return peptides 
