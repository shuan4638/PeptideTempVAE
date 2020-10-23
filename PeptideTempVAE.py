import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives

from utils import *
    
AMPs = pd.read_csv('https://raw.githubusercontent.com/alexarnimueller/LSTM_peptides/master/training_sequences_noC.csv',header = None)

all_char = []
all_len = []
max_len = 0

for i in AMPs.index:
    seq = AMPs[0][i]
    local_len = len(list(seq))
    all_len.append(local_len)
    if local_len > max_len:
        max_len = local_len
    for j in seq:
        if j not in all_char:
            all_char.append(j)

char2idx = {u:i+1 for i, u in enumerate(all_char)}

vocab_size = len(all_char)
seq = np.array([[char2idx[c] for c in i] for i in AMPs[0]])

x_train = make_seq(seq, vocab_size, min_len = 30, max_len = 40)

def VAE(x_train, nb_epoch = 300):
    batch_size = 100
    original_dim = vocab_size
    latent_dim = 2
    intermediate_dim = 100
    epsilon_std = 1.0
    input_len = np.prod(x_train.shape[1:])

    x = Input(shape=(input_len,))
    h = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    def sampling(args): 
        z_mean, z_log_var = args
        batch_size = K.shape(z_mean)[0]
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev = epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
 
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(input_len, activation='sigmoid')
    z_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(z_decoded)

    def vae_loss(x, x_decoded_mean):
        xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss
    
    x_train_choice = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    vae = Model(x, x_decoded_mean)
    vae.compile(optimizer='adam', loss=vae_loss)
    vae.fit(x_train_choice, x_train_choice,
        shuffle=True,
        nb_epoch=nb_epoch,
        verbose=2,
        batch_size=batch_size)   
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    
    decoder = Model(decoder_input, _x_decoded_mean)
    encoder = Model(x, z_mean)
    x_encoded = encoder.predict(x_train_choice, batch_size=batch_size)
    
    return vae, encoder, decoder, x_train_choice

vae, encoder, decoder, x_train_choice = VAE(x_train)

# Sample from (0,0)
sample_data_point = np.array([[0]*2])
x_decoded = decoder.predict(sample_data_point)
prob = x_decoded.reshape(x_train.shape[1], vocab_size+1)

write_learned_pattern(prob, all_char)

sample_Ts = np.arange(0.1, 1.1, 0.1)
def encoding(sequences, max_len = max_len):
    seq = np.array([[char2idx[c] for c in i] for i in sequences])
    seq_onehot = make_seq(seq, vocab_size, min_len = 30, max_len = 40)
    seq_onehot = seq_onehot.reshape((len(seq_onehot), np.prod(seq_onehot.shape[1:])))
    encoded =  encoder.predict(seq_onehot)
    return encoded

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
for i, T in enumerate(sample_Ts):
    T = 1.1 - T
    peptides = make_peptides(T, prob, all_char)
    write_fasta(peptides, i)
    encoded_data = encoding(peptides)
    ax.scatter(encoded_data[:, 0], encoded_data[:, 1], label = 'T = %.1f' % T)

ax.legend()
plt.savefig('Generated_encoded', dpi=500)

