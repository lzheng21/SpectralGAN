modes = ["gen", "dis"]

# training settings
batch_size_gen = 8  # batch size for the generator
batch_size_dis = 8  # batch size for the discriminator
lambda_gen = 0.001  # l2 loss regulation weight for the generator
lambda_dis = 0.001  # l2 loss regulation weight for the discriminator
n_sample_gen = 40  # number of samples for the generator
lr_gen = 1e-3  # learning rate for the generator
lr_dis = 1e-3  # learning rate for the discriminator
n_epochs = 30000  # number of outer loops
n_epochs_gen = 100  # number of inner loops for the generator
n_epochs_dis = 100  # number of inner loops for the discriminator
gen_interval = 1 #n_epochs_gen  # sample new nodes for the generator for every gen_interval iterations
dis_interval = 1 #n_epochs_dis  # sample new nodes for the discriminator for every dis_interval iterations

# model saving
load_model = False  # whether loading existing model for initialization
save_steps = 10

# other hyper-parameters
emb_dim = 16
missing_edge = 128
n_layers = 3
n_eigs = 9719



# path settings
train_filename = "../data/ml-1m/train_users.dat"
test_filename = "../data/ml-1m/test_users.dat"
