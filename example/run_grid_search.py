from GridSearch import GridSearch
from SineWave import SineWave
from Utils import print_status
alr_hps = [ {'initial_rate': 0.0001, 'min_rate': 9.99e-5}, {'initial_rate': 0.0001, 'min_rate': 9.99e-5}]
#alr_hps = [ {'initial_rate': 0.0001, 'min_rate': 1e-5}]
hyper_params = {
    'rnn_type': [ 'vanilla'],
    'n_hidden': [180],
    'min_loss': [ 1e-4],
    'log_dir': [ './logs_sine/'],
    'data_hps': [ {
        'n_batch': 512,
        'n_time': 50,
        'n_bits': 1,
        'p_flip': 0.5} ] ,
    'n_trials_plot' : [6],
    'alr_hps': alr_hps
        }

grid_search = GridSearch(SineWave, hyper_params)
grid_search.run()
