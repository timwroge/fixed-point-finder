'''
flipflop.py
Written using Python 2.7.12
@ Matt Golub, August 2018.
Please direct correspondence to mgolub@stanford.edu.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import numpy as np
from numpy import sin as sine
from Utils import print_status
from time import sleep
from random import randrange as rand

if os.environ.get('DISPLAY','') == '':
    # Ensures smooth running across environments, including servers without
    # graphical backends.
    print('No display found. Using non-interactive Agg backend.')
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from RecurrentWhisperer import RecurrentWhisperer

class CenterOut(RecurrentWhisperer):
    ''' Class for training an RNN to implement an N-bit memory, a.k.a. "the
    flip-flop  task" as described in Sussillo & Barak, Neural Computation,
    2013.

    Task:
        Briefly, a set of inputs carry transient pulses (-1 or +1) to set the
        state of a set of binary outputs (also -1 or +1). Each input drives
        exactly one output. If the sign of an input pulse opposes the sign
        currently held at the corresponding output, the sign of the output
        flips. If an input pulse's sign matches that currently held at the
        corresponding output, the output does not change.

        This class generates synthetic data for the flip-flop task via
        generate_flipflop_trials(...).

    Usage:
        This class trains an RNN to generate the correct outputs given the
        inputs of the flip-flop task. All that is needed to get started is to
        construct a flipflop object and to call .train on that object:

        # dict of hyperparameter key/value pairs
        # (see 'Hyperparameters' section below)
        hps = {...}

        ff = SineWave(**hps)
        ff.train()

    Hyperparameters:
        rnn_type: string specifying the architecture of the RNN. Currently
        must be one of {'vanilla', 'gru', 'lstm'}. Default: 'vanilla'.

        n_hidden: int specifying the number of hidden units in the RNN.
        Default: 24.

        data_hps: dict containing hyperparameters for generating synthetic
        data. Contains the following keys:

            'n_batch': int specifying the number of synthetic trials to use
            per training batch (i.e., for one gradient step). Default: 128.

            'n_time': int specifying the duration of each synthetic trial
            (measured in timesteps). Default: 256.

            'n_bits': int specifying the number of input channels into the
            SineWave device (which will also be the number of output channels).
            Default: 3.

            'p_flip': float between 0.0 and 1.0 specifying the probability
            that a particular input channel at a particular timestep will
            contain a pulse (-1 or +1) on top of its steady-state value (0).
            Pulse signs are chosen by fair coin flips, and pulses are produced
            with the same statistics across all input channels and across all
            timesteps (i.e., there are no history effects, there are no
            interactions across input channels). Default: 0.2.

        log_dir: string specifying the top-level directory for saving various
        training runs (where each training run is specified by a different set
        of hyperparameter settings). When tuning hyperparameters, log_dir is
        meant to be constant across models. Default: '/tmp/flipflop_logs/'.

        n_trials_plot: int specifying the number of synthetic trials to plot
        per visualization update. Default: 4.
    '''

    @staticmethod
    def _default_hash_hyperparameters():
        '''Defines default hyperparameters, specific to SineWave, for the set
        of hyperparameters that are hashed to define a directory structure for
        easily managing multiple runs of the RNN training (i.e., using
        different hyperparameter settings). Additional default hyperparameters
        are defined in RecurrentWhisperer (from which SineWave inherits).

        Args:
            None.

        Returns:
            dict of hyperparameters.
        '''
        return {
            'rnn_type': 'vanilla',
            'n_hidden': 24,
            'noise_type': 0,
            'noise_std': 0.1,
            'data_hps': {
                'n_batch': 128,
                'n_time': 256,
                'n_bits': 3,
                'p_flip': 0.2}
            }

    @staticmethod
    def _default_non_hash_hyperparameters():
        '''Defines default hyperparameters, specific to SineWave, for the set
        of hyperparameters that are NOT hashed. Additional default
        hyperparameters are defined in RecurrentWhisperer (from which SineWave
        inherits).

        Args:
            None.

        Returns:
            dict of hyperparameters.
        '''
        return {
            'log_dir': '/tmp/flipflop_logs/',
            'n_trials_plot': 1,

            # DO NOT OVERWRITE THESE VALUES.
            # See docstrings in recurrent_whisperer.py for definitions. The
            # values given here reflect that SineWave does not use (or require)
            # validation data (because all trials are generated independently).
            'do_generate_lvl_visualizations': False,
            'do_save_lvl_visualizations': False,
            'do_save_lvl_ckpt': False,
            }

    def _setup_model(self):
        '''Defines an RNN in Tensorflow.

        See docstring in RecurrentWhisperer.
        '''
        hps = self.hps
        n_hidden = hps.n_hidden

        data_hps = hps.data_hps
        n_batch = data_hps['n_batch']
        n_time = data_hps['n_time']
        n_inputs = data_hps['n_bits']
        n_output = n_inputs

        # Data handling
        self.inputs_bxtxd = tf.placeholder(tf.float32,
            [n_batch, n_time, n_inputs])
        self.output_bxtxd = tf.placeholder(tf.float32,
            [n_batch, n_time, n_output])

        self.rnn_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
        initial_state = self.rnn_cell.zero_state(n_batch, dtype=tf.float32)


        # Readout from RNN
        #cell = tf.contrib.rnn.OutputProjectionWrapper(self.rnn_cell,  output_size=n_output)

        hidden_activity,  _ = tf.nn.dynamic_rnn(self.rnn_cell,
            self.inputs_bxtxd, initial_state=initial_state)
        if(self.hps.noise_type):
            noise = tf.random.normal( tf.shape(hidden_activity), mean=0.0, stddev=hps.noise_std, dtype=tf.dtypes.float32, seed=None, name=None)
            self.hidden_bxtxd = hidden_activity +noise
        else:
            self.hidden_bxtxd = hidden_activity

        np_W_out, np_b_out = self._np_init_weight_matrix(n_hidden, n_output)
        self.W_out = tf.constant(np_W_out, dtype=tf.float32)
        self.b_out = tf.constant(np_b_out, dtype=tf.float32)
        temp_output = tf.tensordot(self.hidden_bxtxd,
            self.W_out, axes=1) + self.b_out
        if(self.hps.noise_type):
            self.pred_output_bxtxd = temp_output
        else:
            noise = tf.random.normal( tf.shape(temp_output), mean=0.0, stddev=hps.noise_std, dtype=tf.dtypes.float32, seed=None, name=None)
            self.pred_output_bxtxd =  temp_output + noise

        # Readout from RNN
        self.loss = tf.reduce_mean(
            tf.squared_difference(self.output_bxtxd, self.pred_output_bxtxd))

    def _setup_training(self, train_data, valid_data):
        '''Does nothing. Required by RecurrentWhisperer.'''
        pass

    def _train_batch(self, batch_data):
        '''Performs a training step over a single batch of data.

        Args:
            batch_data: dict containing one training batch of data. Contains
            the following key/value pairs:

                'inputs': [n_batch x n_time x n_bits] numpy array specifying
                the inputs to the RNN.

                'outputs': [n_batch x n_time x n_bits] numpy array specifying
                the correct output responses to the 'inputs.'

        Returns:
            summary: dict containing the following summary key/value pairs
            from the training step:

                'loss': scalar float evalutaion of the loss function over the
                data batch.

                'grad_global_norm': scalar float evaluation of the norm of the gradient of the loss function with respect to all trainable variables, taken over the data batch.
        '''

        ops_to_eval = [self.train_op,
            self.grad_global_norm,
            self.loss,
            self.tensorboard['merged_opt_summary']]

        feed_dict = dict()
        feed_dict[self.inputs_bxtxd] = batch_data['inputs']
        feed_dict[self.output_bxtxd] = batch_data['output']
        feed_dict[self.learning_rate] = self.adaptive_learning_rate()
        feed_dict[self.grad_norm_clip_val] = self.adaptive_grad_norm_clip()

        [ev_train_op,
         ev_grad_global_norm,
         ev_loss,
         ev_merged_opt_summary] = \
                self.session.run(ops_to_eval, feed_dict=feed_dict)

        if self.hps.do_save_tensorboard_summaries:

            if self._epoch()==0:
                '''Hack to prevent throwing the vertical axis on the
                Tensorboard figure for grad_norm_clip_val (grad_norm_clip val
                is initialized to an enormous number to prevent clipping
                before we know the scale of the gradients).'''
                feed_dict[self.grad_norm_clip_val] = np.nan
                ev_merged_opt_summary = \
                    self.session.run(
                        self.tensorboard['merged_opt_summary'],
                        feed_dict)

            self.tensorboard['writer'].add_summary(
                ev_merged_opt_summary, self._step())

        summary = {'loss': ev_loss, 'grad_global_norm': ev_grad_global_norm}

        return summary

    def predict(self, batch_data, do_predict_full_LSTM_state=False):
        '''Runs the RNN given its inputs.

        Args:
            batch_data:
                dict containing the key 'inputs': [n_batch x n_time x n_bits]
                numpy array specifying the inputs to the RNN.

            do_predict_full_LSTM_state (optional): bool indicating, if the RNN
            is an LSTM, whether to return the concatenated hidden and cell
            states (True) or simply the hidden states (False). Default: False.

        Returns:
            predictions: dict containing the following key/value pairs:

                'state': [n_batch x n_time x n_states] numpy array containing
                the activations of the RNN units in response to the inputs.
                Here, n_states is the dimensionality of the hidden state,
                which, depending on the RNN architecture and
                do_predict_full_LSTM_state, may or may not include LSTM cell
                states.

                'output': [n_batch x n_time x n_bits] numpy array containing
                the readouts from the RNN.

        '''

        if do_predict_full_LSTM_state:
            return self._predict_with_LSTM_cell_states(batch_data)
        else:
            #ops_to_eval = [ self.pred_output_bxtxd, self.hidden_bxtxd]
            ops_to_eval = [self.pred_output_bxtxd]
            feed_dict = {self.inputs_bxtxd: batch_data['inputs']}
            #ev_hidden_bxtxd, ev_pred_output_bxtxd = \
            #    self.session.run(ops_to_eval, feed_dict=feed_dict)
            ev_pred_output_bxtxd = \
                self.session.run(ops_to_eval, feed_dict=feed_dict)
            # get hidden
            ops_to_eval = [self.hidden_bxtxd]
            feed_dict = {self.inputs_bxtxd: batch_data['inputs']}
            #ev_hidden_bxtxd, ev_pred_output_bxtxd = \
            #    self.session.run(ops_to_eval, feed_dict=feed_dict)
            ev_hidden_bxtxd = \
                self.session.run(ops_to_eval, feed_dict=feed_dict)
            ev_hidden_bxtxd = ev_hidden_bxtxd[0]

            predictions = {
                'state': ev_hidden_bxtxd,
                'output': ev_pred_output_bxtxd
                }
            print_status("Hidden size: {} ".format(  np.array(ev_pred_output_bxtxd) .shape ) )

            return predictions

    def _predict_with_LSTM_cell_states(self, batch_data):
        '''Runs the RNN given its inputs.

        The following is added for execution only when LSTM predictions are
        needed for both the hidden and cell states. Tensorflow does not make
        it easy to access the cell states via dynamic_rnn.

        Args:
            batch_data: as specified by predict.

        Returns:
            predictions: as specified by predict.

        '''

        hps = self.hps
        if hps.rnn_type != 'lstm':
            return self.predict(batch_data)

        n_hidden = hps.n_hidden
        [n_batch, n_time, n_bits] = batch_data['inputs'].shape
        initial_state = self.rnn_cell.zero_state(n_batch, dtype=tf.float32)

        ''' Add ops to the graph for getting the complete LSTM state
        (i.e., hidden and cell) at every timestep.'''
        self.full_state_list = []
        for t in range(n_time):
            input_ = self.inputs_bxtxd[:,t,:]
            if t == 0:
                full_state_t_minus_1 = initial_state
            else:
                full_state_t_minus_1 = self.full_state_list[-1]
            _, full_state_bxd = self.rnn_cell(input_, full_state_t_minus_1)
            self.full_state_list.append(full_state_bxd)

        '''Evaluate those ops'''
        ops_to_eval = [self.full_state_list, self.pred_output_bxtxd]
        feed_dict = {self.inputs_bxtxd: batch_data['inputs']}
        ev_full_state_list, ev_pred_output_bxtxd = \
            self.session.run(ops_to_eval, feed_dict=feed_dict)

        '''Package the results'''
        h = np.zeros([n_batch, n_time, n_hidden]) # hidden states: bxtxd
        c = np.zeros([n_batch, n_time, n_hidden]) # cell states: bxtxd
        for t in range(n_time):
            h[:,t,:] = ev_full_state_list[t].h
            c[:,t,:] = ev_full_state_list[t].c

        ev_LSTMCellState = tf.nn.rnn_cell.LSTMStateTuple(h=h, c=c)

        predictions = {
            'state': ev_LSTMCellState,
            'output': ev_pred_output_bxtxd
            }

        return predictions

    def _get_data_batches(self, train_data):
        '''See docstring in RecurrentWhisperer.'''
        return [self.generate_trials()]

    def _get_batch_size(self, batch_data):
        '''See docstring in RecurrentWhisperer.'''
        return batch_data['inputs'].shape[0]

    def generate_trials(self):
        '''Generates synthetic data (i.e., ground truth trials) for the
        SineWave task. See comments following FlipFlop class definition for a
        description of the input-output relationship in the task.

        Args:
            None.

        Returns:
            dict containing 'inputs' and 'outputs'.

                'inputs': [n_batch x n_time x n_bits] numpy array containing
                input pulses.

                'outputs': [n_batch x n_time x n_bits] numpy array specifying
                the correct behavior of the SineWave memory device.
        '''

        data_hps = self.hps.data_hps
        n_batch = data_hps['n_batch']
        n_time = data_hps['n_time']
        # number of dimensions for the center out task
        n_bits = data_hps['n_bits']
        p_flip = data_hps['p_flip']
        max_freq = 20
        min_freq = 7
        self.rng.seed(7);
        self.max_samples = 71
        if n_bits   != 2:
            raise Exception("The number of dimensions must be 2 for x and y components of the center out task" )
        sqrt_2 = ( 2 ** (1/2.0) ) / 2.0

        x_and_y_options = [ (0.0, 1), (sqrt_2, sqrt_2), (-sqrt_2, sqrt_2),(sqrt_2, -sqrt_2),(-sqrt_2, -sqrt_2),(0, -1),  (-1, 0),  (1, 0)]

        # Randomly generate unsigned input pulses
        inputs = np.zeros([n_batch, n_time, n_bits])
        outputs = np.zeros([n_batch, n_time, n_bits])
        when_trial_begins = int( n_time/4)
        when_trial_ends = int( n_time - n_time/4)
        conditions = [] 

        for batch in range(n_batch):
            #pick a random number for the x dimension
            choice  = self.rng.choice(8)
            conditions.append(choice)
            (x_component, y_component)  = x_and_y_options[choice]

            inputs[batch, when_trial_begins:, 0] = x_component
            inputs[batch, when_trial_begins:, 1] = y_component
            outputs[batch, when_trial_begins:, 0] = x_component
            outputs[batch, when_trial_begins:, 1] = y_component

        return {'inputs': inputs, 'output': outputs, 'condition': conditions}

    def update_visualizations(self,
        train_data=None,
        valid_data=None,
        is_final=False):
        '''See docstring in RecurrentWhisperer.'''
        data = self.generate_trials()
        self.plot_trials(data)
        self.refresh_figs()

    def trial_input_to_color(self, condition):
        colors = [ 'cyan', 'deepskyblue', 'darkblue', 'y', 'r', 'orange', 'm', 'g' ]
        return colors[condition]

    def plot_trials(self, data, start_time=0, stop_time=None):
        '''Plots example trials, complete with input pulses, correct outputs,
        and RNN-predicted outputs.

        Args:
            data: dict as returned by generate_flipflop_trials.

            start_time (optional): int specifying the first timestep to plot.
            Default: 0.

            stop_time (optional): int specifying the last timestep to plot.
            Default: n_time.

        Returns:
            None.
        '''

        FIG_WIDTH = 6 # inches
        FIG_HEIGHT = 13 # inches

        fig = self._get_fig('example_trials',
            width=FIG_WIDTH,
            height=FIG_HEIGHT)

        hps = self.hps
        n_batch = self.hps.data_hps['n_batch']
        n_time = self.hps.data_hps['n_time']
        n_plot = np.min([hps.n_trials_plot, n_batch])

        inputs = data['inputs']
        output = data['output']
        conditions = data['condition']
        predictions = self.predict(data)
        pred_output = np.array( predictions['output'][0])

        if stop_time is None:
            stop_time = n_time

        time_idx = range(start_time, stop_time)

        for trial_idx in range(1, n_plot+1):
            ax = fig.add_subplot(n_plot, 1, trial_idx)
            if n_plot == 1:
                ax.set_title('Example trial', fontweight='bold')
            else:
                ax.set_title('Example trial %d' % (trial_idx), fontweight='bold')

            self._plot_single_trial(
                inputs[trial_idx, time_idx, :],
                output[trial_idx, time_idx, :],
                pred_output[trial_idx, time_idx, :], ax)

            if trial_idx < (n_plot-1):
                ax.set_xticks([])
            else:
                ax.set_xlabel('Timestep', fontweight='bold')
        # print out the center out trials
        fig2 = self._get_fig('center_out',
            no_fixed_size=True)

        sq = 2**(1/2.0) /2.0
        #<my_trial>
        n_center_out = 1
        for trial_idx in range(1, n_center_out+1):
            ax = fig2.add_subplot(n_center_out, 1, trial_idx)
            #plt.subplots(sharex=True)
            # points counter clockwise around the circle
            xs = [1, sq, 0, -sq, -1, -sq,  0,  sq]
            ys = [0, sq, 1,  sq,  0, -sq, -1, -sq]
            #make sure the scale is correct
            ax.scatter(xs, ys, color='white' )
            ax.set(adjustable='box-forced', aspect='equal')
            circle_c = plt.Circle((0, 0), 0.05, color='gray')
            circle_0 = plt.Circle((0.0, 1), 0.05, color='cyan')
            circle_1 = plt.Circle((sq, sq), 0.05, color='deepskyblue')
            circle_2 = plt.Circle((-sq, sq), 0.05, color='darkblue')
            circle_3 = plt.Circle((sq, -sq), 0.05, color='y')
            circle_4 = plt.Circle((-sq, -sq), 0.05, color='r')
            circle_5 = plt.Circle((0, -1), 0.05, color='orange')
            circle_6 = plt.Circle((-1, 0), 0.05, color='m')
            circle_7 = plt.Circle(( 1, 0), 0.05, color='g')

            ax.add_artist(circle_c)
            ax.add_artist(circle_0)
            ax.add_artist(circle_1)
            ax.add_artist(circle_2)
            ax.add_artist(circle_3)
            ax.add_artist(circle_4)
            ax.add_artist(circle_5)
            ax.add_artist(circle_6)
            ax.add_artist(circle_7)
            ax.add_artist(circle_7)

            print(pred_output.shape)
            #current_trial = pred_output[trial_idx, :, :]
            for reach in range(min(64, n_batch)):
                current_trial = pred_output[reach, :, :]
                xs = current_trial[:, 0]
                ys = current_trial[:, 1]

                ax.plot(
                        xs, ys, 0.3, color = self.trial_input_to_color(conditions[reach]  )
                    )
                ax.scatter(
                    xs, ys, 0.5, color = 'black'
                    )




    @staticmethod
    def _plot_single_trial(input_txd, output_txd, pred_output_txd, ax):

        VERTICAL_SPACING = 2.5
        [n_time, n_bits] = input_txd.shape
        tt = range(n_time)

        y_ticks = [VERTICAL_SPACING*bit_idx for bit_idx in range(n_bits)]
        #y_tick_labels = \
        #    ['Bit %d' % (n_bits-bit_idx) for bit_idx in range(n_bits)]
        y_tick_labels = 'Y    X'

        ax.set_yticks(y_ticks)
        ax.set_ylabel(y_tick_labels, fontweight='bold')
        for bit_idx in range(n_bits):

            vertical_offset = VERTICAL_SPACING*bit_idx

            # Input pulses
            ax.fill_between(
                tt,
                vertical_offset + input_txd[:, bit_idx],
                vertical_offset,
                step='mid',
                color='gray')

            # Correct outputs
            ax.step(
                tt,
                vertical_offset + output_txd[:, bit_idx],
                where='mid',
                linewidth=2,
                color='cyan')

            # RNN outputs
            ax.step(
                tt,
                vertical_offset + pred_output_txd[:, bit_idx],
                where='mid',
                color='purple',
                linewidth=1.5,
                linestyle='--')

        ax.set_xlim(-1, n_time)
