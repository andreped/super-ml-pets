# -*- coding: utf-8 -*-
""" Flexible LSTM implementation taken from https://github.com/widmi/widis-lstm-tools

There is a lot of overhead in this file because it includes a highly flexible LSTM cell and LSTM layer implementation.
It supports
  - individual initialization of forward and recurrent LSTM weights and biases,
  - disabling connections to gates or cell input (forward and/or recurrent) via initializers,
  - individual activation functions for LSTM gates, cell input, and cell output,
  - access to cell state, cell output, and gate activations,
  - initialization of (optionally trainable) initial cell state and cell output,
  - modification of recurrent inputs,
  - PyTorch TorchScript optimization,
  - ponder-/tickersteps with dedicated bias weights (https://arxiv.org/abs/1603.08983),
  - return of LSTM activation at every or only last sequence position,
  - inputformat 'NLC' [samples, sequence positions, in_features] or 'NCL',
  - plotting of the LSTM internals
but only a small part of this functionality is required to run our RUDDER example.
"""

import os
import time
from collections import OrderedDict
from itertools import zip_longest
from typing import List

import numpy as np
import torch
import torch.jit as jit
import torch.nn as nn
from matplotlib import pyplot as plt


class LSTMCell(nn.Module):
    def __init__(self, n_fwd_features, n_lstm, n_rec_features=None,
                 w_ci=nn.init.xavier_normal_, w_ig=nn.init.xavier_normal_, w_og=nn.init.xavier_normal_, w_fg=False,
                 b_ci=nn.init.normal_, b_ig=nn.init.normal_, b_og=nn.init.normal_, b_fg=False,
                 a_ci=torch.tanh, a_ig=torch.sigmoid, a_og=torch.sigmoid, a_fg=torch.sigmoid, a_out=torch.tanh,
                 b_ci_tickers=nn.init.normal_, b_ig_tickers=nn.init.normal_, b_og_tickers=nn.init.normal_,
                 b_fg_tickers=False, dtype: torch.dtype = torch.float32):
        """Flexible LSTM cell supporting
          - individual initialization of forward and recurrent LSTM weights and biases,
          - disabling connections to gates or cell input (forward and/or recurrent) via initializers,
          - individual activation functions for LSTM gates, cell input, and cell output,
          - access to cell state, cell output, and gate activations,
          - modification of recurrent inputs,
          - ponder-/tickersteps with dedicated bias weights (https://arxiv.org/abs/1603.08983).

        Provides two functions to compute the next LSTM state:
          - cell(fwd_inputs, rec_input, c_old)
          - cell_tickersteps(rec_input, c_old)
        with
          - fwd_inputs of shape [samples, n_fwd_features],
          - rec_input of shape [samples, n_rec_features], where n_rec_features defaults to n_lstm, and
          - old cell state c_old of shape [samples, n_lstm].

        Abbreviations:
         - ci = cell input,
         - ig = input gate,
         - og = output gate,
         - fg = forget gate weights,
         - out = cell output.

        Default: Forget gate disabled, other connections and biases enabled,
        tanh cell input and cell output activation function,
        sigmoid gate activation functions, nn.init.xavier_normal_ initializer for weights and
        nn.init.normal_ for biases.

        Parameters
        -------
        n_fwd_features : int
            Number of forward input features
        n_lstm : int
            Number of LSTM blocks
        n_rec_features : int
            Number of recurrent input features (=number of LSTM blocks if feeding the LSTM output back as recurrence).
            Defaults to n_lstm.
        w_ci, w_ig, w_og, w_fg : function or list of function or tuple of function bool
            Initializer function(s) for respective weights;
            If 2-element list: Interpreted as [w_fwd, w_rec] to define different weight initializations for forward and
            recurrent connections respectively;
            If single element: forward and recurrent connections will use the same initializer function;
            If False: connection will be cut (single element of 2-element tuple may be set to False as well);
            Shape of weights will be w_fwd: [n_fwd_features, n_lstm], w_rec: [n_rec_features, n_lstm];
        b_ci, b_ig, b_og, b_fg : function or bool
            Initializer function for respective biases;
            If set to False, connection will be cut;
        a_ci, a_ig, a_og, a_fg, a_out : function
            Activation function for gate/cell input/cell output;
        b_ci_tickers, b_ig_tickers, b_og_tickers, b_fg_tickers : function or bool
            Initializer function for respective biases applied only during ticker steps;
            If set to False, connection will be cut;
        dtype : torch.dtype
            Datatype of input tensor and weights;
        """
        super(LSTMCell, self).__init__()
        self.n_fwd_features = n_fwd_features
        self.n_lstm = n_lstm
        self.n_rec_features = n_rec_features if n_rec_features is not None else n_lstm
        self.lstm_inlets = ['ci', 'ig', 'og', 'fg']
        self.dtype = dtype
        
        # Get activation functions
        self.a = OrderedDict(zip(self.lstm_inlets + ['out'], [a_ci, a_ig, a_og, a_fg, a_out]))
        
        # Get initializer for tensors
        def try_split_w(w, i):
            try:
                return w[i]
            except TypeError:
                return w
        
        self.w_fwd_init = OrderedDict(zip(self.lstm_inlets, [try_split_w(w, 0) for w in [w_ci, w_ig, w_og, w_fg]]))
        self.w_rec_init = OrderedDict(zip(self.lstm_inlets, [try_split_w(w, 1) for w in [w_ci, w_ig, w_og, w_fg]]))
        self.b_init = OrderedDict(zip(self.lstm_inlets, [b_ci, b_ig, b_og, b_fg]))
        self.b_tickers_init = OrderedDict(zip(self.lstm_inlets,
                                              [b_ci_tickers, b_ig_tickers, b_og_tickers, b_fg_tickers]))
        
        # Create tensor for concatenated weights and biases (only for active connections)
        n_active_fwds = len([1 for i in self.lstm_inlets if self.w_fwd_init[i] is not False])
        self.w_fwd_cat = nn.Parameter(torch.zeros((n_fwd_features, n_lstm * n_active_fwds), dtype=dtype))
        n_active_recs = len([1 for i in self.lstm_inlets if self.w_rec_init[i] is not False])
        self.w_rec_cat = nn.Parameter(torch.zeros((self.n_rec_features, n_lstm * n_active_recs),
                                                  dtype=dtype))
        n_active_b = len([1 for i in self.lstm_inlets if self.b_init[i] is not False])
        self.b_cat = nn.Parameter(torch.zeros((n_lstm * n_active_b,), dtype=dtype))
        n_active_b_tickers = len([1 for i in self.lstm_inlets if self.b_tickers_init[i] is not False])
        self.b_tickers_cat = nn.Parameter(torch.zeros((n_lstm * n_active_b_tickers,), dtype=dtype))
        
        # Create a dict with inlet indices in concatenated weights, inactive connections will have empty arrays
        self.__fwd_cat_inds__ = OrderedDict()
        self.__rec_cat_inds__ = OrderedDict()
        self.__b_cat_inds__ = OrderedDict()
        self.__b_tickers_cat_inds__ = OrderedDict()
        for inlet in self.lstm_inlets:
            if self.w_fwd_init[inlet] is not False:
                self.__fwd_cat_inds__[inlet] = [len(self.__fwd_cat_inds__) * n_lstm,
                                                (len(self.__fwd_cat_inds__) + 1) * n_lstm]
            
            if self.w_rec_init[inlet] is not False:
                self.__rec_cat_inds__[inlet] = [len(self.__rec_cat_inds__) * n_lstm,
                                                (len(self.__rec_cat_inds__) + 1) * n_lstm]
            
            if self.b_init[inlet] is not False:
                self.__b_cat_inds__[inlet] = [len(self.__b_cat_inds__) * n_lstm,
                                              (len(self.__b_cat_inds__) + 1) * n_lstm]
            
            if self.b_tickers_init[inlet] is not False:
                self.__b_tickers_cat_inds__[inlet] = [len(self.__b_tickers_cat_inds__) * n_lstm,
                                                      (len(self.__b_tickers_cat_inds__) + 1) * n_lstm]
        
        # Create views on concatenated weights for all inlets, inactive connections will have empty arrays
        self.w_fwd = OrderedDict(zip(self.lstm_inlets,
                                     [self.w_fwd_cat[:, self.__fwd_cat_inds__[i][0]:self.__fwd_cat_inds__[i][1]]
                                      if i in self.__fwd_cat_inds__ else False
                                      for i in self.lstm_inlets]))
        self.w_rec = OrderedDict(zip(self.lstm_inlets,
                                     [self.w_rec_cat[:, self.__rec_cat_inds__[i][0]:self.__rec_cat_inds__[i][1]]
                                      if i in self.__rec_cat_inds__ else False
                                      for i in self.lstm_inlets]))
        self.b = OrderedDict(zip(self.lstm_inlets,
                                 [self.b_cat[self.__b_cat_inds__[i][0]:self.__b_cat_inds__[i][1]]
                                  if i in self.__b_cat_inds__ else False
                                  for i in self.lstm_inlets]))
        self.b_tickers = OrderedDict(
                zip(self.lstm_inlets,
                    [self.b_tickers_cat[self.__b_tickers_cat_inds__[i][0]:self.__b_tickers_cat_inds__[i][1]]
                     if i in self.__b_tickers_cat_inds__ else False
                     for i in self.lstm_inlets]))
        
        # Activations of LSTM inlets at each timestep for each sample will be stored in a list
        self.lstm_inlets_activations = OrderedDict(zip(self.lstm_inlets, [[], [], [], []]))
        
        # Initialize tensors
        self.__reset_parameters__()
        
        self.__traced_cell__ = self.__cell_python__
        self.__traced_cell_tickersteps__ = self.__cell_tickersteps_python__
    
    def cell(self, fwd_inputs, rec_input, c_old):
        """Compute the next LSTM state given

        fwd_inputs of shape [samples, n_fwd_features],

        rec_input of shape [samples, n_rec_features], where n_rec_features defaults to n_lstm, and

        old cell state c_old of shape [samples, n_lstm].

        Parameters
        -------
        fwd_inputs : torch.Tensor
            Forward inputs of shape [samples, n_fwd_features]
        rec_input : torch.Tensor
            Recurrent inputs of shape [samples, n_rec_features], where n_rec_features defaults to n_lstm
        c_old : torch.Tensor
            Old cell state of shape [samples, n_lstm]

        Returns
        -------
        lstm_states : tuple of torch.Tensor
            6-element tuple of torch.Tensor elements of shape [samples, n_lstm].
            Contains cell state, cell output, cell input, input gate, output gate, and forget gate activations.
        """
        return self.__traced_cell__(fwd_inputs, rec_input, c_old, self.w_fwd_cat, self.w_rec_cat, self.b_cat)
    
    def cell_tickersteps(self, rec_input, c_old):
        """Compute the next LSTM state during ticker (https://arxiv.org/abs/1603.08983) steps given

        rec_input of shape [samples, n_rec_features], where n_rec_features defaults to n_lstm and

        old cell state c_old of shape [samples, n_lstm].

        During tickersteps, forward input is set to 0 and b_ci_tickers, b_ig_tickers, b_og_tickers, b_fg_tickers are
        activated.

        Parameters
        -------
        rec_input : torch.Tensor
            Recurrent inputs of shape [samples, n_rec_features], where n_rec_features defaults to n_lstm
        c_old : torch.Tensor
            Old cell state of shape [samples, n_lstm]

        Returns
        -------
        lstm_states : (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
            6-element tuple of torch.Tensor elements of shape [samples, n_lstm].
            Contains cell state, cell output, cell input, input gate, output gate, and forget gate activations.
        """
        return self.__traced_cell_tickersteps__(rec_input, c_old, self.w_rec_cat, self.b_cat, self.b_tickers_cat)
    
    def __reset_parameters__(self):
        """ Reset trainable LSTM cell parameters to initial values """
        # Apply initializer for W, b, and b_tickersteps
        _ = [self.w_fwd_init[i](self.w_fwd[i]) for i in self.lstm_inlets if self.w_fwd_init[i] is not False]
        _ = [self.w_rec_init[i](self.w_rec[i]) for i in self.lstm_inlets if self.w_rec_init[i] is not False]
        _ = [self.b_init[i](self.b[i]) for i in self.lstm_inlets if self.b_init[i] is not False]
        
        _ = [self.b_tickers_init[i](self.b_tickers[i]) for i in self.lstm_inlets
             if self.b_tickers_init[i] is not False]
    
    def __cell_python__(self, fwd_input, rec_input, c_old, w_fwd_cat, w_rec_cat, b):
        """Template for self.cell() before torch.jit.trace()"""
        # Compute activations for concatenated weights and split them to inlets
        if self.n_fwd_features > 0:
            net_fwds_cat = torch.mm(fwd_input, w_fwd_cat)
        else:
            net_fwds_cat = torch.zeros_like(fwd_input[:, 0:0])
        if self.n_rec_features > 0:
            net_recs_cat = torch.mm(rec_input, w_rec_cat)
        else:
            net_recs_cat = torch.zeros_like(rec_input[:, 0:0])
        
        acts = OrderedDict()
        for inlet in self.lstm_inlets:
            acts[inlet] = False
            if inlet in self.__fwd_cat_inds__:
                if acts[inlet] is False:
                    acts[inlet] = net_fwds_cat[:, self.__fwd_cat_inds__[inlet][0]:self.__fwd_cat_inds__[inlet][1]]
                else:
                    acts[inlet] = acts[inlet] + net_fwds_cat[:, self.__fwd_cat_inds__[inlet][0]:
                                                                self.__fwd_cat_inds__[inlet][1]]
            
            if inlet in self.__rec_cat_inds__:
                if acts[inlet] is False:
                    acts[inlet] = net_recs_cat[:, self.__rec_cat_inds__[inlet][0]:self.__rec_cat_inds__[inlet][1]]
                else:
                    acts[inlet] = acts[inlet] + net_recs_cat[:, self.__rec_cat_inds__[inlet][0]:
                                                                self.__rec_cat_inds__[inlet][1]]
            
            if inlet in self.__b_cat_inds__:
                if acts[inlet] is False:
                    acts[inlet] = b[None, self.__b_cat_inds__[inlet][0]:
                                          self.__b_cat_inds__[inlet][1]].repeat((fwd_input.shape[0], 1))
                else:
                    acts[inlet] = acts[inlet] + b[None, self.__b_cat_inds__[inlet][0]:self.__b_cat_inds__[inlet][1]]
            
            if acts[inlet] is False:
                acts[inlet] = False
            else:
                acts[inlet] = self.a[inlet](acts[inlet])
        
        # Calculate new cell state
        c = c_old
        if acts['fg'] is not False:
            c = c * acts['fg']
        if acts['ci'] is not False and acts['ig'] is not False:
            c = c + acts['ci'] * acts['ig']
        elif acts['ci'] is not False:
            c = c + acts['ci']
        elif acts['ig'] is not False:
            c = c + acts['ig']
        
        # Calculate new LSTM output with new cell state
        h = self.a['out'](c)
        if acts['og'] is not False:
            h = h * acts['og']
        
        return (c, h, *[v if v is not False else torch.zeros((1,), dtype=self.dtype) for v in acts.values()])
    
    def __cell_tickersteps_python__(self, rec_input, c_old, w_rec_cat, b, b_tickers):
        """Template for self.cell_tickersteps() before torch.jit.trace()"""
        # Compute activations for concatenated weights and split them to inlets
        net_recs_cat = torch.mm(rec_input, w_rec_cat)
        
        acts = OrderedDict()
        for inlet in self.lstm_inlets:
            acts[inlet] = False
            
            if inlet in self.__rec_cat_inds__:
                if acts[inlet] is False:
                    acts[inlet] = net_recs_cat[:, self.__rec_cat_inds__[inlet][0]:self.__rec_cat_inds__[inlet][1]]
                else:
                    acts[inlet] = acts[inlet] + net_recs_cat[:, self.__rec_cat_inds__[inlet][0]:
                                                                self.__rec_cat_inds__[inlet][1]]
            
            if inlet in self.__b_cat_inds__:
                if acts[inlet] is False:
                    acts[inlet] = b[None, self.__b_cat_inds__[inlet][0]:
                                          self.__b_cat_inds__[inlet][1]].repeat((c_old.shape[0], 1))
                else:
                    acts[inlet] = acts[inlet] + b[None, self.__b_cat_inds__[inlet][0]:self.__b_cat_inds__[inlet][1]]
            
            if inlet in self.__b_tickers_cat_inds__:
                if acts[inlet] is False:
                    acts[inlet] = b_tickers[None,
                                  self.__b_tickers_cat_inds__[inlet][0]:
                                  self.__b_tickers_cat_inds__[inlet][1]].repeat((c_old.shape[0], 1))
                else:
                    acts[inlet] = acts[inlet] + b_tickers[None, self.__b_tickers_cat_inds__[inlet][0]:
                                                                self.__b_tickers_cat_inds__[inlet][1]]
            
            if acts[inlet] is False:
                acts[inlet] = False
            else:
                acts[inlet] = self.a[inlet](acts[inlet])
        
        # Calculate new cell state
        c = c_old
        if acts['fg'] is not False:
            c = c * acts['fg']
        if acts['ci'] is not False and acts['ig'] is not False:
            c = c + acts['ci'] * acts['ig']
        elif acts['ci'] is not False:
            c = c + acts['ci']
        elif acts['ig'] is not False:
            c = c + acts['ig']
        
        # Calculate new LSTM output with new cell state
        h = self.a['out'](c)
        if acts['og'] is not False:
            h = h * acts['og']
        
        return (c, h, *[v if v is not False else torch.zeros((1,), dtype=self.dtype) for v in acts.values()])


class LSTMLayer(nn.Module):
    def __init__(self, in_features, out_features,
                 w_ci=nn.init.xavier_normal_, w_ig=nn.init.xavier_normal_, w_og=nn.init.xavier_normal_, w_fg=False,
                 b_ci=nn.init.normal_, b_ig=nn.init.normal_, b_og=nn.init.normal_, b_fg=False,
                 a_ci=torch.tanh, a_ig=torch.sigmoid, a_og=torch.sigmoid, a_fg=torch.sigmoid, a_out=torch.tanh,
                 c_init=lambda t: nn.init.constant_(t, val=0).detach_(),
                 h_init=lambda t: nn.init.constant_(t, val=0).detach_(),
                 b_ci_tickers=nn.init.normal_, b_ig_tickers=nn.init.normal_, b_og_tickers=nn.init.normal_,
                 b_fg_tickers=False, n_tickersteps=0, inputformat='NLC', dtype: torch.dtype = torch.float32,
                 optimize_cell=False):
        """Flexible LSTM layer supporting
          - individual initialization of forward and recurrent LSTM weights and biases,
          - disabling connections to gates or cell input (forward and/or recurrent) via initializers,
          - individual activation functions for LSTM gates, cell input, and cell output,
          - access to cell state, cell output, and gate activations,
          - initialization of (optionally trainable) initial cell state and cell output,
          - modification of recurrent inputs,
          - PyTorch TorchScript optimization,
          - ponder-/tickersteps with dedicated bias weights (https://arxiv.org/abs/1603.08983),
          - return of LSTM activation at every or only last sequence position,
          - inputformat 'NLC' [samples, sequence positions, in_features] or 'NCL',
          - plotting of the LSTM internals.

        Provides functions
          - forward() for processing a sequence with the LSTM,
          - plot_internals() for plotting the LSTM gate-, cell input-, and cell output activations
          - get_weights() and get_biases() for retrieving weight and bias values.

        Abbreviations:
         - ci = cell input,
         - ig = input gate,
         - og = output gate,
         - fg = forget gate weights,
         - out = cell output.

        Default: Forget gate disabled, other connections and biases enabled,
        tanh cell input and cell output activation function,
        sigmoid gate activation functions, nn.init.xavier_normal_ initializer for weights and
        nn.init.normal_ for biases.

        Parameters
        -------
        in_features : int
            Number of input features
        out_features : int
            Number of output features (=number of LSTM blocks)
        w_ci, w_ig, w_og, w_fg : function or list of function or tuple of function or bool
            Initializer function(s) for respective weights;
            If 2-element list: Interpreted as [w_fwd, w_rec] to define different weight initializations for forward and
            recurrent connections respectively;
            If single element: forward and recurrent connections will use the same initializer function;
            If False: connection will be cut (single element of 2-element tuple may be set to False as well);
            Shape of weights will be w_fwd: [n_fwd_features, n_lstm], w_rec: [n_rec_features, n_lstm];
        b_ci, b_ig, b_og, b_fg : function or bool
            Initializer function for respective biases;
            If set to False, connection will be cut;
        a_ci, a_ig, a_og, a_fg, a_out : function
            Activation function for gate/cell input/cell output;
        b_ci_tickers, b_ig_tickers, b_og_tickers, b_fg_tickers : function or bool
            Initializer function for respective biases applied only during ticker steps;
            If set to False, connection will be cut;
        c_init : function
            Initializer function for cell states; Default: Zero and not trainable;
        h_init : function
            Initializer function for cell states; Default: Zero and not trainable;
        n_tickersteps : int
            Number of ponder-/tickersteps to add at end of sequence;
            n_tickersteps sequence positions without forward input will be added at the end of the sequence;
            Only during tickersteps, additional bias units (b_ci_tickers, b_ig_tickers, b_og_tickers, b_fg_tickers)
            will be added to the LSTM input;
        inputformat : str
            Input tensor format;
            'NCL' -> (batchsize, channels, seq.length);
            'NLC' -> (batchsize, seq.length, channels);
        dtype : torch.dtype
            Datatype of input tensor, states, and weights;
        optimize_cell : bool
            If True, use faster LSTMCellTorchScript with TorchScrip optimization (recommended only with PyTorch 1.1.0+)
        """
        super(LSTMLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lstm_inlets = ['ci', 'ig', 'og', 'fg']
        self.n_tickersteps = n_tickersteps
        self.inputformat = inputformat
        
        # Check for valid input format
        if self.inputformat not in ['LNC', 'NLC', 'NCL']:
            raise ValueError("Input format {} not supported".format(self.inputformat))
        
        # Get LSTM cell
        if optimize_cell:
            self.lstm_cell = LSTMCellTorchScript(n_fwd_features=in_features, n_lstm=out_features,
                                                 w_ci=w_ci, w_ig=w_ig, w_og=w_og, w_fg=w_fg,
                                                 b_ci=b_ci, b_ig=b_ig, b_og=b_og, b_fg=b_fg,
                                                 a_ci=a_ci, a_ig=a_ig, a_og=a_og, a_fg=a_fg, a_out=a_out,
                                                 b_ci_tickers=b_ci_tickers, b_ig_tickers=b_ig_tickers,
                                                 b_og_tickers=b_og_tickers, b_fg_tickers=b_fg_tickers, dtype=dtype)
        else:
            self.lstm_cell = LSTMCell(n_fwd_features=in_features, n_lstm=out_features,
                                      w_ci=w_ci, w_ig=w_ig, w_og=w_og, w_fg=w_fg,
                                      b_ci=b_ci, b_ig=b_ig, b_og=b_og, b_fg=b_fg,
                                      a_ci=a_ci, a_ig=a_ig, a_og=a_og, a_fg=a_fg, a_out=a_out,
                                      b_ci_tickers=b_ci_tickers, b_ig_tickers=b_ig_tickers,
                                      b_og_tickers=b_og_tickers, b_fg_tickers=b_fg_tickers, dtype=dtype)
        
        # Get initializer for tensors
        self.c_init = c_init
        self.h_init = h_init
        self.c_first = nn.Parameter(torch.zeros((out_features,), dtype=dtype))
        self.h_first = nn.Parameter(torch.zeros((out_features,), dtype=dtype))
        
        # Cell states and LSTM outputs at each timestep for each sample will be stored in a list
        self.c = []
        self.h = []
        
        # Activations of LSTM inlets at each timestep for each sample will be stored in a list
        self.lstm_inlets_activations = OrderedDict(zip(self.lstm_inlets, [[], [], [], []]))
        
        # Initialize tensors
        self.__reset_parameters__()
        
        # This will optionally hold the true sequence lengths (before padding) for plotting
        self.true_seq_lens = None
        
        # This will hold the input sequences for plotting
        self.x = None
        
        # Optimize function for input splitting for chosen input format
        if self.inputformat == 'LNC':
            def unbind_inputs(fwd_inputs):
                fwd_inputs = fwd_inputs.unbind(dim=0)
                return fwd_inputs
        elif self.inputformat == 'NLC':
            def unbind_inputs(fwd_inputs):
                fwd_inputs = fwd_inputs.unbind(dim=1)
                return fwd_inputs
        else:
            def unbind_inputs(fwd_inputs):
                fwd_inputs = fwd_inputs.unbind(dim=2)
                return fwd_inputs
        self.unbind_inputs = unbind_inputs
    
    def __reset_parameters__(self):
        """Reset trainable parameters (initial cell state and initial cell output)"""
        # Initialize cell state and LSTM output at timestep -1
        self.c_init(self.c_first)
        self.h_init(self.h_first)
    
    def __reset_lstm_internals__(self, n_samples):
        """Reset LSTM state and start new sequence (=reset cell state, cell output, and stored activations)"""
        # Cell states and LSTM outputs at each timestep for each sample will be stored in a list
        self.c = [self.c_first.repeat((n_samples, 1))]
        self.h = [self.h_first.repeat((n_samples, 1))]
        self.x = []
        
        # Activations of LSTM inlets at each timestep for each sample will be stored in a list
        self.lstm_inlets_activations = OrderedDict(zip(self.lstm_inlets, [[], [], [], []]))
    
    def __apply_cell__(self, fwd_inputs, h_old, c_old):
        """Apply LSTM cell to a sequence"""
        c = torch.jit.annotate(List[torch.Tensor], [])
        h = torch.jit.annotate(List[torch.Tensor], [])
        ci = torch.jit.annotate(List[torch.Tensor], [])
        ig = torch.jit.annotate(List[torch.Tensor], [])
        og = torch.jit.annotate(List[torch.Tensor], [])
        fg = torch.jit.annotate(List[torch.Tensor], [])
        
        for fwd_input in self.unbind_inputs(fwd_inputs):
            # Calculate activations for LSTM inlets and append them to self.lstm_inlets_activations
            cell_rets = self.lstm_cell.cell(fwd_input, h_old, c_old)
            c_old = cell_rets[0]
            h_old = cell_rets[1]
            c.append(c_old)
            h.append(h_old)
            ci.append(cell_rets[2])
            ig.append(cell_rets[3])
            og.append(cell_rets[4])
            fg.append(cell_rets[5])
        
        return c, h, ci, ig, og, fg
    
    def __apply_cell_tickersteps__(self, tickersteps, h_old, c_old):
        """Apply LSTM cell to a sequence during tickersteps"""
        c = torch.jit.annotate(List[torch.Tensor], [])
        h = torch.jit.annotate(List[torch.Tensor], [])
        ci = torch.jit.annotate(List[torch.Tensor], [])
        ig = torch.jit.annotate(List[torch.Tensor], [])
        og = torch.jit.annotate(List[torch.Tensor], [])
        fg = torch.jit.annotate(List[torch.Tensor], [])
        
        for _ in torch.unbind(tickersteps, dim=0):
            # Calculate activations for LSTM inlets and append them to self.lstm_inlets_activations
            cell_rets = self.lstm_cell.cell_tickersteps(h_old, c_old)
            c_old = cell_rets[0]
            h_old = cell_rets[1]
            c.append(c_old)
            h.append(h_old)
            ci.append(cell_rets[2])
            ig.append(cell_rets[3])
            og.append(cell_rets[4])
            fg.append(cell_rets[5])
        
        return c, h, ci, ig, og, fg
    
    def forward(self, x, return_all_seq_pos: bool = False, true_seq_lens=None, reset_state=True):
        """ Process sequence with LSTM layer.

        Process an input sequence x with the LSTM layer.
        If return_all_seq_pos, activations at each sequence position will be returned,
        otherwise only the activation at the last sequence position will be returned.

        For padded sequences, true_seq_lens can be used to feed the true sequence lengths.

        If using tickersteps and true_seq_lens, the tickerstep activations are appended at the end of the padded
        sequences. This also holds for the LSTM internals and the cellstate. However, the tickerstep activations are
        correctly computed beginning with the last activation at the true sequence length.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape [samples, sequence positions, in_features]
            or [samples, in_features, sequence positions], depending on chosen inputformat.
        return_all_seq_pos : bool
            True: Return output for all sequence positions (continuous prediction);
            False: Only return output at last sequence position (single target prediction);
        true_seq_lens: list of int or None
            If not None: True sequence lengths of padded sequences in x. len(true_seq_lens) has to be len(x).
        reset_state : bool
            Reset LSTM cell state and cell output at beginning of sequence.
            If False, the sequence will be continued from the last processed sequence.

        Returns
        --------
        (torch.Tensor, OrderedDict)
            h_out : torch.Tensor
                If return_all_seq_pos: Output of LSTM for all (possibly padded) sequence positions in shape NLC or
                NCL.
                Tickerstep activations are appended at the end of the padded sequences if true_seq_lens is provided.
                If not return_all_seq_pos: Output of LSTM at last sequence position (after tickersteps) in shape NC.
            lstm_inlets_activations: OrderedDict of list of torch.Tensor
                OrderedDict containing lists of internal LSTM states.
                Each list contains the activations for every sequence position, with every list element of shape NC.
                OrderedDict keys are ['h', 'c', 'ci', 'ig', 'og', 'fg'], representing cell output, cell state,
                 cell input, input gate, output gate, and forget gate.
        """
        self.true_seq_lens = true_seq_lens
        
        if self.inputformat == 'LNC':
            n_seqpos, n_samples, n_features = x.shape
        elif self.inputformat == 'NLC':
            n_samples, n_seqpos, n_features = x.shape
        elif self.inputformat == 'NCL':
            n_samples, n_features, n_seqpos = x.shape
        else:
            raise ValueError("Input format {} not supported".format(self.inputformat))
        
        if reset_state:
            self.__reset_lstm_internals__(n_samples=n_samples)
        
        self.x.append(x)
        
        # Calculate activations for LSTM inlets and append them to self.lstm_inlets_activations
        cell_rets = self.__apply_cell__(x, self.h[-1], self.c[-1])
        self.c += cell_rets[0]
        self.h += cell_rets[1]
        self.lstm_inlets_activations['ci'] += cell_rets[2]
        self.lstm_inlets_activations['ig'] += cell_rets[3]
        self.lstm_inlets_activations['og'] += cell_rets[4]
        self.lstm_inlets_activations['fg'] += cell_rets[5]
        
        #
        # Get LSTM output at sequence end, considering true_seq_len
        #
        if true_seq_lens is None:
            last_h = self.h[-1]
        else:
            last_h = torch.stack([self.h[tsl][sample_i] for sample_i, tsl in enumerate(true_seq_lens)], dim=0)
        
        #
        # Handle tickersteps
        #
        if self.n_tickersteps:
            #
            # Set input for tickersteps and consider true_seq_len
            #
            if true_seq_lens is None:
                ticker_h = last_h
                ticker_c = self.c[-1]
            else:
                ticker_h = last_h
                ticker_c = torch.stack([self.c[tsl][sample_i] for sample_i, tsl in enumerate(true_seq_lens)], dim=0)
            
            #
            # Calculate tickerstep activations
            #
            cell_rets = self.__apply_cell_tickersteps__(torch.arange(self.n_tickersteps), ticker_h, ticker_c)
            
            # Append tickerstep activations to end of (possibly padded) sequences
            self.c += cell_rets[0]
            self.h += cell_rets[1]
            self.lstm_inlets_activations['ci'] += cell_rets[2]
            self.lstm_inlets_activations['ig'] += cell_rets[3]
            self.lstm_inlets_activations['og'] += cell_rets[4]
            self.lstm_inlets_activations['fg'] += cell_rets[5]
            
            # Set output after tickersteps as new final output
            last_h = self.h[-1]
        
        #
        # Finalize output of LSTM
        #
        if return_all_seq_pos:
            # Output is LSTM output at each position
            if self.inputformat == 'LNC':
                h_out = torch.stack(self.h[1:], 0)
            elif self.inputformat == 'NLC':
                h_out = torch.stack(self.h[1:], 1)
            elif self.inputformat == 'NCL':
                h_out = torch.stack(self.h[1:], 2)
            else:
                raise ValueError("Input format {} not supported".format(self.inputformat))
        else:
            # Output is LSTM output at last (existing) position
            h_out = last_h
        
        return h_out, OrderedDict([('h', self.h), ('c', self.c)] + list(self.lstm_inlets_activations.items()))
    
    def get_weights(self):
        """Return dictionaries for w_fwd and w_rec; These are views on the actual concatenated parameters;"""
        return self.lstm_cell.w_fwd, self.lstm_cell.w_rec
    
    def get_biases(self):
        """Return dictionaries for with biases and """
        return self.lstm_cell.b
    
    def __tensor_to_numpy__(self, t):
        """Try to convert a tensor or numpy.ndarray t to a numpy.ndarray"""
        try:
            t = t.numpy()
        except TypeError:
            t = t.cpu().numpy()
        except AttributeError:
            t = np.asarray(t)
        except RuntimeError:
            t = t.clone().data.cpu().numpy()
        return t
    
    def plot_internals(self, mb_index: int = 0, filename: str = None, show_plot: bool = False, fdict: dict = None,
                       verbose=True):
        """Plot LSTM internal states and LSTM input for sample mb_index.

        Plots LSTM input, LSTM output ('h'), LSTM cell state ('c'), LSTM gate activations ('ig', 'og', 'fg'),
        and LSTM cell input ('ci') in separated axes.
        Each plot contains activations for all LSTM blocks (line colors identify individual LSTM units) for all
        sequence positions.
        The current internal states of the LSTM will be plotted, that is the last processed sequence at sample index
        mb_index.

        Parameters
        ----------
        mb_index: int
            Index of sample to plot (index in minibatch)
        filename: str
            If specified, plot will be saved at location specified by filename. Necessary paths will be created
            automatically.
        show_plot: bool
            True: Display plot
            False: Do not display plot
        fdict: dict
            kwargs passed to matplotlib.pyplot.subplots() function
        verbose: bool
            Verbose printing of plotting runtime
        """
        if filename is not None:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        if verbose:
            print(f"  Plotting LSTM states...", end='')
            start_time = time.time()
        
        #
        # Get sample activations at mb_index
        #
        if self.inputformat == 'LNC':
            lstm_internals_copy = OrderedDict([(k, self.__tensor_to_numpy__(torch.stack([t[mb_index] for t in v], 0)))
                                               for k, v in self.lstm_inlets_activations.items() if v[0] is not 1])
            lstm_internals_copy['c'] = self.__tensor_to_numpy__(torch.stack([t[mb_index] for t in self.c], 0))
            lstm_internals_copy['h'] = self.__tensor_to_numpy__(torch.stack([t[mb_index] for t in self.h], 0))
            lstm_internals_copy['input'] = self.__tensor_to_numpy__(torch.cat([t[:, mb_index] for t in self.x], 0))
        elif self.inputformat == 'NLC':
            lstm_internals_copy = OrderedDict([(k, self.__tensor_to_numpy__(torch.stack([t[mb_index] for t in v], 0)))
                                               for k, v in self.lstm_inlets_activations.items() if v[0] is not 1])
            lstm_internals_copy['c'] = self.__tensor_to_numpy__(torch.stack([t[mb_index] for t in self.c], 0))
            lstm_internals_copy['h'] = self.__tensor_to_numpy__(torch.stack([t[mb_index] for t in self.h], 0))
            lstm_internals_copy['input'] = self.__tensor_to_numpy__(torch.cat([t[mb_index] for t in self.x], 0))
        elif self.inputformat == 'NCL':
            lstm_internals_copy = OrderedDict([(k, self.__tensor_to_numpy__(torch.stack([t[mb_index] for t in v], 0)))
                                               for k, v in self.lstm_inlets_activations.items() if v[0] is not 1])
            lstm_internals_copy['c'] = self.__tensor_to_numpy__(torch.stack([t[mb_index] for t in self.c], 0))
            lstm_internals_copy['h'] = self.__tensor_to_numpy__(torch.stack([t[mb_index] for t in self.h], 0))
            lstm_internals_copy['input'] = self.__tensor_to_numpy__(torch.cat([t[mb_index] for t in self.x], 1))
            lstm_internals_copy['input'] = lstm_internals_copy['input'].permute(1, 0)
        else:
            raise ValueError("Input format {} not supported".format(self.inputformat))
        
        #
        # Remove activations between last true sequence position and start_time of tickersteps for padded sequences
        #
        if self.true_seq_lens is not None:
            true_seq_len = self.true_seq_lens[mb_index]
            for key in lstm_internals_copy.keys():
                if key == 'c' or key == 'h':
                    tsl = true_seq_len + 1
                else:
                    tsl = true_seq_len
                if key == 'input' or not self.n_tickersteps:
                    lstm_internals_copy[key] = lstm_internals_copy[key][:tsl]
                else:
                    lstm_internals_copy[key] = np.concatenate((lstm_internals_copy[key][:tsl],
                                                               lstm_internals_copy[key][-self.n_tickersteps:]), axis=0)
        
        plot_labels = [a for z in zip_longest(['input', 'h', 'c'], self.lstm_inlets) for a in z]
        max_len = max([v.shape[0] for v in lstm_internals_copy.values()])
        
        #
        # Do plotting: 1 axis per LSTM internal
        #
        if fdict is None:
            fdict = dict()
        # plt.rcParams.update({'font.size': 5})
        fig, axes = plt.subplots(4, 2, **fdict)
        axes = [a for aa in axes for a in aa]
        plt.tight_layout()
        for i, ax in enumerate(axes):
            try:
                label = plot_labels[i]
            except IndexError:
                ax.axis('off')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                continue
            ax.set_title(label)
            if label not in lstm_internals_copy.keys():
                continue
            
            if label == 'c' or label == 'h':
                _ = ax.plot(np.arange(len(lstm_internals_copy[label])) - 1, lstm_internals_copy[label], label=label)
            else:
                _ = ax.plot(lstm_internals_copy[label], label=label)
            ax.set_xlim(left=-1, right=max_len)
            ax.grid(True)
        
        if filename is not None:
            fig.savefig(filename)
        
        if show_plot:
            fig.show()
        else:
            plt.close(fig)
            del fig
        
        if verbose:
            print(f" done! ({time.time() - start_time:8.7f}sec)")
