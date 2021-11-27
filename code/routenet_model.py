"""
   Copyright 2021 Universitat Politècnica de Catalunya & AGH University of Science and Technology
                                        BSD 3-Clause License
   Redistribution and use in source and binary forms, with or without modification, are permitted
   provided that the following conditions are met:
    1. Redistributions of source code must retain the above copyright notice, this list of conditions
       and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice, this list of
       conditions and the following disclaimer in the documentation and/or other materials provided
       with the distribution.
    3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse
       or promote products derived from this software without specific prior written permission.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
    WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
    TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
"""
import tensorflow as tf


class RouteNetModel(tf.keras.Model):
    """ Init method for the custom model.
    Args:
        config (dict): Python dictionary containing the diferent configurations
                       and hyperparameters.
        output_units (int): Output units for the last readout's layer.
    Attributes:
        config (dict): Python dictionary containing the diferent configurations
                       and hyperparameters.
        link_update (GRUCell): Link GRU Cell used in the Message Passing step.
        path_update (GRUCell): Path GRU Cell used in the Message Passing step.
        readout (Keras Model): Readout Neural Network. It expects as input the
                               path states and outputs the per-path delay.
    """

    def __init__(self, config, output_units=1):
        super(RouteNetModel, self).__init__()

        # Configuration dictionary. It contains the needed Hyperparameters for the model.
        # All the Hyperparameters can be found in the config.ini file
        self.config = config

        # GRU Cells used in the Message Passing step
        self.link_update = tf.keras.layers.GRUCell(int(self.config['HYPERPARAMETERS']['link_state_dim']))
        self.path_update = tf.keras.layers.GRUCell(int(self.config['HYPERPARAMETERS']['path_state_dim']))

        # Used to mask the input sequence to skip timesteps
        self.masking = tf.keras.layers.Masking()

        # Embedding to compute the initial link hidden state
        self.link_embedding = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=2),
            tf.keras.layers.Dense(int(int(self.config['HYPERPARAMETERS']['link_state_dim']) / 2),
                                  activation=tf.nn.relu),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['link_state_dim']), activation=tf.nn.relu)
        ])

        # Embedding to compute the initial path hidden state
        self.path_embedding = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=4),
            tf.keras.layers.Dense(int(int(self.config['HYPERPARAMETERS']['path_state_dim']) / 2),
                                  activation=tf.nn.relu),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['path_state_dim']), activation=tf.nn.relu)
        ])

        # Readout Neural Network. It expects as input the path states and outputs the per-path delay
        self.readout = tf.keras.Sequential([
            #tf.keras.layers.InputLayer(input_shape=int(self.config['HYPERPARAMETERS']['path_state_dim'])), #for prediting per-path delay
            tf.keras.layers.InputLayer(input_shape=int(self.config['HYPERPARAMETERS']['link_state_dim'])), #for prediting per-link Occupancy
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['readout_units']),
                                  activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['readout_units']),
                                  activation=tf.nn.relu),
            #tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(output_units,
                                  kernel_regularizer=tf.keras.regularizers.l2(
                                      float(self.config['HYPERPARAMETERS']['l2_2'])))
        ])

    @tf.function
    def call(self, inputs):
        """This function is execution each time the model is called
        Args:
            inputs (dict): Features used to make the predictions.
        Returns:
            tensor: A tensor containing the per-path delay.
        """

        traffic = tf.cast(tf.expand_dims(tf.squeeze(inputs['traffic']), axis=1), dtype=tf.float32)
        capacity = tf.cast(tf.expand_dims(tf.squeeze(inputs['capacity']), axis=1), dtype=tf.float32)
        #occupancy = tf.expand_dims(tf.squeeze(inputs['occupancy']), axis=1)
        queueSizes = tf.cast(tf.expand_dims(tf.squeeze(inputs['queueSizes']), axis=1), dtype=tf.float32)
        packets = tf.cast(tf.expand_dims(tf.squeeze(inputs['packets']), axis=1), dtype=tf.float32)
        EqLambda = tf.cast(tf.expand_dims(tf.squeeze(inputs['EqLambda']), axis=1), dtype=tf.float32)
        AvgPktSize = tf.cast(tf.expand_dims(tf.squeeze(inputs['AvgPktSize']), axis=1), dtype=tf.float32)
        link_to_path = tf.cast(tf.squeeze(inputs['link_to_path']), dtype=tf.int32)
        path_to_link = tf.cast(tf.squeeze(inputs['path_to_link']), dtype=tf.int32)
        path_ids = tf.cast(tf.squeeze(inputs['path_ids']), dtype=tf.int32)
        sequence_path = tf.cast(tf.squeeze(inputs['sequence_path']), dtype=tf.int32)
        sequence_links = tf.cast(tf.squeeze(inputs['sequence_links']), dtype=tf.int32)
        n_links = tf.cast(inputs['n_links'], dtype=tf.int32)
        n_paths = tf.cast(inputs['n_paths'], dtype=tf.int32)

        # Initialize the initial hidden state for links
        shape = tf.stack([
            n_links,
            int(self.config['HYPERPARAMETERS']['link_state_dim']) - 2
        ], axis=0)
        
        link_state = tf.concat([
            capacity,
            queueSizes,
            #occupancy,
            tf.zeros(shape)
        ], axis=1)

        #link_state = self.link_embedding(link_state)

        # Initialize the initial hidden state for paths
        shape = tf.stack([
            n_paths,
            int(self.config['HYPERPARAMETERS']['path_state_dim']) - 4
        ], axis=0)
        
        path_state = tf.concat([
            traffic,
            packets,
            EqLambda,
            AvgPktSize,
            tf.zeros(shape)
        ], axis=1)

        #path_state = self.path_embedding(path_state)

        for _ in range(int(self.config['HYPERPARAMETERS']['t'])):
            # The following lines generate a tensor of dimensions [n_paths, max_len_path, dimension_link] with all 0
            # but the link hidden states
            link_gather = tf.gather(link_state, link_to_path)

            ids = tf.stack([path_ids, sequence_path], axis=1)
            max_len = tf.reduce_max(sequence_path) + 1
            shape = tf.stack([
                n_paths,
                max_len,
                int(self.config['HYPERPARAMETERS']['link_state_dim'])])

            # Generate the aforementioned tensor [n_paths, max_len_path, dimension_link]
            link_inputs = tf.scatter_nd(ids, link_gather, shape)

            path_update_rnn = tf.keras.layers.RNN(self.path_update,
                                                  return_sequences=True,
                                                  return_state=True)

            path_state_sequence, path_state = path_update_rnn(inputs=self.masking(link_inputs),
                                                              initial_state=path_state)

            # For every link, gather and sum the sequence of hidden states of the paths that contain it
            path_gather = tf.gather(path_state, path_to_link)
            #path_gather = tf.gather_nd(path_state_sequence, ids)
            path_sum = tf.math.unsorted_segment_sum(path_gather, sequence_links, n_links)

            # Second message passing: update the link_state
            # The ensure shape is needed for Graph_compatibility
            path_sum = tf.ensure_shape(path_sum, [None, int(self.config['HYPERPARAMETERS']['path_state_dim'])])
            link_state, _ = self.link_update(path_sum, [link_state])

        # Call the readout ANN and return its predictions
        #r = self.readout(path_state) #for prediting per-path delay
        r = self.readout(link_state) #for prediting per-link Occupancy
        
        #for prediting per-link Occupancy
        return r 
        
'''
        #for prediting per-path delay
        predi_occupancy = r
        predi_occupancy_gather = tf.gather(predi_occupancy, link_to_path)
        queueSizes_gather = tf.gather(queueSizes, link_to_path)
        AvgPktSize_gather = tf.gather(AvgPktSize, path_ids)
        capacity_gather = tf.gather(capacity, link_to_path)
        delay_per_link = predi_occupancy_gather * queueSizes_gather * AvgPktSize_gather / capacity_gather
        predi_delay = tf.math.unsorted_segment_sum(delay_per_link, path_ids, n_paths)
        
        return predi_delay
'''
