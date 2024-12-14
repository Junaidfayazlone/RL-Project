import numpy as np
import tensorflow as tf

class Agent(object):

    def __init__(self, params):

        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']
        self.ePAD = tf.constant(params['entity_vocab']['PAD'], dtype=tf.int32)
        self.rPAD = tf.constant(params['relation_vocab']['PAD'], dtype=tf.int32)
        
        if params['use_entity_embeddings']:
            self.entity_initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        else:
            self.entity_initializer = tf.zeros_initializer()
        
        self.train_entities = params['train_entity_embeddings']
        self.train_relations = params['train_relation_embeddings']

        self.num_rollouts = params['num_rollouts']
        self.test_rollouts = params['test_rollouts']
        self.LSTM_Layers = params['LSTM_layers']
        self.batch_size = params['batch_size'] * params['num_rollouts']
        self.dummy_start_label = tf.constant(
            np.ones(self.batch_size, dtype='int64') * params['relation_vocab']['DUMMY_START_RELATION'])

        self.entity_embedding_size = self.embedding_size
        self.use_entity_embeddings = params['use_entity_embeddings']
        if self.use_entity_embeddings:
            self.m = 4
        else:
            self.m = 2

        # Relation embedding lookup table
        self.action_embedding_placeholder = tf.Variable(
            initial_value=tf.zeros([self.action_vocab_size, 2 * self.embedding_size]), trainable=False
        )

        self.relation_lookup_table = tf.Variable(
            initial_value=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")(
                shape=[self.action_vocab_size, 2 * self.embedding_size]),
            trainable=self.train_relations
        )

        # Entity embedding lookup table
        self.entity_embedding_placeholder = tf.Variable(
            initial_value=tf.zeros([self.entity_vocab_size, 2 * self.embedding_size]), trainable=False
        )

        self.entity_lookup_table = tf.Variable(
            initial_value=self.entity_initializer(
                shape=[self.entity_vocab_size, 2 * self.entity_embedding_size]),
            trainable=self.train_entities
        )

        # Policy step LSTM
        cells = [
            tf.keras.layers.LSTMCell(self.m * self.hidden_size)
            for _ in range(self.LSTM_Layers)
        ]
        self.policy_step = tf.keras.layers.StackedRNNCells(cells)

    def get_mem_shape(self):
        return (self.LSTM_Layers, 2, None, self.m * self.hidden_size)

    def policy_MLP(self, state):
        hidden = tf.keras.layers.Dense(4 * self.hidden_size, activation='relu')(state)
        output = tf.keras.layers.Dense(self.m * self.embedding_size, activation='relu')(hidden)
        return output

    def action_encoder(self, next_relations, next_entities):
        relation_embedding = tf.nn.embedding_lookup(self.relation_lookup_table, next_relations)
        entity_embedding = tf.nn.embedding_lookup(self.entity_lookup_table, next_entities)
        if self.use_entity_embeddings:
            action_embedding = tf.concat([relation_embedding, entity_embedding], axis=-1)
        else:
            action_embedding = relation_embedding
        return action_embedding

    def step(self, next_relations, next_entities, prev_state, prev_relation, query_embedding, current_entities,
             label_action, range_arr, first_step_of_test):

        prev_action_embedding = self.action_encoder(prev_relation, current_entities)
        output, new_state = self.policy_step(prev_action_embedding, prev_state)  # RNN step

        prev_entity = tf.nn.embedding_lookup(self.entity_lookup_table, current_entities)
        if self.use_entity_embeddings:
            state = tf.concat([output, prev_entity], axis=-1)
        else:
            state = output

        candidate_action_embeddings = self.action_encoder(next_relations, next_entities)
        state_query_concat = tf.concat([state, query_embedding], axis=-1)

        # MLP for policy
        output = self.policy_MLP(state_query_concat)
        output_expanded = tf.expand_dims(output, axis=1)
        prelim_scores = tf.reduce_sum(tf.multiply(candidate_action_embeddings, output_expanded), axis=2)

        # Masking PAD actions
        mask = tf.equal(next_relations, self.rPAD)
        dummy_scores = tf.ones_like(prelim_scores) * -99999.0
        scores = tf.where(mask, dummy_scores, prelim_scores)

        # Sample action
        action = tf.random.categorical(logits=scores, num_samples=1, dtype=tf.int32)

        # Loss computation
        label_action = tf.squeeze(action, axis=1)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=label_action)

        # Map back to true id
        action_idx = tf.squeeze(action)
        chosen_relation = tf.gather_nd(next_relations, tf.stack([range_arr, action_idx], axis=1))

        return loss, new_state, tf.nn.log_softmax(scores), action_idx, chosen_relation

    def __call__(self, candidate_relation_sequence, candidate_entity_sequence, current_entities,
                 path_label, query_relation, range_arr, first_step_of_test, T=3, entity_sequence=0):
        self.baseline_inputs = []

        query_embedding = tf.nn.embedding_lookup(self.relation_lookup_table, query_relation)
        state = self.policy_step.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
        prev_relation = self.dummy_start_label

        all_loss = []
        all_logits = []
        action_idx = []

        for t in range(T):
            next_possible_relations = candidate_relation_sequence[t]
            next_possible_entities = candidate_entity_sequence[t]
            current_entities_t = current_entities[t]
            path_label_t = path_label[t]

            loss, state, logits, idx, chosen_relation = self.step(
                next_possible_relations, next_possible_entities, state, prev_relation, query_embedding,
                current_entities_t, label_action=path_label_t, range_arr=range_arr,
                first_step_of_test=first_step_of_test
            )

            all_loss.append(loss)
            all_logits.append(logits)
            action_idx.append(idx)
            prev_relation = chosen_relation

        return all_loss, all_logits, action_idx
