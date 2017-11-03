from . import utils
from dateutil.relativedelta import relativedelta
import warnings
import logging
import itertools

class Timechop(object):
    def __init__(
        self,
        feature_start_time, 
        feature_end_time,
        label_start_time,
        label_end_time,
        model_update_frequency,
        training_data_frequencies,
        max_training_histories,
        training_prediction_spans,
        test_data_frequencies,
        test_spans,
        test_prediction_spans
    ):
        self.feature_start_time = feature_start_time # earliest time included in any feature
        self.feature_end_time = feature_end_time # all data included in features are < this time
        if self.feature_start_time > self.feature_end_time:
            raise ValueError('Feature start time after feature end time.')
        self.label_start_time = label_start_time # earliest time included in any label
        self.label_end_time = label_end_time # all data in any label are < this time
        if self.label_start_time > self.label_end_time:
            raise ValueError('Label start time after label end time.')
        self.model_update_frequency = utils.convert_str_to_relativedelta(model_update_frequency) # how frequently to retrain models
        self.training_data_frequencies = utils.convert_to_list(training_data_frequencies) # time between rows for same entity in train matrix
        self.test_data_frequencies = utils.convert_to_list(test_data_frequencies) # time between rows for same entity in test matrix
        self.max_training_histories = utils.convert_to_list(max_training_histories) # how much history for each entity to train on
        self.test_spans = utils.convert_to_list(test_spans) # how long into the future to make predictions for each entity
        self.training_prediction_spans = utils.convert_to_list(training_prediction_spans) # how much time is included in a label in the train matrix
        self.test_prediction_spans = utils.convert_to_list(test_prediction_spans) # how much time is included in a label in the test matrix

    def chop_time(self):
        """ Given the attributes of the object, define all train/test splits
        for all combinations of the temporal parameters.

        :return: a list of dictionaries defining train/test splits
        :rtype: list
        """
        matrix_set_definitions = []
        for training_prediction_span, test_prediction_span, test_span in itertools.product(
                self.training_prediction_spans,
                self.test_prediction_spans,
                self.test_spans
            ):
            logging.info(
                'Calculating train/test split times for training prediction span {}, test prediction span {}, test span {}'.format(
                    training_prediction_span,
                    test_prediction_span,
                    test_span
                )
            )
            train_test_split_times = self.calculate_train_test_split_times(
                utils.convert_str_to_relativedelta(training_prediction_span),
                utils.convert_str_to_relativedelta(test_prediction_span),
                test_span
            )
            logging.info('Train/test split times: {}'.format(train_test_split_times))
            for training_data_frequency, max_training_history in itertools.product(
                self.training_data_frequencies,
                self.max_training_histories
            ):
                logging.info(
                    'Generating matrix definitions for training_data_frequency {}, max_training_history {}'.format(
                        training_data_frequency,
                        max_training_history
                    )
                )
                for train_test_split_time in train_test_split_times:
                    logging.info(
                        'Generating matrix definitions for split {}'.format(
                            train_test_split_time
                        )
                    )
                    matrix_set_definitions.append(
                        self.generate_matrix_definitions(
                            train_test_split_time,
                            training_data_frequency,
                            max_training_history,
                            test_span,
                            test_prediction_span,
                            training_prediction_span
                        )
                    )
        return(matrix_set_definitions)

    def calculate_train_test_split_times(
            self,
            training_prediction_span,
            test_prediction_span,
            test_span
        ):
        """ Calculate the split times between train and test matrices. All
        label spans in train matrices will end at this time, and this will be
        the first as of time in the respective test matrix.

        :param training_prediction_span: how much time is included in training 
                                         labels
        :type training_prediction_span: dateutil.relativedelta.relativedelta
        :param test_prediction_span: how much time is included in test labels
        :type test_prediction_span: dateutil.relativedelta.relativedelta
        :param test_span: for how long after the end of a training matrix are
                          test predictions made
        :type test_span: str

        :return: all split times for the temporal parameters
        :rtype: list

        :raises: ValueError if there are no valid split times in the temporal
                 config
        """
        last_test_label_time = self.label_end_time - test_prediction_span
        # final label must be able to have feature data associated with it
        if last_test_label_time > self.feature_end_time:
            last_test_label_time = self.feature_end_time
            raise ValueError('Final test label date cannot be after end of feature time.')
        logging.info('Final label as of date: {}'.format(last_test_label_time))

        # all split times have to allow at least one training label before them
        earliest_possible_split_time = (training_prediction_span
            + max(self.feature_start_time, self.label_start_time))
        logging.info(
            'Earliest possible train/test split time: {}'.format(
                earliest_possible_split_time
            )
        )

        # last split is the first as of time in the final test matrix
        test_delta = utils.convert_str_to_relativedelta(test_span)
        last_split_time = (
            last_test_label_time - test_delta
        )
        logging.info('Final split time: {}'.format(last_split_time))
        if last_split_time < earliest_possible_split_time:
            raise ValueError(
                'No valid train/test split times in temporal config.'
            )
        
        train_test_split_times = []
        train_test_split_time = last_split_time

        while train_test_split_time >= earliest_possible_split_time:
            train_test_split_times.insert(0, train_test_split_time)
            train_test_split_time -= self.model_update_frequency

        return(train_test_split_times)

    # matrix_end_time is now matrix_end_time - label_window
    def calculate_as_of_times(
        self,
        matrix_start_time, # change names
        matrix_end_time,
        data_frequency,
        forward=False
    ):
        """ Given a start and stop time, a frequncy, and a direction, calculate the
        as of times for a matrix.

        :param matrix_start_time: the earliest possible as of time for a matrix
        :param matrix_end_time: the last possible as of time for the matrix
        :param data_frequency: how much time between rows for a single entity
        :param forward: whether to generate times forward from the start time 
                        (True) or backward from the end time (False)

        :return: list of as of times for the matrix
        :rtype: list
        """
        logging.info(
            'Calculating as_of_times from %s to %s using example frequency %s',
            matrix_start_time,
            matrix_end_time,
            data_frequency
        )

        as_of_times = []

        if forward:
            as_of_time = matrix_start_time
            # essentially a do-while loop for test matrices since
            # identical start and end times should include the first
            # date (e.g., ['2017-01-01', '2017-01-01') should give
            # preference to the inclusive side)
            as_of_times.append(as_of_time)
            as_of_time += data_frequency
            while as_of_time < matrix_end_time:
                as_of_times.append(as_of_time)
                as_of_time += data_frequency

        else:
            as_of_time = matrix_end_time
            while as_of_time >= matrix_start_time:
                as_of_times.insert(0, as_of_time)
                as_of_time -= data_frequency

        return(as_of_times)

    def generate_matrix_definitions(
            self,
            train_test_split_time,
            training_data_frequency,
            max_training_history,
            test_span,
            training_prediction_span,
            test_prediction_span
        ):
        """ Given a split time and parameters for train and test matrices,
        generate as of times and metadata for the matrices in the split.

        :param train_test_split_time: the limit of the last label in the matrix
        :type train_test_split_time: datetime.datetime
        :param training_data_frequency: how much time between rows for an entity
                                        in a training matrix
        :type training_data_frequency: str
        :param max_training_history: how far back from split do train 
                                    as_of_times go
        :type max_training_history: str
        :param test_span: how far forward from split do test as_of_times go
        :type test_span: str
        :param training_prediction_span: how much time covered by train labels
        :type training_prediction_span: str
        :param test_prediction_span: how much time is covered by test labels
        :type test_prediction_span: str

        :return: dictionary defining the train and test matrices for a split
        :rtype: dict
        """

        train_matrix_definition = self.define_train_matrix(
            train_test_split_time,
            training_prediction_span,
            max_training_history,
            training_data_frequency
        )

        test_matrix_definitions = self.define_test_matrices(
            train_test_split_time,
            test_span,
            test_prediction_span
        )
            
        matrix_set_definition = {
            'feature_start_time': self.feature_start_time,
            'feature_end_time': self.feature_end_time,
            'label_start_time': self.label_start_time,
            'label_end_time': self.label_end_time,
            'train_matrix': train_matrix_definition,
            'test_matrices': test_matrix_definitions
        }
        logging.info(
            'Matrix definitions for train/test split {}: {}'.format(
                train_test_split_time,
                matrix_set_definition
            )
        )

        return(matrix_set_definition)

    def define_train_matrix(
        self,
        train_test_split_time,
        training_prediction_span,
        max_training_history,
        training_data_frequency
    ):
        """ Given a split time and the parameters of a training matrix, generate
        the as of times and metadata for a train matrix.

        :param train_test_split_time: the limit of the last label in the matrix
        :type train_test_split_time: datetime.datetime
        :param training_prediction_span: how much time is covered by the labels
        :type training_prediction_span: str
        :param max_training_history: how far back from split do as_of_times go
        :type max_training_history: str
        :param training_data_frequency: how much time between rows for an entity
        :type training_data_frequency: str

        :return: dictionary containing the temporal parameters and as of times
                 for a train matrix
        :rtype: dict
        """
        # last as of time in the matrix is 1 label span before split
        training_prediction_delta = utils.convert_str_to_relativedelta(
            training_prediction_span
        )
        last_train_as_of_time = (
            train_test_split_time - training_prediction_delta
        )
        logging.info('last train as of time: {}'.format(last_train_as_of_time))

        # earliest time in matrix can't be farther back than the training
        # history length, the beginning of label time, or the beginning of
        # feature time -- whichever is latest is the limit
        max_training_delta = utils.convert_str_to_relativedelta(
            max_training_history
        )
        earliest_possible_train_as_of_time = (
            last_train_as_of_time - max_training_delta
        )
        experiment_as_of_time_limit = max(
            self.label_start_time,
            self.feature_start_time
        )
        if earliest_possible_train_as_of_time < experiment_as_of_time_limit:
            earliest_possible_train_as_of_time = experiment_as_of_time_limit
        logging.info(
            'earliest possible train as of time: {}'.format(
                earliest_possible_train_as_of_time
            )
        )

        # with the last as of time and the earliest possible time known,
        # calculate all the as of times for the matrix
        train_as_of_times = self.calculate_as_of_times(
            earliest_possible_train_as_of_time,
            last_train_as_of_time,
            utils.convert_str_to_relativedelta(training_data_frequency)
        )
        logging.info('train as of times: {}'.format(train_as_of_times))

        # create a dict of the matrix metadata
        matrix_definition = {
            'matrix_start_time': earliest_possible_train_as_of_time, #rename to 'earliest as of time'?
            'matrix_end_time': ( # make this just split time, rename to 'information cutoff time'?
                train_test_split_time - training_prediction_delta
            ),
            'as_of_times': train_as_of_times,
            # last as of time as new matrix definition parameter
            'training_prediction_span': training_prediction_span,
            'training_data_frequency': training_data_frequency,
            'max_training_history': max_training_history
        }

        return(matrix_definition)

    def define_test_matrices(
        self,
        train_test_split_time,
        test_span,
        test_prediction_span
    ):
        """ Given a train/test split time and a set of testing parameters, 
        generate the metadata and as of times for the test matrices in a split.

        :param train_test_split_time: the limit of the last label in the matrix
        :type train_test_split_time: datetime.datetime
        :param test_span: how far forward from split do test as_of_times go
        :type test_span: str
        :param test_prediction_span: how much time is covered by test labels
        :type test_prediction_span: str

        :return: list of dictionaries defining the test matrices for a split
        :rtype: list
        """
        logging.info(
            'Generating test matrix definitions for train/test split {}'.format(
                train_test_split_time
            )
        )
        test_definitions = []
        test_delta = utils.convert_str_to_relativedelta(test_span)
        as_of_time_limit = train_test_split_time + test_delta
        logging.info('All test as of times before {}'.format(as_of_time_limit))
        for test_data_frequency in self.test_data_frequencies:
            logging.info(
                'Generating test matrix definitions for test data frequency {}'.format(
                    test_data_frequency
                )
            )
            test_as_of_times = self.calculate_as_of_times(
                train_test_split_time,
                as_of_time_limit,
                utils.convert_str_to_relativedelta(test_data_frequency),
                True
            )
            logging.info('Test as of times: {}'.format(test_as_of_times))
            test_definition = {
                'matrix_start_time': train_test_split_time,
                'matrix_end_time': as_of_time_limit,
                'as_of_times': test_as_of_times,
                'test_prediction_span': test_prediction_span,
                'test_data_frequency': test_data_frequency,
                'test_span': test_span
            }
            test_definitions.append(test_definition)
        return(test_definitions)
