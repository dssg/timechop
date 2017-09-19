from timechop.timechop import Timechop
from timechop.utils import convert_str_to_relativedelta
import datetime
from unittest import TestCase
import warnings
import logging

logging.basicConfig(level=logging.DEBUG)

class test_calculate_train_test_split_times(TestCase):
    def test_valid_input(self):
        expected_result = [
            datetime.datetime(2015, 2, 1, 0, 0),
            datetime.datetime(2015, 5, 1, 0, 0),
            datetime.datetime(2015, 8, 1, 0, 0)
        ]
        chopper = Timechop(
            feature_start_time=datetime.datetime(2010, 1, 1, 0, 0),
            feature_end_time=datetime.datetime(2016, 1, 1, 0, 0),
            label_start_time=datetime.datetime(2015, 1, 1, 0, 0),
            label_end_time=datetime.datetime(2017, 1, 1, 0, 0),
            model_update_frequency='3 months',
            training_data_frequencies=['1 day'],
            test_data_frequencies=['1 day'],
            max_training_histories=['1 year'],
            test_spans=['6 months'],
            test_prediction_spans=['1 months'],
            training_prediction_spans=['3 days']
        )
        
        # this should throw a warning because last possible label date is after
        # end of feature time
        with warnings.catch_warnings(record = True) as w:
            warnings.simplefilter("always")
            result = chopper.calculate_train_test_split_times(
                training_prediction_span=convert_str_to_relativedelta('3 days'),
                test_span='6 months',
                test_prediction_span=convert_str_to_relativedelta('1 month')
            )
            assert len(w) == 1
            assert "Final" in str(w[-1].message)

        assert result == expected_result

    def test_invalid_input(self):
        chopper = Timechop(
            feature_start_time=datetime.datetime(2010, 1, 1, 0, 0),
            feature_end_time=datetime.datetime(2016, 1, 1, 0, 0),
            label_start_time=datetime.datetime(2015, 1, 1, 0, 0),
            label_end_time=datetime.datetime(2015, 2, 1, 0, 0),
            model_update_frequency='3 months',
            training_data_frequencies=['1 day'],
            test_data_frequencies=['1 day'],
            max_training_histories=['1 year'],
            test_spans=['6 months'],
            test_prediction_spans=['1 months'],
            training_prediction_spans=['3 days']
        )

        # this should raise an error because there are no valid label dates in
        # the labeling time (label span is longer than labeling time)
        with self.assertRaises(ValueError):
            chopper.calculate_train_test_split_times(
                training_prediction_span=convert_str_to_relativedelta('3 days'),
                test_span='6 months',
                test_prediction_span=convert_str_to_relativedelta('1 month')
            )


def test_calculate_as_of_times_one_day_freq():
    expected_result = [
        datetime.datetime(2011, 1, 1, 0, 0),
        datetime.datetime(2011, 1, 2, 0, 0),
        datetime.datetime(2011, 1, 3, 0, 0),
        datetime.datetime(2011, 1, 4, 0, 0),
        datetime.datetime(2011, 1, 5, 0, 0),
        datetime.datetime(2011, 1, 6, 0, 0),
        datetime.datetime(2011, 1, 7, 0, 0),
        datetime.datetime(2011, 1, 8, 0, 0),
        datetime.datetime(2011, 1, 9, 0, 0),
        datetime.datetime(2011, 1, 10, 0, 0),
        datetime.datetime(2011, 1, 11, 0, 0)
    ]
    chopper = Timechop(
        feature_start_time=datetime.datetime(1990, 1, 1, 0, 0),
        feature_end_time=datetime.datetime(2012, 1, 1, 0, 0),
        label_start_time=datetime.datetime(2010, 1, 1, 0, 0),
        label_end_time=datetime.datetime(2012, 1, 1, 0, 0),
        model_update_frequency='1 year',
        training_data_frequencies=['1 days'],
        test_data_frequencies=['7 days'],
        max_training_histories=['10 days', '1 year'],
        test_spans=['1 month'],
        test_prediction_spans=['1 day'],
        training_prediction_spans=['3 months']
    )
    result = chopper.calculate_as_of_times(
        matrix_start_time = datetime.datetime(2011, 1, 1, 0, 0),
        matrix_end_time = datetime.datetime(2011, 1, 11, 0, 0),
        data_frequency = convert_str_to_relativedelta('1 days')
    )
    assert(result == expected_result)


def test_calculate_as_of_times_three_day_freq():
    expected_result = [
        datetime.datetime(2011, 1, 1, 0, 0),
        datetime.datetime(2011, 1, 4, 0, 0),
        datetime.datetime(2011, 1, 7, 0, 0),
        datetime.datetime(2011, 1, 10, 0, 0),
    ]
    chopper = Timechop(
        feature_start_time=datetime.datetime(1990, 1, 1, 0, 0),
        feature_end_time=datetime.datetime(2012, 1, 1, 0, 0),
        label_start_time=datetime.datetime(2010, 1, 1, 0, 0),
        label_end_time=datetime.datetime(2012, 1, 1, 0, 0),
        model_update_frequency='1 year',
        training_data_frequencies=['1 days'],
        test_data_frequencies=['7 days'],
        max_training_histories=['10 days', '1 year'],
        test_spans=['1 month'],
        test_prediction_spans=['1 day'],
        training_prediction_spans=['3 months']
    )
    result = chopper.calculate_as_of_times(
        matrix_start_time = datetime.datetime(2011, 1, 1, 0, 0),
        matrix_end_time = datetime.datetime(2011, 1, 11, 0, 0),
        data_frequency = convert_str_to_relativedelta('3 days'),
        forward=True
    )
    assert(result == expected_result)


class test_generate_matrix_definitions(TestCase):
    def test_look_back_time_equal_modeling_start(self):
        expected_result = {
            'feature_start_time': datetime.datetime(1990, 1, 1, 0, 0),
            'label_start_time': datetime.datetime(2010, 1, 1, 0, 0),
            'feature_end_time': datetime.datetime(2010, 1, 11, 0, 0),
            'label_end_time': datetime.datetime(2010, 1, 11, 0, 0),
            'train_matrix': {
                'matrix_start_time': datetime.datetime(2010, 1, 1, 0, 0),
                'matrix_end_time': datetime.datetime(2010, 1, 6, 0, 0),
                'as_of_times': [
                    datetime.datetime(2010, 1, 1, 0, 0),
                    datetime.datetime(2010, 1, 2, 0, 0),
                    datetime.datetime(2010, 1, 3, 0, 0),
                    datetime.datetime(2010, 1, 4, 0, 0),
                    datetime.datetime(2010, 1, 5, 0, 0)
                ],
                'training_prediction_span': '1 day',
                'training_data_frequency': '1 days',
                'max_training_history': '5 days'
            },
            'test_matrices': [{
                'matrix_start_time': datetime.datetime(2010, 1, 6, 0, 0),
                'matrix_end_time': datetime.datetime(2010, 1, 11, 0, 0),
                'as_of_times': [
                    datetime.datetime(2010, 1, 6, 0, 0),
                    datetime.datetime(2010, 1, 9, 0, 0),
                ],
                'test_prediction_span': '1 day',
                'test_data_frequency': '3 days',
                'test_span': '5 days'
            }]
        }
        chopper = Timechop(
            feature_start_time=datetime.datetime(1990, 1, 1, 0, 0),
            feature_end_time=datetime.datetime(2010, 1, 11, 0, 0),
            label_start_time=datetime.datetime(2010, 1, 1, 0, 0),
            label_end_time=datetime.datetime(2010, 1, 11, 0, 0),
            model_update_frequency='5 days',
            training_data_frequencies=['1 days'],
            test_data_frequencies=['3 days'],
            max_training_histories=['5 days'],
            test_spans=['5 days'],
            test_prediction_spans=['1 day'],
            training_prediction_spans=['1 day']
        )
        result = chopper.generate_matrix_definitions(
            train_test_split_time = datetime.datetime(2010, 1, 6, 0, 0),
            training_data_frequency='1 days',
            max_training_history='5 days',
            test_span='5 days',
            test_prediction_span='1 day',
            training_prediction_span='1 day'
        )
        assert result == expected_result

    def test_look_back_time_before_modeling_start(self):
        expected_result = {
            'feature_start_time': datetime.datetime(1990, 1, 1, 0, 0),
            'label_start_time': datetime.datetime(2010, 1, 1, 0, 0),
            'feature_end_time': datetime.datetime(2010, 1, 11, 0, 0),
            'label_end_time': datetime.datetime(2010, 1, 11, 0, 0),
            'train_matrix': {
                'matrix_start_time': datetime.datetime(2010, 1, 1, 0, 0),
                'matrix_end_time': datetime.datetime(2010, 1, 6, 0, 0),
                'as_of_times': [
                    datetime.datetime(2010, 1, 1, 0, 0),
                    datetime.datetime(2010, 1, 2, 0, 0),
                    datetime.datetime(2010, 1, 3, 0, 0),
                    datetime.datetime(2010, 1, 4, 0, 0),
                    datetime.datetime(2010, 1, 5, 0, 0)
                ],
                'training_prediction_span': '1 day',
                'training_data_frequency': '1 days',
                'max_training_history': '10 days'
            },
            'test_matrices': [
                {
                    'matrix_start_time': datetime.datetime(2010, 1, 6, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 11, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 6, 0, 0),
                        datetime.datetime(2010, 1, 9, 0, 0)
                    ],
                    'test_prediction_span': '1 day',
                    'test_data_frequency': '3 days',
                    'test_span': '5 days'
                },
                {
                    'matrix_start_time': datetime.datetime(2010, 1, 6, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 11, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 6, 0, 0),
                    ],
                    'test_prediction_span': '1 day',
                    'test_data_frequency': '6 days',
                    'test_span': '5 days'
                }
            ]
        }
        chopper = Timechop(
            feature_start_time=datetime.datetime(1990, 1, 1, 0, 0),
            feature_end_time=datetime.datetime(2010, 1, 11, 0, 0),
            label_start_time=datetime.datetime(2010, 1, 1, 0, 0),
            label_end_time=datetime.datetime(2010, 1, 11, 0, 0),
            model_update_frequency='5 days',
            training_data_frequencies=['1 days'],
            test_data_frequencies=['3 days', '6 days'],
            max_training_histories=['10 days'],
            test_spans=['5 days'],
            test_prediction_spans=['1 day'],
            training_prediction_spans=['1 day']
        )
        result = chopper.generate_matrix_definitions(
            train_test_split_time = datetime.datetime(2010, 1, 6, 0, 0),
            training_data_frequency='1 days',
            max_training_history='10 days',
            test_span='5 days',
            test_prediction_span='1 day',
            training_prediction_span='1 day'
        )
        assert result == expected_result


class test_chop_time(TestCase):
    def test_evenly_divisible_values(self):
        expected_result = [
            {
                'feature_start_time': datetime.datetime(1990, 1, 1, 0, 0),
                'label_start_time': datetime.datetime(2010, 1, 1, 0, 0),
                'feature_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                'label_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                'train_matrix': {
                    'matrix_start_time': datetime.datetime(2010, 1, 1, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 6, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 1, 0, 0),
                        datetime.datetime(2010, 1, 2, 0, 0),
                        datetime.datetime(2010, 1, 3, 0, 0),
                        datetime.datetime(2010, 1, 4, 0, 0),
                        datetime.datetime(2010, 1, 5, 0, 0)
                    ],
                    'training_prediction_span': '1 day',
                    'training_data_frequency': '1 days',
                    'max_training_history': '5 days'
                },
                'test_matrices': [{
                    'matrix_start_time': datetime.datetime(2010, 1, 6, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 11, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 6, 0, 0),
                        datetime.datetime(2010, 1, 7, 0, 0),
                        datetime.datetime(2010, 1, 8, 0, 0),
                        datetime.datetime(2010, 1, 9, 0, 0),
                        datetime.datetime(2010, 1, 10, 0, 0)
                    ],
                    'test_prediction_span': '1 day',
                    'test_data_frequency': '1 days',
                    'test_span': '5 days'
                }]
            },
            {
                'feature_start_time': datetime.datetime(1990, 1, 1, 0, 0),
                'label_start_time': datetime.datetime(2010, 1, 1, 0, 0),
                'feature_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                'label_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                'train_matrix': {
                    'matrix_start_time': datetime.datetime(2010, 1, 6, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 11, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 6, 0, 0),
                        datetime.datetime(2010, 1, 7, 0, 0),
                        datetime.datetime(2010, 1, 8, 0, 0),
                        datetime.datetime(2010, 1, 9, 0, 0),
                        datetime.datetime(2010, 1, 10, 0, 0)
                    ],
                    'training_prediction_span': '1 day',
                    'training_data_frequency': '1 days',
                    'max_training_history': '5 days'
                },
                'test_matrices': [{
                    'matrix_start_time': datetime.datetime(2010, 1, 11, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 11, 0, 0),
                        datetime.datetime(2010, 1, 12, 0, 0),
                        datetime.datetime(2010, 1, 13, 0, 0),
                        datetime.datetime(2010, 1, 14, 0, 0),
                        datetime.datetime(2010, 1, 15, 0, 0)
                    ],
                    'test_prediction_span': '1 day',
                    'test_data_frequency': '1 days',
                    'test_span': '5 days'
                }]
            }
        ]
        chopper = Timechop(
            feature_start_time=datetime.datetime(1990, 1, 1, 0, 0),
            feature_end_time=datetime.datetime(2010, 1, 16, 0, 0),
            label_start_time=datetime.datetime(2010, 1, 1, 0, 0),
            label_end_time=datetime.datetime(2010, 1, 16, 0, 0),
            model_update_frequency='5 days',
            training_data_frequencies=['1 days'],
            test_data_frequencies=['1 days'],
            max_training_histories=['5 days'],
            test_spans=['5 days'],
            test_prediction_spans=['1 day'],
            training_prediction_spans=['1 day']
        )
        result = chopper.chop_time()
        assert(result == expected_result)

    def test_training_prediction_span_longer_than_1_day(self):
        expected_result = [
            {
                'feature_start_time': datetime.datetime(1990, 1, 1, 0, 0),
                'label_start_time': datetime.datetime(2010, 1, 1, 0, 0),
                'feature_end_time': datetime.datetime(2010, 1, 19, 0, 0),
                'label_end_time': datetime.datetime(2010, 1, 19, 0, 0),
                'train_matrix': {
                    'matrix_start_time': datetime.datetime(2010, 1, 1, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 6, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 1, 0, 0),
                        datetime.datetime(2010, 1, 2, 0, 0),
                        datetime.datetime(2010, 1, 3, 0, 0),
                        datetime.datetime(2010, 1, 4, 0, 0),
                        datetime.datetime(2010, 1, 5, 0, 0)
                    ],
                    'training_prediction_span': '5 days',
                    'training_data_frequency': '1 days',
                    'max_training_history': '5 days'
                },
                'test_matrices': [{
                    'matrix_start_time': datetime.datetime(2010, 1, 10, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 15, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 10, 0, 0),
                        datetime.datetime(2010, 1, 11, 0, 0),
                        datetime.datetime(2010, 1, 12, 0, 0),
                        datetime.datetime(2010, 1, 13, 0, 0),
                        datetime.datetime(2010, 1, 14, 0, 0)
                    ],
                    'test_prediction_span': '5 days',
                    'test_data_frequency': '1 days',
                    'test_span': '5 days'
                }]
            }
        ]
        chopper = Timechop(
            feature_start_time=datetime.datetime(1990, 1, 1, 0, 0),
            feature_end_time=datetime.datetime(2010, 1, 19, 0, 0),
            label_start_time=datetime.datetime(2010, 1, 1, 0, 0),
            label_end_time=datetime.datetime(2010, 1, 19, 0, 0),
            model_update_frequency='5 days',
            training_data_frequencies=['1 days'],
            test_data_frequencies=['1 days'],
            max_training_histories=['5 days'],
            test_spans=['5 days'],
            test_prediction_spans=['5 days'],
            training_prediction_spans=['5 days']
        )
        result = chopper.chop_time()
        assert(result == expected_result)

    def test_unevenly_divisible_lookback_duration(self):
        expected_result = [
            {
                'feature_start_time': datetime.datetime(1990, 1, 1, 0, 0),
                'label_start_time': datetime.datetime(2010, 1, 1, 0, 0),
                'feature_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                'label_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                'train_matrix': {
                    'matrix_start_time': datetime.datetime(2010, 1, 1, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 6, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 1, 0, 0),
                        datetime.datetime(2010, 1, 2, 0, 0),
                        datetime.datetime(2010, 1, 3, 0, 0),
                        datetime.datetime(2010, 1, 4, 0, 0),
                        datetime.datetime(2010, 1, 5, 0, 0)
                    ],
                    'training_prediction_span': '1 day',
                    'training_data_frequency': '1 days',
                    'max_training_history': '7 days'
                },
                'test_matrices': [{
                    'matrix_start_time': datetime.datetime(2010, 1, 6, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 11, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 6, 0, 0),
                        datetime.datetime(2010, 1, 7, 0, 0),
                        datetime.datetime(2010, 1, 8, 0, 0),
                        datetime.datetime(2010, 1, 9, 0, 0),
                        datetime.datetime(2010, 1, 10, 0, 0)
                    ],
                    'test_prediction_span': '1 day',
                    'test_data_frequency': '1 days',
                    'test_span': '5 days'
                }]
            },
            {
                'feature_start_time': datetime.datetime(1990, 1, 1, 0, 0),
                'label_start_time': datetime.datetime(2010, 1, 1, 0, 0),
                'feature_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                'label_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                'train_matrix': {
                    'matrix_start_time': datetime.datetime(2010, 1, 4, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 11, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 4, 0, 0),
                        datetime.datetime(2010, 1, 5, 0, 0),
                        datetime.datetime(2010, 1, 6, 0, 0),
                        datetime.datetime(2010, 1, 7, 0, 0),
                        datetime.datetime(2010, 1, 8, 0, 0),
                        datetime.datetime(2010, 1, 9, 0, 0),
                        datetime.datetime(2010, 1, 10, 0, 0)
                    ],
                    'training_prediction_span': '1 day',
                    'training_data_frequency': '1 days',
                    'max_training_history': '7 days'
                },
                'test_matrices': [{
                    'matrix_start_time': datetime.datetime(2010, 1, 11, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 11, 0, 0),
                        datetime.datetime(2010, 1, 12, 0, 0),
                        datetime.datetime(2010, 1, 13, 0, 0),
                        datetime.datetime(2010, 1, 14, 0, 0),
                        datetime.datetime(2010, 1, 15, 0, 0)
                    ],
                    'test_prediction_span': '1 day',
                    'test_data_frequency': '1 days',
                    'test_span': '5 days'
                }]
            }
        ]
        chopper = Timechop(
            feature_start_time=datetime.datetime(1990, 1, 1, 0, 0),
            feature_end_time=datetime.datetime(2010, 1, 16, 0, 0),
            label_start_time=datetime.datetime(2010, 1, 1, 0, 0),
            label_end_time=datetime.datetime(2010, 1, 16, 0, 0),
            model_update_frequency='5 days',
            training_data_frequencies=['1 days'],
            test_data_frequencies=['1 days'],
            max_training_histories=['7 days'],
            test_spans=['5 days'],
            test_prediction_spans=['1 day'],
            training_prediction_spans=['1 day']
        )
        result = chopper.chop_time()
        assert(result == expected_result)

    def test_unevenly_divisible_update_window(self):
        expected_result = [
            {
                'feature_start_time': datetime.datetime(1990, 1, 1, 0, 0),
                'label_start_time': datetime.datetime(2010, 1, 3, 0, 0),
                'feature_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                'label_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                'train_matrix': {
                    'matrix_start_time': datetime.datetime(2010, 1, 3, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 6, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 3, 0, 0),
                        datetime.datetime(2010, 1, 4, 0, 0),
                        datetime.datetime(2010, 1, 5, 0, 0)
                    ],
                    'training_prediction_span': '1 day',
                    'training_data_frequency': '1 days',
                    'max_training_history': '5 days'
                },
                'test_matrices': [{
                    'matrix_start_time': datetime.datetime(2010, 1, 6, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 11, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 6, 0, 0),
                        datetime.datetime(2010, 1, 7, 0, 0),
                        datetime.datetime(2010, 1, 8, 0, 0),
                        datetime.datetime(2010, 1, 9, 0, 0),
                        datetime.datetime(2010, 1, 10, 0, 0)
                    ],
                    'test_prediction_span': '1 day',
                    'test_data_frequency': '1 days',
                    'test_span': '5 days'
                }]
            },
            {
                'feature_start_time': datetime.datetime(1990, 1, 1, 0, 0),
                'label_start_time': datetime.datetime(2010, 1, 3, 0, 0),
                'feature_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                'label_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                'train_matrix': {
                    'matrix_start_time': datetime.datetime(2010, 1, 6, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 11, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 6, 0, 0),
                        datetime.datetime(2010, 1, 7, 0, 0),
                        datetime.datetime(2010, 1, 8, 0, 0),
                        datetime.datetime(2010, 1, 9, 0, 0),
                        datetime.datetime(2010, 1, 10, 0, 0)
                    ],
                    'training_prediction_span': '1 day',
                    'training_data_frequency': '1 days',
                    'max_training_history': '5 days'
                },
                'test_matrices': [{
                    'matrix_start_time': datetime.datetime(2010, 1, 11, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 11, 0, 0),
                        datetime.datetime(2010, 1, 12, 0, 0),
                        datetime.datetime(2010, 1, 13, 0, 0),
                        datetime.datetime(2010, 1, 14, 0, 0),
                        datetime.datetime(2010, 1, 15, 0, 0)
                    ],
                    'test_prediction_span': '1 day',
                    'test_data_frequency': '1 days',
                    'test_span': '5 days'
                }]
            }
        ]
        chopper = Timechop(
            feature_start_time=datetime.datetime(1990, 1, 1, 0, 0),
            feature_end_time=datetime.datetime(2010, 1, 16, 0, 0),
            label_start_time=datetime.datetime(2010, 1, 3, 0, 0),
            label_end_time=datetime.datetime(2010, 1, 16, 0, 0),
            model_update_frequency='5 days',
            training_data_frequencies=['1 days'],
            test_data_frequencies=['1 days'],
            max_training_histories=['5 days'],
            test_spans=['5 days'],
            test_prediction_spans=['1 day'],
            training_prediction_spans=['1 day']
        )
        result = chopper.chop_time()
        assert(result == expected_result)


class test__init__(TestCase):
    def test_bad_feature_start_time(self):
        with self.assertRaises(ValueError):
            chopper = Timechop(
                feature_start_time=datetime.datetime(2011, 1, 1, 0, 0),
                feature_end_time=datetime.datetime(2010, 1, 16, 0, 0),
                label_start_time=datetime.datetime(2010, 1, 3, 0, 0),
                label_end_time=datetime.datetime(2010, 1, 16, 0, 0),
                model_update_frequency='5 days',
                training_data_frequencies=['1 days'],
                test_data_frequencies=['1 days'],
                max_training_histories=['5 days'],
                test_spans=['5 days'],
                test_prediction_spans=['1 day'],
                training_prediction_spans=['1 day']
            )
