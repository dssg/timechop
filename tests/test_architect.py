from timechop.architect import Architect
from tests.utils import create_features_and_labels_schemas
from tests.utils import create_entity_date_df
from tests.utils import convert_string_column_to_date
from tests.utils import NamedTempFile
from tests.utils import TemporaryDirectory
import testing.postgresql
import csv
import datetime
import pandas as pd
import os
from sqlalchemy import create_engine
from unittest import TestCase
from metta import metta_io as metta
from mock import Mock


# make some fake features data

features0 = [
    [0, '2016-01-01', 2, 0],
    [1, '2016-01-01', 1, 2],
    [0, '2016-02-01', 2, 3],
    [1, '2016-02-01', 2, 4],
    [0, '2016-03-01', 3, 3],
    [1, '2016-03-01', 3, 4],
    [0, '2016-04-01', 4, 3],
    [1, '2016-05-01', 5, 4]
] 

features1 = [
    [2, '2016-01-01', 1, 1],
    [3, '2016-01-01', 1, 2],
    [2, '2016-02-01', 2, 3],
    [3, '2016-02-01', 2, 2],
    [0, '2016-03-01', 3, 3],
    [1, '2016-03-01', 3, 4],
    [2, '2016-03-01', 3, 3],
    [3, '2016-03-01', 3, 4]
] 

features_tables = [features0, features1]

# make some fake labels data

labels = [
    [0, '2016-02-01', '1 month', 'booking', 'binary', 0],
    [0, '2016-03-01', '1 month', 'booking', 'binary', 0],
    [0, '2016-04-01', '1 month', 'booking', 'binary', 0],
    [0, '2016-05-01', '1 month', 'booking', 'binary', 1],
    [0, '2016-01-01', '1 month', 'ems',     'binary', 0],
    [0, '2016-02-01', '1 month', 'ems',     'binary', 0],
    [0, '2016-03-01', '1 month', 'ems',     'binary', 0],
    [0, '2016-04-01', '1 month', 'ems',     'binary', 0],
    [0, '2016-05-01', '1 month', 'ems',     'binary', 0],
    [1, '2016-01-01', '1 month', 'booking', 'binary', 0],
    [1, '2016-02-01', '1 month', 'booking', 'binary', 0],
    [1, '2016-03-01', '1 month', 'booking', 'binary', 0],
    [1, '2016-04-01', '1 month', 'booking', 'binary', 0],
    [1, '2016-05-01', '1 month', 'booking', 'binary', 1],
    [1, '2016-01-01', '1 month', 'ems',     'binary', 0],
    [1, '2016-02-01', '1 month', 'ems',     'binary', 0],
    [1, '2016-03-01', '1 month', 'ems',     'binary', 0],
    [1, '2016-04-01', '1 month', 'ems',     'binary', 0],
    [1, '2016-05-01', '1 month', 'ems',     'binary', 0],
    [2, '2016-01-01', '1 month', 'booking', 'binary', 0],
    [2, '2016-02-01', '1 month', 'booking', 'binary', 0],
    [2, '2016-03-01', '1 month', 'booking', 'binary', 1],
    [2, '2016-04-01', '1 month', 'booking', 'binary', 0],
    [2, '2016-05-01', '1 month', 'booking', 'binary', 1],
    [2, '2016-01-01', '1 month', 'ems',     'binary', 0],
    [2, '2016-02-01', '1 month', 'ems',     'binary', 0],
    [2, '2016-03-01', '1 month', 'ems',     'binary', 0],
    [2, '2016-04-01', '1 month', 'ems',     'binary', 0],
    [2, '2016-05-01', '1 month', 'ems',     'binary', 1],
    [3, '2016-01-01', '1 month', 'booking', 'binary', 0],
    [3, '2016-02-01', '1 month', 'booking', 'binary', 0],
    [3, '2016-03-01', '1 month', 'booking', 'binary', 1],
    [3, '2016-04-01', '1 month', 'booking', 'binary', 0],
    [3, '2016-05-01', '1 month', 'booking', 'binary', 1],
    [3, '2016-01-01', '1 month', 'ems',     'binary', 0],
    [3, '2016-02-01', '1 month', 'ems',     'binary', 0],
    [3, '2016-03-01', '1 month', 'ems',     'binary', 0],
    [3, '2016-04-01', '1 month', 'ems',     'binary', 1],
    [3, '2016-05-01', '1 month', 'ems',     'binary', 0],
]

label_name = 'booking'
label_type = 'binary'

db_config = {
    'features_schema_name': 'features',
    'labels_schema_name': 'labels',
    'labels_table_name': 'labels',
}

def test_build_labels_query():
    """ Test the generate_labels_query function by checking whether the query
    produces the correct labels
    """
    # set up labeling config variables
    dates = [datetime.datetime(2016, 1, 1, 0, 0),
             datetime.datetime(2016, 2, 1, 0, 0)]

    with testing.postgresql.Postgresql() as postgresql:
        # create an engine and generate a table with fake feature data
        engine = create_engine(postgresql.url())
        create_features_and_labels_schemas(engine, features_tables, labels)

    # make a dataframe of labels to test against
    labels_df = pd.DataFrame(
        labels,
        columns = [
            'entity_id',
            'as_of_date',
            'prediction_window',
            'label_name',
            'label_type',
            'label'
        ]
    )
    labels_df['as_of_date'] = convert_string_column_to_date(labels_df['as_of_date'])

    # create an engine and generate a table with fake feature data
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        create_features_and_labels_schemas(engine, features_tables, labels)
        with TemporaryDirectory() as temp_dir:
            matrix_maker = Architect(
                beginning_of_time = datetime.datetime(2010, 1, 1, 0, 0),
                label_names = ['booking'],
                label_types = ['binary'],
                db_config = db_config,
                matrix_directory = temp_dir,
                user_metadata = {},
                engine = engine
            )       

            # get the queries and test them
            for date in dates:
                date = date.date()
                df = labels_df[labels_df['label_name'] == label_name]
                df = df[labels_df['label_type'] == label_type]
                df = df[labels_df['prediction_window'] == '1 month']
                print(date)
                df = df[labels_df['as_of_date'] == date]
                df = df[['entity_id', 'as_of_date', 'label']]
                df = df.rename(
                    columns = {
                        "entity_id": "entity_id",
                        "as_of_date": "as_of_date",
                        "label": label_name
                    }
                )
                df = df.reset_index(drop = True)
                query = matrix_maker.build_labels_query(
                    as_of_times = [date],
                    label_type = label_type,
                    label_name = label_name,
                    final_column = ', label as {}'.format(label_name)
                )
                result = pd.read_sql(query, engine)
                test = (result == df)
                assert(test.all().all())

def test_write_to_csv():
    """ Test the write_to_csv function by checking whether the csv contains the
    correct number of lines.
    """
    with testing.postgresql.Postgresql() as postgresql:
        # create an engine and generate a table with fake feature data
        engine = create_engine(postgresql.url())
        create_features_and_labels_schemas(engine, features_tables, labels)

        with TemporaryDirectory() as temp_dir:
            matrix_maker = Architect(
                beginning_of_time = datetime.datetime(2010, 1, 1, 0, 0),
                label_names = ['booking'],
                label_types = ['binary'],
                db_config = db_config,
                matrix_directory = temp_dir,
                user_metadata = {},
                engine = engine
            )

            # for each table, check that corresponding csv has the correct # of rows
            for table in features_tables:
                with NamedTempFile() as f:
                    matrix_maker.write_to_csv(
                        '''
                            select * 
                            from features.features{}
                        '''.format(features_tables.index(table)),
                        f.name
                    )
                    f.seek(0)
                    reader = csv.reader(f)
                    assert(len([row for row in reader]) == len(table) + 1)


def test_make_entity_date_table():
    """ Test that the make_entity_date_table function contains the correct
    values.
    """
    dates = [datetime.datetime(2016, 1, 1, 0, 0),
             datetime.datetime(2016, 2, 1, 0, 0),
             datetime.datetime(2016, 3, 1, 0, 0)]

    # make a dataframe of entity ids and dates to test against
    ids_dates = create_entity_date_df(dates, labels, dates, 'booking', 'binary')

    with testing.postgresql.Postgresql() as postgresql:
        # create an engine and generate a table with fake feature data
        engine = create_engine(postgresql.url())
        create_features_and_labels_schemas(engine, features_tables, labels)

        with TemporaryDirectory() as temp_dir:
            matrix_maker = Architect(
                beginning_of_time = datetime.datetime(2010, 1, 1, 0, 0),
                label_names = ['booking'],
                label_types = ['binary'],
                db_config = db_config,
                matrix_directory = temp_dir,
                user_metadata = {},
                engine = engine
            )
            engine.execute(
                'CREATE TABLE features.tmp_entity_date (a int, b date);'
            )
            # call the function to test the creation of the table
            entity_date_table_name = matrix_maker.make_entity_date_table(
                as_of_times = dates,
                label_type = 'binary',
                label_name = 'booking',
                feature_table_names = ['features0', 'features1'],
                matrix_uuid = 'my_uuid',
                matrix_type = 'train'
            )

            # read in the table
            result = pd.read_sql(
                "select * from features.{} order by entity_id, as_of_date".format(entity_date_table_name),
                engine
            )
            labels_df = pd.read_sql('select * from labels.labels', engine)

            # compare the table to the test dataframe
            print("ids_dates")
            for i, row in ids_dates.iterrows():
                print(row.values)
            print("result")
            for i, row in result.iterrows():
                print(row.values)
            test = (result == ids_dates)
            print(test)
            assert(test.all().all())

def test_build_outer_join_query():
    """ 
    """
    dates = [datetime.datetime(2016, 1, 1, 0, 0),
             datetime.datetime(2016, 2, 1, 0, 0)]

    # make dataframe for entity ids and dates
    ids_dates = create_entity_date_df(dates, labels, dates, 'booking', 'binary')
    
    # make dataframes of features to test against
    features_dfs = []
    for table in features_tables:
        temp_df = pd.DataFrame(
            table,
            columns = ['entity_id', 'as_of_date', 'f1', 'f2']
        )
        temp_df['as_of_date'] = convert_string_column_to_date(temp_df['as_of_date'])
        features_dfs.append(
            ids_dates.merge(
                right = temp_df,
                how = 'left',
                on = ['entity_id', 'as_of_date']
            )
        )

    # create an engine and generate a table with fake feature data
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        create_features_and_labels_schemas(engine, features_tables, labels)

        with TemporaryDirectory() as temp_dir:
            matrix_maker = Architect(
                beginning_of_time = datetime.datetime(2010, 1, 1, 0, 0),
                label_names = ['booking'],
                label_types = ['binary'],
                db_config = db_config,
                matrix_directory = temp_dir,
                user_metadata = {},
                engine = engine
            )

            # make the entity-date table
            entity_date_table_name = matrix_maker.make_entity_date_table(
                as_of_times = dates,
                label_type = 'binary',
                label_name = 'booking',
                feature_table_names = ['features0', 'features1'],
                matrix_type = 'train',
                matrix_uuid = 'my_uuid',
            )

            # get the queries and test them
            for table_number, df in enumerate(features_dfs):
                table_name = 'features{}'.format(table_number)
                df = df.fillna(0)
                query = matrix_maker.build_outer_join_query(
                    as_of_times = dates,
                    right_table_name = 'features.{}'.format(table_name),
                    entity_date_table_name = 'features.{}'.format(entity_date_table_name),
                    right_column_selections = matrix_maker._format_imputations(
                        ['f1', 'f2']
                    )
                )
                result = pd.read_sql(query, engine)
                test = (result == df)
                assert(test.all().all())

class TestMergeFeatureCSVs(TestCase):
    def test_merge_feature_csvs(self):
        with TemporaryDirectory() as temp_dir:
            matrix_maker = Architect(
                beginning_of_time = datetime.datetime(2010, 1, 1, 0, 0),
                label_names = ['booking'],
                label_types = ['binary'],
                db_config = db_config,
                matrix_directory = temp_dir,
                user_metadata = {},
                engine = None
            )
            rowlists = [
                [
                    ('entity_id', 'date', 'label'),
                    (1, 2, True),
                    (4, 5, False),
                    (7, 8, True),
                ],
                [
                    ('entity_id', 'date', 'f1'),
                    (1, 2, 3),
                    (4, 5, 6),
                    (7, 8, 9),
                ],
                [
                    ('entity_id', 'date', 'f2'),
                    (1, 2, 3),
                    (4, 5, 9),
                    (7, 8, 15),
                ],
                [
                    ('entity_id', 'date', 'f3'),
                    (1, 2, 2),
                    (4, 5, 20),
                    (7, 8, 56),
                ],
            ]

            sourcefiles = []
            for rows in rowlists:

                f = NamedTempFile()
                sourcefiles.append(f)
                writer = csv.writer(f)
                for row in rows:
                    writer.writerow(row)
                f.seek(0)
            try:
                with NamedTempFile() as outfile:
                    matrix_maker.merge_feature_csvs(
                        [f.name for f in sourcefiles],
                        outfile.name
                    )
                    reader = csv.reader(outfile)
                    result = [row for row in reader]
                    self.assertEquals(result, [
                        ['entity_id', 'date', 'f1', 'f2', 'f3','label'],
                        ['1', '2', '3', '3', '2', 'True'],
                        ['4', '5', '6', '9', '20', 'False'],
                        ['7', '8', '9', '15', '56', 'True']
                    ])
            finally:
                for sourcefile in sourcefiles:
                    sourcefile.close()


    def test_badinput(self):
        with TemporaryDirectory() as temp_dir:
            matrix_maker = Architect(
                beginning_of_time = datetime.datetime(2010, 1, 1, 0, 0),
                label_names = ['booking'],
                label_types = ['binary'],
                db_config = db_config,
                matrix_directory = temp_dir,
                user_metadata = {},
                engine = None
            )
            rowlists = [
                [
                    ('entity_id', 'date', 'f1'),
                    (1, 3, 3),
                    (4, 5, 6),
                    (7, 8, 9),
                ],
                [
                    ('entity_id', 'date', 'f2'),
                    (1, 2, 3),
                    (4, 5, 9),
                    (7, 8, 15),
                ],
                [
                    ('entity_id', 'date', 'f3'),
                    (1, 2, 2),
                    (4, 5, 20),
                    (7, 8, 56),
                ],
            ]

            sourcefiles = []
            for rows in rowlists:
                f = NamedTempFile()
                sourcefiles.append(f)
                writer = csv.writer(f)
                for row in rows:
                    writer.writerow(row)
                f.seek(0)
            try:
                with NamedTempFile() as outfile:
                    with self.assertRaises(ValueError):
                        matrix_maker.merge_feature_csvs(
                            [f.name for f in sourcefiles],
                            outfile.name
                        )
            finally:
                for sourcefile in sourcefiles:
                    sourcefile.close()

def test_generate_plans():
    matrix_set_definitions = [
        {
            'beginning_of_time': datetime.datetime(1990, 1, 1, 0, 0),
            'modeling_start_time': datetime.datetime(2010, 1, 1, 0, 0),
            'modeling_end_time': datetime.datetime(2010, 1, 16, 0, 0),
            'train_matrix': {
                'matrix_start_time': datetime.datetime(2010, 1, 1, 0, 0),
                'matrix_end_time': datetime.datetime(2010, 1, 6, 0, 0),
                'as_of_times': [
                    datetime.datetime(2010, 1, 1, 0, 0),
                    datetime.datetime(2010, 1, 2, 0, 0),
                    datetime.datetime(2010, 1, 3, 0, 0),
                    datetime.datetime(2010, 1, 4, 0, 0),
                    datetime.datetime(2010, 1, 5, 0, 0)
                ]
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
                ]
            }]
        },
        {
            'beginning_of_time': datetime.datetime(1990, 1, 1, 0, 0),
            'modeling_start_time': datetime.datetime(2010, 1, 1, 0, 0),
            'modeling_end_time': datetime.datetime(2010, 1, 16, 0, 0),
            'train_matrix': {
                'matrix_start_time': datetime.datetime(2010, 1, 6, 0, 0),
                'matrix_end_time': datetime.datetime(2010, 1, 11, 0, 0),
                'as_of_times': [
                    datetime.datetime(2010, 1, 6, 0, 0),
                    datetime.datetime(2010, 1, 7, 0, 0),
                    datetime.datetime(2010, 1, 8, 0, 0),
                    datetime.datetime(2010, 1, 9, 0, 0),
                    datetime.datetime(2010, 1, 10, 0, 0)
                ]
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
                ]
            }]
        }
    ]
    feature_dict_one = {'features0': ['f1', 'f2'], 'features1': ['f1', 'f2']}
    feature_dict_two = {'features2': ['f3', 'f4'], 'features3': ['f5', 'f6']}
    feature_dicts = [feature_dict_one, feature_dict_two]
    architect = Architect(
        beginning_of_time = datetime.datetime(2010, 1, 1, 0, 0),
        label_names = ['booking'],
        label_types = ['binary'],
        db_config = db_config,
        user_metadata = {},
        matrix_directory = '', # this test won't write anything
        engine = None # or look at the db!
    )

    updated_matrix_definitions, build_tasks = architect.generate_plans(matrix_set_definitions, feature_dicts)
    # test that it added uuids: we don't much care what they are
    for matrix_def in updated_matrix_definitions:
        assert isinstance(matrix_def['train_uuid'], str)
        for test_uuid in matrix_def['test_uuids']:
            assert isinstance(test_uuid, str)

    # not going to assert anything on the keys (uuids), just get out the values
    build_tasks = build_tasks.values()
    assert len(build_tasks) == 8 # 2 splits * 2 matrices per split * 2 feature dicts

    assert sum(1 for task in build_tasks if task['matrix_type'] == 'train') == 4
    assert sum(1 for task in build_tasks if task['matrix_type'] == 'test') == 4
    assert all(task for task in build_tasks if task['matrix_directory'] == '')
    assert sum(1 for task in build_tasks if task['feature_dictionary'] == feature_dict_one) == 4
    assert sum(1 for task in build_tasks if task['feature_dictionary'] == feature_dict_two) == 4


class TestBuildMatrix(object):
    def test_train_matrix(self):
        with testing.postgresql.Postgresql() as postgresql:
            # create an engine and generate a table with fake feature data
            engine = create_engine(postgresql.url())
            create_features_and_labels_schemas(engine, features_tables, labels)

            dates = [datetime.datetime(2016, 1, 1, 0, 0),
                     datetime.datetime(2016, 2, 1, 0, 0),
                     datetime.datetime(2016, 3, 1, 0, 0)]

            with TemporaryDirectory() as temp_dir:
                matrix_maker = Architect(
                    beginning_of_time = datetime.datetime(2010, 1, 1, 0, 0),
                    label_names = ['booking'],
                    label_types = ['binary'],
                    db_config = db_config,
                    matrix_directory = temp_dir,
                    user_metadata = {},
                    engine = engine
                )
                feature_dictionary = {
                    'features0': ['f1', 'f2'],
                    'features1': ['f1', 'f2'],
                }
                matrix_metadata = {
                    'matrix_id': 'hi',
                    'label_name': 'booking',
                    'end_time': datetime.datetime(2016, 3, 1, 0, 0),
                    'start_time': datetime.datetime(2016, 1, 1, 0, 0),
                    'prediction_window': '1d'
                }
                uuid = metta.generate_uuid(matrix_metadata)
                matrix_maker.build_matrix(
                    as_of_times = dates,
                    label_name = 'booking',
                    label_type = 'binary',
                    feature_dictionary = feature_dictionary,
                    matrix_directory = temp_dir,
                    matrix_metadata = matrix_metadata,
                    matrix_uuid = uuid,
                    matrix_type = 'train'
                )

                matrix_filename = os.path.join(
                    temp_dir,
                    '{}.csv'.format(uuid)
                )
                with open(matrix_filename, 'r') as f:
                    reader = csv.reader(f)
                    assert(len([row for row in reader]) == 12)

    def test_test_matrix(self):
        with testing.postgresql.Postgresql() as postgresql:
            # create an engine and generate a table with fake feature data
            engine = create_engine(postgresql.url())
            create_features_and_labels_schemas(engine, features_tables, labels)

            dates = [datetime.datetime(2016, 1, 1, 0, 0),
                     datetime.datetime(2016, 2, 1, 0, 0),
                     datetime.datetime(2016, 3, 1, 0, 0)]

            with TemporaryDirectory() as temp_dir:
                matrix_maker = Architect(
                    beginning_of_time = datetime.datetime(2010, 1, 1, 0, 0),
                    label_names = ['booking'],
                    label_types = ['binary'],
                    db_config = db_config,
                    matrix_directory = temp_dir,
                    user_metadata = {},
                    engine = engine
                )

                matrix_dates = {
                    'matrix_start_time': datetime.datetime(2016, 1, 1, 0, 0),
                    'matrix_end_time': datetime.datetime(2016, 3, 1, 0, 0),
                    'as_of_times': dates
                }
                feature_dictionary = {
                    'features0': ['f1', 'f2'],
                    'features1': ['f1', 'f2'],
                }
                matrix_metadata = {
                    'matrix_id': 'hi',
                    'label_name': 'booking',
                    'end_time': datetime.datetime(2016, 3, 1, 0, 0),
                    'start_time': datetime.datetime(2016, 1, 1, 0, 0),
                    'prediction_window': '1d'
                }
                uuid = metta.generate_uuid(matrix_metadata)
                matrix_maker.build_matrix(
                    as_of_times = dates,
                    label_name = 'booking',
                    label_type = 'binary',
                    feature_dictionary = feature_dictionary,
                    matrix_directory = temp_dir,
                    matrix_metadata = matrix_metadata,
                    matrix_uuid = uuid,
                    matrix_type = 'test'
                )
                print(os.listdir(temp_dir))
                matrix_filename = os.path.join(
                    temp_dir,
                    '{}.csv'.format(uuid)
                )

                with open(matrix_filename, 'r') as f:
                    reader = csv.reader(f)
                    assert(len([row for row in reader]) == 13)

    def test_replace(self):
        with testing.postgresql.Postgresql() as postgresql:
            # create an engine and generate a table with fake feature data
            engine = create_engine(postgresql.url())
            create_features_and_labels_schemas(engine, features_tables, labels)

            dates = [datetime.datetime(2016, 1, 1, 0, 0),
                     datetime.datetime(2016, 2, 1, 0, 0),
                     datetime.datetime(2016, 3, 1, 0, 0)]

            with TemporaryDirectory() as temp_dir:
                matrix_maker = Architect(
                    beginning_of_time = datetime.datetime(2010, 1, 1, 0, 0),
                    label_names = ['booking'],
                    label_types = ['binary'],
                    db_config = db_config,
                    matrix_directory = temp_dir,
                    user_metadata = {},
                    engine = engine,
                    replace=False
                )

                matrix_dates = {
                    'matrix_start_time': datetime.datetime(2016, 1, 1, 0, 0),
                    'matrix_end_time': datetime.datetime(2016, 3, 1, 0, 0),
                    'as_of_times': dates
                }
                feature_dictionary = {
                    'features0': ['f1', 'f2'],
                    'features1': ['f1', 'f2'],
                }

                uuid = metta.generate_uuid(matrix_metadata)
                matrix_maker.build_matrix(
                    as_of_times = dates,
                    label_name = 'booking',
                    label_type = 'binary',
                    feature_dictionary = feature_dictionary,
                    matrix_directory = temp_dir,
                    matrix_metadata = {
                        'matrix_id': 'hi',
                        'label_name': 'booking',
                        'end_time': datetime.datetime(2016, 3, 1, 0, 0),
                        'start_time': datetime.datetime(2016, 1, 1, 0, 0),
                        'prediction_window': '1d'
                    },
                    matrix_uuid = uuid,
                    matrix_type = 'test'
                )

                matrix_filename = os.path.join(
                    temp_dir,
                    '{}.csv'.format(uuid)
                )

                with open(matrix_filename, 'r') as f:
                    reader = csv.reader(f)
                    assert(len([row for row in reader]) == 13)

                # rerun
                matrix_maker.make_entity_date_table = Mock()
                matrix_maker.build_matrix(
                    as_of_times = dates,
                    label_name = 'booking',
                    label_type = 'binary',
                    feature_dictionary = feature_dictionary,
                    matrix_directory = temp_dir,
                    matrix_metadata = {
                        'matrix_id': 'hi',
                        'label_name': 'booking',
                        'end_time': datetime.datetime(2016, 3, 1, 0, 0),
                        'start_time': datetime.datetime(2016, 1, 1, 0, 0),
                        'prediction_window': '1d'
                    },
                    matrix_uuid = uuid,
                    matrix_type = 'test'
                )
                assert not matrix_maker.make_entity_date_table.called
