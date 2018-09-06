import unittest
import sys
import os
import pprint
import nbformat
from nbconvert.preprocessors import (
    ClearOutputPreprocessor, ExecutePreprocessor
)
from nbconvert.preprocessors.execute import CellExecutionError
import properties


__all__ = ['TestNotebooks']


def get_test(nbname, nbpath, timeout=600):
    """
    construct a test method based on on the notebook name, path, and
    nbconvert Preprocessor options

    ** Required Inputs **
    :param str nbname: name of the notebook (without the file extension)
    :param str nbpath: full filepath to the notebook including the '.ipynb'
                       file extension

    ** Optional Inputs **
    :param int timeout: preprocessor timeout

    ** Returns **
    :returns: test_func a function for testing the notebook using nbconvert
    """

    # use nbconvert to execute the notebook
    def test_func(self):
        cwd = os.getcwd()
        passing = True
        print(
            "\n---------------------"
            " Testing {0}.ipynb "
            "---------------------".format(nbname)
        )

        if (
            (nbname in self.ignore) or
            (nbname in self.py2_ignore and sys.version_info[0] == 2)
        ):
            print(" Skipping {}".format(nbname))
            return

        run_path = os.path.sep.join(nbpath.split(os.path.sep)[:-1])
        os.chdir(run_path)
        ep = ClearOutputPreprocessor(
            resources={'metadata': {'path': run_path}}
        )

        with open(nbpath) as f:
            nb = nbformat.read(f, as_version=4)

            ep.preprocess(nb, {})

            ex = ExecutePreprocessor(
                timeout=timeout,
                kernel_name='python{}'.format(sys.version_info[0]),
                allow_errors=True,
                resources={'metadata': {'path': run_path}}
            )

            out = ex.preprocess(nb, {})
            os.chdir(cwd)

            for cell in out[0]['cells']:
                if 'outputs' in cell.keys():
                    for output in cell['outputs']:
                        if output['output_type'] == 'error':
                            passing = False

                            err_msg = []
                            for o in output['traceback']:
                                err_msg += ["{}".format(o)]
                            err_msg = "\n".join(err_msg)

                            msg = """
\n ... {} FAILED \n
{} in cell [{}] \n-----------\n{}\n-----------\n
                            """.format(
                                nbname, output['ename'],
                                cell['execution_count'], cell['source'],
                            )

                            traceback = """
----------------- >> begin Traceback << ----------------- \n
{}\n
\n----------------- >> end Traceback << -----------------\n
                            """.format(err_msg)

                            print(u"{}".format(msg + traceback))

                            assert passing, msg

            print("   ... {0} Passed \n".format(nbname))

    return test_func


class TestNotebooks(properties.HasProperties, unittest.TestCase):

    _name = properties.String(
        "test name",
        default = "NbTestCase"
    )

    directory = properties.String(
        "directory where the notebooks are stored",
        required=True,
        default='.'
    )

    ignore = properties.List(
        "list of notebooks to ignore when testing",
        properties.String("file to ignore when testing"),
        default=[]
    )

    py2_ignore = properties.List(
        "list of notebook names to ignore if testing on python 2",
        properties.String("file to ignore in python 2"),
        default=[]
    )

    timeout = properties.Integer(
        "timeout length for the execution of the notebook",
        default=600,
        min=0
    )

    _nbpaths = properties.List(
        "paths to all of the notebooks",
        properties.String("path to notebook")
    )

    _nbnames = properties.List(
        "names of all of the notebooks without the '.ipynb' file extension",
        properties.String("name of notebook")
    )

    @properties.validator('directory')
    def _use_abspath(self, change):
        change['value'] = os.path.abspath(change['value'])

    def __init__(self, **kwargs):
        super(TestNotebooks, self).__init__(**kwargs)
        nbpaths = []  # list of notebooks, with file paths
        nbnames = []  # list of notebook names (for making the tests)

        # walk the test directory and find all notebooks
        for dirname, dirnames, filenames in os.walk(self.directory):
            for filename in filenames:
                if (
                    filename.endswith(".ipynb") and not
                    filename.endswith("-checkpoint.ipynb")
                ):
                    # get abspath of notebook
                    nbpaths.append(
                        dirname + os.path.sep + filename
                    )
                    # strip off the file extension
                    nbnames.append("".join(filename[:-6]))
        self._nbpaths = nbpaths
        self._nbnames = nbnames

    @property
    def test_dict(self):
        """
        dictionary of the name of the test (keys) and test functions (values)
        built based upon the directory provided
        """
        if getattr(self, '_test_dict', None) is None:
            tests = dict()

            # build test for each notebook
            for nb, nbpath in zip(self._nbnames, self._nbpaths):
                tests["test_"+nb] = get_test(nb, nbpath, timeout=self.timeout)
            self._test_dict = tests
        return self._test_dict

    def get_tests(self, obj=None):
        """
        Create a unittest.TestCase object to attach the unit tests to.
        """
        # create class to unit test notebooks
        if obj is None:
            obj = "{}".format(self._name)
            obj = type(obj, (unittest.TestCase,), self.test_dict)
        else:
            for key, val in self.test_dict:
                setattr(obj, key, val)
        obj.ignore = self.ignore
        obj.py2_ignore = self.py2_ignore
        return obj


    def run_tests(self):
        """
        Run the unit-tests. Returns :code:`True` if all tests were successful
        and code`False` if there was a failure.

        .. code:: python

            import nbtest
            test = nbtest.TestNotebooks(directory='./notebooks')
            passed = test.run_tests()
            assert(passed)

        """
        NbTestCase = self.get_tests()
        tests = unittest.TestSuite(map(NbTestCase, self.test_dict.keys()))
        result = unittest.TestResult()
        testRunner = unittest.TextTestRunner()
        result = testRunner.run(tests)
        return result.wasSuccessful()


