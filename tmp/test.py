import pytest

dicts = {'a' : 1,
         'b' : 2,
         'c' : 3}

params = {'a' : dicts,
          'b' : dicts,
          'c' : dicts}


keys = ['a', 'b', 'c']

class BaseTest:
    def setup_class(cls):
        print ("setup class:TestFoo")
        # Do some setup based on param

    def test_something(self, param):
        key, val = param
        assert val[key] != 0

    @pytest.mark.parametrize('m', keys)
    def test_something_else(self, param, m):
        key, val = param
        assert  val[m] != 'd'    


@pytest.mark.parametrize('param', params.iteritems())
class TestFoo(BaseTest):
    pass

