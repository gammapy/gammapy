from test import BaseTest
import pytest

params = ['D', 'E', 'F']

@pytest.mark.parametrize('n', params)
class TestBar(BaseTest):
    pass
    
