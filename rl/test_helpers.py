import unittest
from control_unit import helpers


class TestFunction(unittest.TestCase):
    def test_mean_array(self):
        arr = [[10, 1], [30, 2]]
        self.assertEqual(helpers.mean_array(arr).tolist(), [20, 1.5])  # add assertion here


if __name__ == '__main__':
    unittest.main()
