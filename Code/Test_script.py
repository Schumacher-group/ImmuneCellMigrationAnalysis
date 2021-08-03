import unittest
from pathlib import Path
from Data_cleaning import wound_locations, filelocation, reformat_file




class TestDataCleaning(unittest.TestCase):
    def test_wound_locations_true(self):
        ActualResult1, ActualResult2 = wound_locations(True)

        ExpectedResult1 = [199, 175, 170, 184, 170, 163, 184]
        ExpectedResult2 = [(353 - 234), (353 - 150), (353 - 107), (353 - 110), (353 - 238), (353 - 226), (353 - 220)]

        self.assertEqual(ActualResult1, ExpectedResult1, 'First argument wrong')
        self.assertEqual(ActualResult2, ExpectedResult2, 'Second argument wrong')

    def test_file_location(self):
        num = 1
        data_dir = Path('../data')
        ActualRest = filelocation(1)
        ExpectedResult = data_dir/f'ImageJcsvs/Control_{num}_new.csv'

        self.assertEqual(ActualRest, ExpectedResult, f'Incorrect file location called: correct location num ={num}')
    def test_reformat_file(self):
        ActualReformatedFile = reformat_file()


if __name__ == '__main__':
    unittest.main()
