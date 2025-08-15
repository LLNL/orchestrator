from orchestrator.test.simulator.simulator_unit_testers import (
    simulator_unit_test, )
import pytest

tests_ran = [False] * 1

tests_ran[0] = simulator_unit_test('test_inputs/01_lammps_test_input.json')

# now validate the tests:
validation_tests = ['test_lammps']
test_strings = []
for ran, test in zip(tests_ran, validation_tests):
    if ran:
        test_strings.append(f'test_simulator.py::{test}')
    else:
        print(f'{test} was not run or did not complete, omitting from pytest')
if len(test_strings) > 0:
    _ = pytest.main(['-v'] + test_strings)
else:
    print('No tests ran successfully, skipping pytest validation')
