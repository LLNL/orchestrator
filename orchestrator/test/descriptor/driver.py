from orchestrator.test.descriptor.descriptor_unit_testers import (
    descriptor_unit_test, )
import pytest

tests_ran = [False] * 6

tests_ran[0] = descriptor_unit_test('test_inputs/01_local.json')
tests_ran[1] = descriptor_unit_test('test_inputs/02_colabfit.json')
tests_ran[2] = descriptor_unit_test('test_inputs/03_local_batched.json')
tests_ran[3] = descriptor_unit_test('test_inputs/04_local_quests.json')
tests_ran[4] = descriptor_unit_test('test_inputs/05_colabfit_quests.json')
tests_ran[5] = descriptor_unit_test('test_inputs/06_local_batched_quests.json')

# now validate the tests:
validation_tests = [
    'test_descriptors_local',
    'test_descriptors_colabfit',
    'test_descriptors_local_batched',
    'test_descriptors_local_quests',
    'test_descriptors_colabfit_quests',
    'test_descriptors_local_batched_quests',
]
test_strings = []
for ran, test in zip(tests_ran, validation_tests):
    if ran:
        test_strings.append(f'test_descriptor.py::{test}')
    else:
        print(f'{test} was not run or did not complete, omitting from pytest')
if len(test_strings) > 0:
    _ = pytest.main(['-v'] + test_strings)
else:
    print('No tests ran successfully, skipping pytest validation')
