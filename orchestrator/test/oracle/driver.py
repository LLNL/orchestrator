from orchestrator.test.oracle.oracle_unit_testers import oracle_unit_test
import pytest

tests_ran = [False] * 6

tests_ran[0] = oracle_unit_test('test_inputs/01_qe_colabfit_input.json')
tests_ran[1] = oracle_unit_test('test_inputs/02_kim_input.json')
tests_ran[2] = oracle_unit_test(
    'test_inputs/03_lammps_kim_colabfit_input.json')
tests_ran[3] = oracle_unit_test(
    'test_inputs/04_lammps_snap_colabfit_input.json')
tests_ran[4] = oracle_unit_test('test_inputs/05_aiida_vasp_input.json')
tests_ran[5] = oracle_unit_test('test_inputs/06_aiida_qe_input.json')

# now validate the tests:
validation_tests = [
    'test_espresso',
    'test_kim',
    'test_lammps_kim',
    'test_lammps_snap',
    'test_aiida_vasp',
    'test_aiida_qe',
]
test_strings = []
for ran, test in zip(tests_ran, validation_tests):
    if ran:
        test_strings.append(f'test_oracle.py::{test}')
    else:
        print(f'{test} was not run or did not complete, omitting from pytest')
if len(test_strings) > 0:
    _ = pytest.main(['-v'] + test_strings)
else:
    print('No tests ran successfully, skipping pytest validation')
