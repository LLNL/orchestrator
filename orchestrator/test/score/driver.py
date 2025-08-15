from orchestrator.test.score.score_unit_testers import (
    atom_level_score_unit_test,
    atom_level_score_unit_test_with_in_memory_arrays,
    dataset_level_score_unit_test,
    model_level_score_unit_test,
)
import pytest

tests_ran = [False] * 13

tests_ran[0] = atom_level_score_unit_test('test_inputs/01_ltau.json')
tests_ran[1] = atom_level_score_unit_test('test_inputs/02_ltau_batched.json')
tests_ran[2] = atom_level_score_unit_test('test_inputs/03_ltau_colabfit.json')
tests_ran[3] = atom_level_score_unit_test_with_in_memory_arrays(
    'test_inputs/04_ltau_in_memory.json')
tests_ran[4] = dataset_level_score_unit_test(
    'test_inputs/05_quests_efficiency.json')
tests_ran[5] = dataset_level_score_unit_test(
    'test_inputs/06_quests_diversity.json')
tests_ran[6] = atom_level_score_unit_test(
    'test_inputs/07_quests_delta_entropy.json')
tests_ran[7] = atom_level_score_unit_test(
    'test_inputs/08_quests_delta_entropy_batched.json')
tests_ran[8] = atom_level_score_unit_test_with_in_memory_arrays(
    'test_inputs/09_quests_delta_entropy_in_memory.json')
tests_ran[9] = atom_level_score_unit_test(
    'test_inputs/10_fim_training_set.json')
tests_ran[10] = model_level_score_unit_test('test_inputs/11_fim_property.json')
tests_ran[11] = model_level_score_unit_test(
    'test_inputs/12_fim_property_kimrun.json')
tests_ran[12] = atom_level_score_unit_test('test_inputs/13_fim_matching.json')

# now validate the tests:
validation_tests = [
    'test_score_ltau_local',
    'test_score_ltau_batched',
    'test_score_ltau_colabfit',
    'test_score_ltau_in_memory',  # in-memory version
    'test_score_quests_efficiency',
    'test_score_quests_diversity',
    'test_score_quests_delta_entropy_local',
    'test_score_quests_delta_entropy_batched',
    'test_score_quests_delta_entropy_in_memory',  # in-memory version
    'test_score_fim_training_set',
    'test_score_fim_property',
    'test_score_fim_property_kimrun',
    'test_score_fim_matching',
]
test_strings = []
for ran, test in zip(tests_ran, validation_tests):
    if ran:
        test_strings.append(f'test_score.py::{test}')
    else:
        print(f'{test} was not run or did not complete, omitting from pytest')
if len(test_strings) > 0:
    _ = pytest.main(['-v'] + test_strings)
else:
    print('No tests ran successfully, skipping pytest validation')
