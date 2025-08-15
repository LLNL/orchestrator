from orchestrator.test.storage.storage_unit_testers import storage_unit_test

tests_ran = [False] * 2

tests_ran[0] = storage_unit_test('test_inputs/01_local.json')
tests_ran[1] = storage_unit_test('test_inputs/02_colabfit.json')

test_names = ['LocalStorage', 'ColabfitStorage']
for ran, test in zip(tests_ran, test_names):
    if ran:
        print(f'{test} test completed, output will confirm success')
    else:
        print(f'{test} test did not complete')
