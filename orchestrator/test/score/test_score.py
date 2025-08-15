import pytest
from orchestrator.test.reference_output.compare_outputs import (
    compare_outputs, compare_atoms_info, compare_json)

ref_dir = 'TEST_PATH/reference_output'
test_dir = 'INSTALL_PATH/'


@pytest.mark.parametrize('job_id', range(9))
def test_score_ltau_local(job_id):
    """
    Tests LTAU score evaluation and storage in the Local storage backend.

    In this test the LTAUForcesUQScore is used to compute a score for a
    small selection of atomic configurations.

    :param job_id: id of the score run to test, set by pytest parametrize
    :type job_id: int
    """
    ref_file = (f'{ref_dir}/score/LTAUForcesUQScore/01_ltau/'
                f'{job_id:05}/saved_config.extxyz')
    test_file = (f'{test_dir}/score/computer/LTAUForcesUQScore/01_ltau/'
                 f'{job_id:05}/saved_config.extxyz')

    compare_outputs(ref_file, test_file)


@pytest.mark.parametrize('job_id', range(2))
def test_score_ltau_batched(job_id):
    """
    Tests LTAU score evaluation and storage in the Local storage backend.

    In this test the LTAUForcesUQScore is used to compute a score for a
    small selection of atomic configurations.

    :param job_id: id of the score run to test, set by pytest parametrize
    :type job_id: int
    """
    ref_file = (f'{ref_dir}/score/LTAUForcesUQScore/02_ltau_batched/'
                f'{job_id:05}/saved_config.extxyz')
    test_file = (
        f'{test_dir}/score/computer/LTAUForcesUQScore/02_ltau_batched/'
        f'{job_id:05}/saved_config.extxyz')

    compare_outputs(ref_file, test_file)


@pytest.mark.parametrize('job_id', range(9))
def test_score_ltau_colabfit(job_id):
    """
    Tests LTAU score evaluation and storage in the ColabFit storage backend.

    In this test the LTAUForcesUQScore is used to compute a score for a
    small selection of atomic configurations. Uses batched evaluation.

    :param job_id: id of the score run to test, set by pytest parametrize
    :type job_id: int
    """
    ref_file = (f'{ref_dir}/score/LTAUForcesUQScore/'
                f'03_ltau_colabfit/{job_id:05}/saved_config.extxyz')
    test_file = (f'{test_dir}/score/computer/LTAUForcesUQScore/'
                 f'03_ltau_colabfit/{job_id:05}/saved_config.extxyz')

    compare_outputs(ref_file, test_file)


@pytest.mark.parametrize('job_id', range(9))
def test_score_ltau_in_memory(job_id):
    """
    Tests LTAU score evaluation and storage in the ColabFit storage backend.

    In this test the LTAUForcesUQScore is used to compute a score for a
    small selection of atomic configurations. Uses in-memory evaluation.

    :param job_id: id of the score run to test, set by pytest parametrize
    :type job_id: int
    """
    ref_file = (f'{ref_dir}/score/LTAUForcesUQScore/'
                f'04_ltau_in_memory/{job_id:05}/saved_config.extxyz')
    test_file = (f'{test_dir}/score/computer/LTAUForcesUQScore/'
                 f'04_ltau_in_memory/{job_id:05}/saved_config.extxyz')

    compare_outputs(ref_file, test_file)


@pytest.mark.parametrize('job_id', range(1))
def test_score_quests_efficiency(job_id):
    """
    Tests QUESTS efficiency score evaluation.

    In this test the QUESTSEfficiencyScore is used to compute a score for a
    single dataset.

    :param job_id: id of the score run to test, set by pytest parametrize
    :type job_id: int
    """
    ref_file = (f'{ref_dir}/score/QUESTSEfficiencyScore/05_quests_efficiency/'
                f'{job_id:05}/score_results.json')
    test_file = (f'{test_dir}/score/computer/QUESTSEfficiencyScore/'
                 f'05_quests_efficiency/{job_id:05}/score_results.json')

    compare_outputs(ref_file, test_file)


@pytest.mark.parametrize('job_id', range(1))
def test_score_quests_diversity(job_id):
    """
    Tests QUESTS diversity score evaluation.

    In this test the QUESTSDiversityScore is used to compute a score for a
    single dataset.

    :param job_id: id of the score run to test, set by pytest parametrize
    :type job_id: int
    """
    ref_file = (f'{ref_dir}/score/QUESTSDiversityScore/06_quests_diversity/'
                f'{job_id:05}/score_results.json')
    test_file = (f'{test_dir}/score/computer/QUESTSDiversityScore/'
                 f'06_quests_diversity/{job_id:05}/score_results.json')

    compare_outputs(ref_file, test_file)


@pytest.mark.parametrize('job_id', range(9))
def test_score_quests_delta_entropy_local(job_id):
    """
    Tests QUESTS deltaH score evaluation.

    In this test the QUESTSDeltaEntropyScore is used to compute a score for a
    small selection of atomic configurations.

    :param job_id: id of the score run to test, set by pytest parametrize
    :type job_id: int
    """
    ref_file = (f'{ref_dir}/score/QUESTSDeltaEntropyScore/'
                f'07_quests_delta_entropy/{job_id:05}/saved_config.extxyz')
    test_file = (f'{test_dir}/score/computer/QUESTSDeltaEntropyScore/'
                 f'07_quests_delta_entropy/{job_id:05}/saved_config.extxyz')

    compare_outputs(ref_file, test_file)


@pytest.mark.parametrize('job_id', range(2))
def test_score_quests_delta_entropy_batched(job_id):
    """
    Tests QUESTS deltaH score evaluation.

    In this test the QUESTSDeltaEntropyScore is used to compute a score for a
    small selection of atomic configurations.

    :param job_id: id of the score run to test, set by pytest parametrize
    :type job_id: int
    """
    ref_file = (f'{ref_dir}/score/QUESTSDeltaEntropyScore/'
                '08_quests_delta_entropy_batched/'
                f'{job_id:05}/saved_config.extxyz')
    test_file = (f'{test_dir}/score/computer/QUESTSDeltaEntropyScore/'
                 '08_quests_delta_entropy_batched'
                 f'/{job_id:05}/saved_config.extxyz')

    compare_outputs(ref_file, test_file)


@pytest.mark.parametrize('job_id', range(9))
def test_score_quests_delta_entropy_in_memory(job_id):
    """
    Tests QUESTS deltaH score evaluation.

    In this test the QUESTSDeltaEntropyScore is used to compute a score for a
    small selection of atomic configurations.

    :param job_id: id of the score run to test, set by pytest parametrize
    :type job_id: int
    """
    ref_file = (f'{ref_dir}/score/QUESTSDeltaEntropyScore/'
                '09_quests_delta_entropy_in_memory/'
                f'{job_id:05}/saved_config.extxyz')
    test_file = (f'{test_dir}/score/computer/QUESTSDeltaEntropyScore/'
                 '09_quests_delta_entropy_in_memory'
                 f'/{job_id:05}/saved_config.extxyz')

    compare_outputs(ref_file, test_file)


@pytest.mark.parametrize('job_id', range(9))
def test_score_fim_training_set(job_id):
    """
    Tests FIMTrain score evaluation.

    In this test the FIMTrainingSetScore is used to compute a score for a
    small selection of atomic configurations.

    :param job_id: id of the score run to test, set by pytest parametrize
    :type job_id: int
    """
    ref_file = (f'{ref_dir}/score/FIMTrainingSetScore/10_fim_training_set/'
                f'{job_id:05}/saved_config.extxyz')
    test_file = (
        f'{test_dir}/score/computer/FIMTrainingSetScore/10_fim_training_set/'
        f'{job_id:05}/saved_config.extxyz')

    compare_atoms_info(ref_file, test_file, 'fim_training_set_score')


@pytest.mark.parametrize('job_id', range(1))
def test_score_fim_property(job_id):
    """
    Tests FIMProperty score evaluation.

    In this test the FIMPropertyScore is used to compute a score for target
    property ElasticConstants.

    :param job_id: id of the score run to test, set by pytest parametrize
    :type job_id: int
    """
    ref_file = (f'{ref_dir}/score/FIMPropertyScore/11_fim_property/'
                f'{job_id:05}/score_results.json')
    test_file = (f'{test_dir}/score/computer/FIMPropertyScore/11_fim_property/'
                 f'{job_id:05}/score_results.json')

    compare_json(ref_file, test_file, 'fim_property_score')


@pytest.mark.parametrize('job_id', range(1))
def test_score_fim_property_kimrun(job_id):
    """
    Tests FIMProperty score evaluation with KIMRUn target property.

    In this test the FIMPropertyScore is used to compute a score for elastic
    constants KIM test, where the calculation is done via KIMRun.

    :param job_id: id of the score run to test, set by pytest parametrize
    :type job_id: int
    """
    ref_file = (f'{ref_dir}/score/FIMPropertyScore/12_fim_property_kimrun/'
                f'{job_id:05}/score_results.json')
    test_file = (
        f'{test_dir}/score/computer/FIMPropertyScore/12_fim_property_kimrun/'
        f'{job_id:05}/score_results.json')

    compare_json(ref_file, test_file, 'fim_property_score')


@pytest.mark.parametrize('job_id', range(1))
def test_score_fim_matching(job_id):
    """
    Tests FIMMatching score evaluation.

    In this test the FIMMatchingScore is used to compute importance score via
    information-matching method for an example problem.

    :param job_id: id of the score run to test, set by pytest parametrize
    :type job_id: int
    """
    ref_file = (f'{ref_dir}/score/FIMMatchingScore/13_fim_matching/'
                f'{job_id:05}/saved_config.extxyz')
    test_file = (f'{test_dir}/score/computer/FIMMatchingScore/13_fim_matching/'
                 f'{job_id:05}/saved_config.extxyz')

    compare_atoms_info(ref_file, test_file, 'fim_matching_weight_score')
