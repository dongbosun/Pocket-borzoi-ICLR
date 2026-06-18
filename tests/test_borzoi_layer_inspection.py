from pocketreg.borzoi.layer_inspection import identify_candidate_layers


def test_identify_candidate_layers_from_synthetic_rank3_layers():
    layers = [
        {"index": 0, "name": "input", "rank": 3, "can_intermediate_output": True, "class_name": "InputLayer", "output_shape": [None, 1024, 4]},
        {"index": 1, "name": "stem", "rank": 3, "can_intermediate_output": True, "class_name": "Conv1D", "output_shape": [None, 512, 64]},
        {"index": 2, "name": "trunk", "rank": 3, "can_intermediate_output": True, "class_name": "Conv1D", "output_shape": [None, 128, 128]},
        {"index": 3, "name": "head_input", "rank": 3, "can_intermediate_output": True, "class_name": "Conv1D", "output_shape": [None, 128, 64]},
        {"index": 4, "name": "final", "rank": 3, "can_intermediate_output": True, "class_name": "Dense", "output_shape": [None, 128, 94]},
    ]
    candidates = identify_candidate_layers(layers)
    assert candidates["final_output"]["name"] == "final"
    assert candidates["last_spatial"]["name"] == "final"
    assert candidates["penultimate_spatial"]["name"] == "head_input"
    assert candidates["head_input"]["name"] == "trunk"
