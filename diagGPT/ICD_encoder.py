"""
ICD encoder. It translates ICD10 diagnoses into sequences of integers, where each integer represents an ICD10 diagnose
"""

import os
import json
import torch

# -----------------------------------------------------------------------------


class DiagTokenizer:
    def __init__(self, encoder: dict):
        # diag token encoder/decoder
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}

    def encode(self, diagnoses: list):
        """ list of diagnoses goes in, list of integers comes out """
        diag_idx = [self.encoder[diagnose] for diagnose in diagnoses]
        out = torch.tensor(diag_idx, dtype=torch.long)

        return out

    def decode(self, diag_idx: torch.tensor):
        """ list of integers comes in, list of diagnoses comes out """
        assert diag_idx.ndim == 1  # ensure a simple 1D tensor for now
        diagnoses = [self.decoder[idx] for idx in diag_idx.tolist()]

        return diagnoses


def get_encoder(which_mapping: str):
    """
    Returns an instance of the GPT diagnose encoder/decoder
    """
    assert type(which_mapping) == str, "which_mapping should be string"
    assert which_mapping.lower() in ['chapter', 'subchapter', 'category'], "value of which_mapping should be one of chapter, subchapter, category"

    # load encoder.json that has the raw mappings from diagnoses -> diag index
    encoder_local_file = os.path.join('diagGPT', 'utils/encoder.json')  # todo: one encoder per mapping
    with open(encoder_local_file, 'r') as f:
        encoder = json.load(f)
    #assert len(encoder) == xx  # number of diagnoses and tokens for if died or not died as last observation todo

    enc = DiagTokenizer(encoder)
    return enc


if __name__ == '__main__':

    print(os.getcwd())
    # Simple example of using the diagnose encoder/decoder
    diagnoses = ['DA164', 'DA00']
    e = get_encoder()
    diag_idx = e.encode(diagnoses)
    print('Original list of diagnoses:')
    print(diagnoses)
    print('Tokenized list of diagnoses:')
    print(diag_idx)
    print('decoded list of diagnoses:')
    print(e.decode(diag_idx))  # should be equal to diagnoses
