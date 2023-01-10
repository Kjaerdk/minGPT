"""
Trains a GPT to predict next ICD10 diagnose
"""

import os
import sys

import torch
import numpy as np
import datetime as dt

from mingpt.utils import set_seed, setup_logging, CfgNode as CN

from diagGPT.data_utils import DiagDataset, BucketBatchSampler
from diagGPT.model import GPT
from diagGPT.trainer import Trainer
from diagGPT.ICD_encoder import get_encoder


# -----------------------------------------------------------------------------

def get_config():
    C = CN()

    # data
    C.data = DiagDataset.get_default_config()

    # system
    C.system = CN()
    C.system.seed = 107
    C.system.date = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    C.system.work_dir = f'./out/diag/{C.data.which_mapping}/{C.system.date}'

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-nano'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4  # the model we're using is so small that we can go a bit faster

    return C


# -----------------------------------------------------------------------------

def make_diagnose_lists(config, n_lists: int, poisson_n_diags: int=25):
    """ Output randomly generated lists of diagnoses"""
    e = get_encoder(config.data.which_mapping)
    all_diags = list(e.encoder.keys())
    n_patient_diags = np.random.poisson(poisson_n_diags, n_lists)
    print(f"max number of diagnoses is {max(n_patient_diags)}")
    patients = []
    for p in range(n_lists):
        patient = list(np.random.choice(all_diags, size=n_patient_diags[p]))
        patients.append(patient)

    return patients  # list of lists


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    patients = make_diagnose_lists(config, n_lists=100)
    print(patients)
    print([len(p) for p in patients])

    train_dataset = DiagDataset(config.data, patients)
    batch_sampler = BucketBatchSampler(patients, config.trainer.batch_size)

    # construct the model
    config.model.vocab_size = train_dataset.get_n_diagnoses()
    config.model.block_size = train_dataset.get_block_size()

    if config.model.load_existing:
        model = GPT(config.model).load_state_dict(config)  # todo: continue train on model if workdir already exists (custom workdir can be passed in console)
    else:
        model = GPT(config.model)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset, batch_sampler)

    # todo: more evaluation of model in callback (see eval_split in adder.py for inspiration)
    # iteration callback
    def batch_end_callback(trainer):
        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                # sample from the model...
                context = ['DQ710A', 'DC183', 'DG951', 'DQ991', 'DK400']  # some random diagnoses to warm up, for now
                x = torch.tensor([train_dataset.encoder[d] for d in context], dtype=torch.long)[None, ...].to(trainer.device)
                y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
                completion = [train_dataset.decoder[int(i)] for i in y]
                print(completion)
            # save the latest model
            print("saving model")
            print(config)
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")  # todo: add more to this checkpoint path (model type, size, data etc.)
            torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()
