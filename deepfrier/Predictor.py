import os
import csv
import glob
import json
import gzip
import secrets

import numpy as np
import tensorflow as tf
import torch

from .utils import load_catalogue, load_FASTA, load_predicted_PDB, seq2onehot
from .layers import MultiGraphConv, GraphConv, FuncPredictor, SumPooling
from .metric import area_under_roc, area_under_prc, f1_max


class GradCAM(object):
    """
    GradCAM for protein sequences.
    [Adjusted for GCNs based on https://arxiv.org/abs/1610.02391]
    """
    def __init__(self, model, layer_name="GCNN_concatenate"):
        self.grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    def _get_gradients_and_filters(self, inputs, class_idx, use_guided_grads=False):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(inputs)
            loss = predictions[:, class_idx, 0]
        grads = tape.gradient(loss, conv_outputs)

        if use_guided_grads:
            grads = tf.cast(conv_outputs > 0, "float32")*tf.cast(grads > 0, "float32")*grads

        return conv_outputs, grads

    def _compute_cam(self, output, grad):
        weights = tf.reduce_mean(grad, axis=1)
        # perform weighted sum
        cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1).numpy()

        return cam

    def heatmap(self, inputs, class_idx, use_guided_grads=False):
        output, grad = self._get_gradients_and_filters(inputs, class_idx, use_guided_grads=use_guided_grads)
        cam = self._compute_cam(output, grad)
        heatmap = (cam - cam.min())/(cam.max() - cam.min())

        return heatmap.reshape(-1)


class Predictor(object):
    """
    Class for loading trained models and computing GO/EC predictions and class activation maps (CAMs).
    """
    def __init__(self, model_prefix, gcn=True):
        self.model_prefix = model_prefix
        self.gcn = gcn
        self._load_model()

    def _load_model(self):
        self.model = tf.keras.models.load_model(self.model_prefix + '.hdf5',
                                                custom_objects={'MultiGraphConv': MultiGraphConv,
                                                                'GraphConv': GraphConv,
                                                                'FuncPredictor': FuncPredictor,
                                                                'SumPooling': SumPooling})
        # load parameters
        with open(self.model_prefix + "_model_params.json") as json_file:
            metadata = json.load(json_file)

        self.gonames = np.asarray(metadata['gonames'])
        self.goterms = np.asarray(metadata['goterms'])
        self.thresh = 0.1*np.ones(len(self.goterms))

    def _load_cmap(self, filename, seq_dict=None, cmap_thresh=10.0):
        if filename.endswith('.pdb'):
            D, seq = load_predicted_PDB(filename)
            A = np.double(D < cmap_thresh)
        elif filename.endswith('.npz'):
            cmap = np.load(filename)
            if 'C_alpha' not in cmap:
                raise ValueError("C_alpha not in *.npz dict.")
            D = cmap['C_alpha']
            A = np.double(D < cmap_thresh)
            seq = str(cmap['seqres'])
        elif filename.endswith('.pdb.gz'):
            rnd_fn = "".join([secrets.token_hex(10), '.pdb'])
            with gzip.open(filename, 'rb') as f, open(rnd_fn, 'w') as out:
                out.write(f.read().decode())
            D, seq = load_predicted_PDB(rnd_fn)
            A = np.double(D < cmap_thresh)
            os.remove(rnd_fn)
        else:
            raise ValueError("File must be given in *.npz or *.pdb format.")

        if seq_dict is not None:
            _id = filename.split('/')[-1].split('.')[0].split('_')[0]
            if _id in seq_dict:
                seq = seq_dict[_id]
        # ##
        S = seq2onehot(seq)
        S = S.reshape(1, *S.shape)
        A = A.reshape(1, *A.shape)

        return A, S, seq

    def predict(self, test_prot, cmap_thresh=10.0, chain='query_prot'):
        print ("### Computing predictions on a single protein...")
        self.Y_hat = np.zeros((1, len(self.goterms)), dtype=float)
        self.goidx2chains = {}
        self.prot2goterms = {}
        self.data = {}
        self.test_prot_list = [chain]
        if self.gcn:
            A, S, seqres = self._load_cmap(test_prot, cmap_thresh=cmap_thresh)

            y = self.model([A, S], training=False).numpy()[:, :, 0].reshape(-1)
            self.Y_hat[0] = y
            self.prot2goterms[chain] = []
            self.data[chain] = [[A, S], seqres]
            go_idx = np.where((y >= self.thresh) == True)[0]
            for idx in go_idx:
                if idx not in self.goidx2chains:
                    self.goidx2chains[idx] = set()
                self.goidx2chains[idx].add(chain)
                self.prot2goterms[chain].append((self.goterms[idx], self.gonames[idx], float(y[idx])))
        else:
            S = seq2onehot(str(test_prot))
            S = S.reshape(1, *S.shape)
            y = self.model(S, training=False).numpy()[:, :, 0].reshape(-1)
            self.Y_hat[0] = y
            self.prot2goterms[chain] = []
            self.data[chain] = [[S], test_prot]
            go_idx = np.where((y >= self.thresh) == True)[0]
            for idx in go_idx:
                if idx not in self.goidx2chains:
                    self.goidx2chains[idx] = set()
                self.goidx2chains[idx].add(chain)
                self.prot2goterms[chain].append((self.goterms[idx], self.gonames[idx], float(y[idx])))

    def predict_from_PDB_dir(self, dir_name, labels, seq_dict=None, cmap_thresh=10.0):
        print ("### Computing predictions from directory with PDB files...")
        pdb_fn_list = glob.glob(dir_name + '/*.pdb*')
        if seq_dict is not None:
            pdb_fn_list = [pdb_fn for pdb_fn in pdb_fn_list if pdb_fn.split('/')[-1].split('.')[0].split('_')[0] in seq_dict]
        self.index_list = [int(pdb_fn.split('/')[-1].split('.')[0].split('_')[1]) for pdb_fn in pdb_fn_list]
        self.chain2path = {pdb_fn.split('/')[-1].split('.')[0]: pdb_fn for pdb_fn in pdb_fn_list}
        self.test_prot_list = list(self.chain2path.keys())
        self.Y_hat = np.zeros((len(self.test_prot_list), len(self.goterms)), dtype=float)
        self.goidx2chains = {}
        self.prot2goterms = {}
        self.data = {}
        self.preds = []
        self.targets = []

        for i, chain, index in zip(range(len(self.test_prot_list)), self.test_prot_list, self.index_list):
            try:
                A, S, seqres = self._load_cmap(self.chain2path[chain], seq_dict=seq_dict, cmap_thresh=cmap_thresh)
                print(A.shape, S.shape)
                y = self.model([A, S], training=False).numpy()[:, :, 0].reshape(-1)
                self.Y_hat[i] = y
                self.preds.append(torch.tensor(y))
                target = torch.zeros(len(self.goterms))
                for pos_task in labels[chain]:
                    target[int(pos_task)] = 1
                self.targets.append(target)
                self.prot2goterms[chain] = []
                self.data[chain] = [[A, S], seqres]
                go_idx = np.where((y >= self.thresh) == True)[0]
                for idx in go_idx:
                    if idx not in self.goidx2chains:
                        self.goidx2chains[idx] = set()
                    self.goidx2chains[idx].add(chain)
                    self.prot2goterms[chain].append((self.goterms[idx], self.gonames[idx], float(y[idx])))
                print(self.chain2path[chain])
            except:
                continue

        pred_ = torch.stack(self.preds, dim=0)
        target_ = torch.stack(self.targets, dim=0)
        print('Total: {}\tValid: {}'.format(len(self.test_prot_list), len(self.preds)))
        score = area_under_prc(pred_.flatten(), target_.flatten())
        score = score.mean()
        print("auprc: ", score)
        score = area_under_prc(pred_, target_, dim=1)
        score = score.mean()
        print("auprc (protein-centric): ", score)
        score = area_under_prc(pred_, target_, dim=0)
        score = score.mean()
        print("auprc (term-centric): ", score)
        score = area_under_roc(pred_.flatten(), target_.flatten())
        score = score.mean()
        print("auroc: ", score)
        score = area_under_roc(pred_, target_, dim=1)
        score = score.mean()
        print("auroc (protein-centric): ", score)
        score = area_under_roc(pred_, target_, dim=0)
        score = score.mean()
        print("auroc (term-centric): ", score)
        score = f1_max(pred_, target_)
        print("f1_max: ", score)

    def predict_from_catalogue(self, catalogue_fn, cmap_thresh=10.0):
        print ("### Computing predictions from catalogue...")
        self.chain2path = load_catalogue(catalogue_fn)
        self.test_prot_list = list(self.chain2path.keys())
        self.Y_hat = np.zeros((len(self.test_prot_list), len(self.goterms)), dtype=float)
        self.goidx2chains = {}
        self.prot2goterms = {}
        self.data = {}
        for i, chain in enumerate(self.test_prot_list):
            A, S, seqres = self._load_cmap(self.chain2path[chain], cmap_thresh=cmap_thresh)
            y = self.model([A, S], training=False).numpy()[:, :, 0].reshape(-1)
            self.Y_hat[i] = y
            self.prot2goterms[chain] = []
            self.data[chain] = [[A, S], seqres]
            go_idx = np.where((y >= self.thresh) == True)[0]
            for idx in go_idx:
                if idx not in self.goidx2chains:
                    self.goidx2chains[idx] = set()
                self.goidx2chains[idx].add(chain)
                self.prot2goterms[chain].append((self.goterms[idx], self.gonames[idx], float(y[idx])))

    def predict_from_fasta(self, fasta_fn, labels, split_file='', cutoff=95):
        print ("### Computing predictions from fasta...")
        self.test_prot_list, sequences = load_FASTA(fasta_fn, split_file=split_file, cutoff=cutoff)
        self.Y_hat = np.zeros((len(self.test_prot_list), len(self.goterms)), dtype=float)
        self.goidx2chains = {}
        self.prot2goterms = {}
        self.data = {}
        self.preds = []
        self.targets = []

        for i, chain in enumerate(self.test_prot_list):
            S = seq2onehot(str(sequences[i]))
            S = S.reshape(1, *S.shape)
            y = self.model(S, training=False).numpy()[:, :, 0].reshape(-1)
            self.Y_hat[i] = y
            self.preds.append(torch.tensor(y))
            target = torch.zeros(len(self.goterms))
            for pos_task in labels[chain]:
                target[int(pos_task)] = 1
            self.targets.append(target)
            self.prot2goterms[chain] = []
            self.data[chain] = [[S], str(sequences[i])]
            go_idx = np.where((y >= self.thresh) == True)[0]
            for idx in go_idx:
                if idx not in self.goidx2chains:
                    self.goidx2chains[idx] = set()
                self.goidx2chains[idx].add(chain)
                self.prot2goterms[chain].append((self.goterms[idx], self.gonames[idx], float(y[idx])))
            print(chain)

        pred_ = torch.stack(self.preds, dim=0)
        target_ = torch.stack(self.targets, dim=0)
        print('Total: {}\tValid: {}'.format(len(self.test_prot_list), len(self.preds)))
        score = area_under_prc(pred_.flatten(), target_.flatten())
        score = score.mean()
        print("auprc: ", score)
        score = area_under_prc(pred_, target_, dim=1)
        score = score.mean()
        print("auprc (protein-centric): ", score)
        score = area_under_prc(pred_, target_, dim=0)
        score = score.mean()
        print("auprc (term-centric): ", score)
        score = area_under_roc(pred_.flatten(), target_.flatten())
        score = score.mean()
        print("auroc: ", score)
        score = area_under_roc(pred_, target_, dim=1)
        score = score.mean()
        print("auroc (protein-centric): ", score)
        score = area_under_roc(pred_, target_, dim=0)
        score = score.mean()
        print("auroc (term-centric): ", score)
        score = f1_max(pred_, target_)
        print("f1_max: ", score)

    def save_predictions(self, output_fn):
        print ("### Saving predictions to *.json file...")
        # pickle.dump({'pdb_chains': self.test_prot_list, 'Y_hat': self.Y_hat, 'goterms': self.goterms, 'gonames': self.gonames}, open(output_fn, 'wb'))
        with open(output_fn, 'w') as fw:
            out_data = {'pdb_chains': self.test_prot_list,
                        'Y_hat': self.Y_hat.tolist(),
                        'goterms': self.goterms.tolist(),
                        'gonames': self.gonames.tolist()}
            json.dump(out_data, fw, indent=1)

    def export_csv(self, output_fn, verbose):
        with open(output_fn, 'w') as csvFile:
            writer = csv.writer(csvFile, delimiter=',', quotechar='"')
            writer.writerow(['### Predictions made by DeepFRI.'])
            writer.writerow(['Protein', 'GO_term/EC_number', 'Score', 'GO_term/EC_number name'])
            if verbose:
                print ('Protein', 'GO-term/EC-number', 'Score', 'GO-term/EC-number name')
            for prot in self.prot2goterms:
                sorted_rows = sorted(self.prot2goterms[prot], key=lambda x: x[2], reverse=True)
                for row in sorted_rows:
                    if verbose:
                        print (prot, row[0], '{:.5f}'.format(row[2]), row[1])
                    writer.writerow([prot, row[0], '{:.5f}'.format(row[2]), row[1]])
        csvFile.close()

    def compute_GradCAM(self, layer_name='GCNN_concatenate', use_guided_grads=False):
        print ("### Computing GradCAM for each function of every predicted protein...")
        gradcam = GradCAM(self.model, layer_name=layer_name)

        self.pdb2cam = {}
        for go_indx in self.goidx2chains:
            pred_chains = list(self.goidx2chains[go_indx])
            print ("### Computing gradCAM for ", self.gonames[go_indx], '... [# proteins=', len(pred_chains), ']')
            for chain in pred_chains:
                if chain not in self.pdb2cam:
                    self.pdb2cam[chain] = {}
                    self.pdb2cam[chain]['GO_ids'] = []
                    self.pdb2cam[chain]['GO_names'] = []
                    self.pdb2cam[chain]['sequence'] = None
                    self.pdb2cam[chain]['saliency_maps'] = []
                self.pdb2cam[chain]['GO_ids'].append(self.goterms[go_indx])
                self.pdb2cam[chain]['GO_names'].append(self.gonames[go_indx])
                self.pdb2cam[chain]['sequence'] = self.data[chain][1]
                self.pdb2cam[chain]['saliency_maps'].append(gradcam.heatmap(self.data[chain][0], go_indx, use_guided_grads=use_guided_grads).tolist())

    def save_GradCAM(self, output_fn):
        print ("### Saving CAMs to *.json file...")
        # pickle.dump(self.pdb2cam, open(output_fn, 'wb'))
        with open(output_fn, 'w') as fw:
            json.dump(self.pdb2cam, fw, indent=1)
