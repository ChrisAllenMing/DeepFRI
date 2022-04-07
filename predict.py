import json
import argparse
from deepfrier.Predictor import Predictor


def get_all_labels(annot_file, ont='mf'):
    if ont == 'ec':
        with open(annot_file, "r") as f:
            f.readline()
            tasks = f.readline().strip().split("\t")
            task_dict = {v: k for k, v in enumerate(tasks)}
            f.readline()
            labels = {}
            for line in f:
                name, pos_tasks = line.strip().split("\t")
                pos_tasks = [task_dict[x] for x in pos_tasks.split(",")]
                labels[name] = pos_tasks
    else:
        with open(annot_file, "r") as f:
            lines = f.readlines()
            if ont == 'mf':
                idx = 1
            elif ont == 'bp':
                idx = 2
            elif ont == 'cc':
                idx = 3
            tasks = lines[(idx - 1) * 4 + 1].strip().split("\t")
            task_dict = {v: k for k, v in enumerate(tasks)}
            lines = lines[13:]
            labels = {}
            for line in lines:
                name = line.strip().split("\t")[0]
                try:
                    pos_tasks = line.strip().split("\t")[idx]
                    pos_tasks = [task_dict[x] for x in pos_tasks.split(",")]
                except:
                    pos_tasks = []
                labels[name] = pos_tasks

    return labels


def get_seq_dict(fasta_file, split_file='', cutoff=95):
    if not split_file == '':
        select_list = []
        with open(split_file, 'r') as f:
            head = f.readline().strip()
            fields = head.split(',')
            col = fields.index("<{}%".format(str(cutoff)))
            for line in f.readlines():
                line = line.strip()
                if line == '':
                    continue
                pdb_id = line.split(',')[0]
                valid = int(line.split(',')[col])
                if valid:
                    select_list.append(pdb_id)
    else:
        select_list = None

    seq_dict = {}
    f = open(fasta_file, 'r')
    for line in f.readlines():
        line = line.strip()
        if line == '':
            continue
        if line.startswith('>'):
            _id = line.replace('>', '').split(' ')[0]
            seq_dict[_id] = ''
        else:
            seq_dict[_id] += line

    if select_list is not None:
        seq_dict = {k: v for k, v in seq_dict.items() if k in select_list}

    return seq_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--seq', type=str,  help="Protein sequence to be annotated.")
    parser.add_argument('-cm', '--cmap', type=str,  help="Protein contact map to be annotated (in *npz file format).")
    parser.add_argument('-pdb', '--pdb_fn', type=str,  help="Protein PDB file to be annotated.")
    parser.add_argument('--cmap_csv', type=str,  help="Catalogue with chain to file path mapping.")
    parser.add_argument('--pdb_dir', type=str,  help="Directory with PDB files of predicted Rosetta/DMPFold structures.")
    parser.add_argument('--fasta_fn', type=str,  help="Fasta file with protein sequences.")
    parser.add_argument('--model_config', type=str, default='./trained_models/model_config.json', help="JSON file with model names.")
    parser.add_argument('-ont', '--ontology', type=str, default=['mf'], nargs='+', required=True, choices=['mf', 'bp', 'cc', 'ec'],
                        help="Gene Ontology/Enzyme Commission.")
    parser.add_argument('-o', '--output_fn_prefix', type=str, default='DeepFRI', help="Save predictions/saliency in file.")
    parser.add_argument('-v', '--verbose', help="Prints predictions.", action="store_true")
    parser.add_argument('--use_guided_grads', help="Use guided grads to compute gradCAM.", action="store_true")
    parser.add_argument('--saliency', help="Compute saliency maps for every protein and every MF-GO term/EC number.", action="store_true")
    parser.add_argument('--annot_file', type=str, help='The annotation file.')
    parser.add_argument('--fasta_file', type=str, default='', help='The fasta file for all test sequences')
    parser.add_argument('--split_file', type=str, default='', help='The split file for all test sequences')
    parser.add_argument('--cutoff', type=int, default=95, choices=[30, 40, 50, 70, 95], help='Sequence identity cutoff')
    args = parser.parse_args()

    with open(args.model_config) as json_file:
        params = json.load(json_file)

    if args.seq is not None or args.fasta_fn is not None:
        params = params['cnn']
    elif args.cmap is not None or args.pdb_fn is not None or args.cmap_csv is not None or args.pdb_dir is not None:
        params = params['gcn']
    gcn = params['gcn']
    layer_name = params['layer_name']
    models = params['models']
    labels = get_all_labels(args.annot_file, ont=args.ontology[0])
    if not args.fasta_file == '':
        seq_dict = get_seq_dict(args.fasta_file, split_file=args.split_file, cutoff=args.cutoff)
    else:
        seq_dict = None

    for ont in args.ontology:
        predictor = Predictor(models[ont], gcn=gcn)
        if args.seq is not None:
            predictor.predict(args.seq)
        if args.cmap is not None:
            predictor.predict(args.cmap)
        if args.pdb_fn is not None:
            predictor.predict(args.pdb_fn)
        if args.fasta_fn is not None:
            predictor.predict_from_fasta(args.fasta_fn, labels, split_file=args.split_file, cutoff=args.cutoff)
        if args.cmap_csv is not None:
            predictor.predict_from_catalogue(args.cmap_csv)
        if args.pdb_dir is not None:
            predictor.predict_from_PDB_dir(args.pdb_dir, labels, seq_dict=seq_dict)

        # save predictions
        # predictor.export_csv(args.output_fn_prefix + "_" + ont.upper() + "_predictions.csv", args.verbose)
        # predictor.save_predictions(args.output_fn_prefix + "_" + ont.upper() + "_pred_scores.json")
        #
        # # save saliency maps
        # if args.saliency and ont in ['mf', 'ec']:
        #     predictor.compute_GradCAM(layer_name=layer_name, use_guided_grads=args.use_guided_grads)
        #     predictor.save_GradCAM(args.output_fn_prefix + "_" + ont.upper() + "_saliency_maps.json")
