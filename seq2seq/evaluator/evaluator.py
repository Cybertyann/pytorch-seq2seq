from __future__ import print_function, division

import os
import io
import numpy as np
from sklearn.metrics import f1_score

import torch
import torchtext

import seq2seq
from seq2seq.loss import NLLLoss

class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), batch_size=64):
        self.loss = loss
        self.batch_size = batch_size

    def iobes_iob(self, tags):
        """
        IOBES -> IOB
        """
        new_tags = []
        # print(tags)
        for i, tag in enumerate(tags):
            if tag.split('-')[0] == 'B':
                new_tags.append(tag)
            elif tag.split('-')[0] == 'I':
                new_tags.append(tag)
            elif tag.split('-')[0] == 'S':
                new_tags.append(tag.replace('S-', 'B-'))
            elif tag.split('-')[0] == 'E':
                new_tags.append(tag.replace('E-', 'I-'))
            elif tag.split('-')[0] == 'O':
                new_tags.append(tag)
            elif tag == '<eos>':
                new_tags.append('O')
            # elif tag == '<eos>':
            #     new_tags.append(tag)
            else:
                raise Exception('Invalid format!')
        # print(new_tags)
        return new_tags

    def print_confusion_matrix(self, id_to_tag, count, task_name):
        n_tags = len(id_to_tag)
        # Confusion matrix with accuracy for each tag
        print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_tags)).format(
            "ID", task_name, "Total", * ([id_to_tag[i] for i in xrange(n_tags)] + ["Percent"])))
        for i in xrange(n_tags):
            if count[i].sum() == 0:
                continue
            print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_tags)).format(
                str(i), id_to_tag[i], str(count[i].sum()),
                *([count[i][j] for j in xrange(n_tags)] +
                  ["%.3f" % (count[i][i] * 100. / max(1, count[i].sum()))])))
        print("%i/%i (%.5f%%)" % (count.trace(), count.sum(), 100. * count.trace() / max(1, count.sum())))

    def evaluate_conll(self, predictions, eval_temp='tmp/', eval_script='data/conlleval'):
        # Write predictions to disk and run CoNLL script externally
        eval_id = np.random.randint(1000000, 2000000)
        output_path = os.path.join(eval_temp, "eval.nll.%i.output" % (eval_id))
        scores_path = os.path.join(eval_temp, "eval.nll.%i.scores" % (eval_id))
        with io.open(output_path, 'w', encoding='utf8') as f:
            f.write("\n".join(predictions))
        os.system("%s < %s > %s" % (eval_script, output_path, scores_path))

        # CoNLL evaluation results
        eval_lines = [l.rstrip() for l in io.open(scores_path, 'r', encoding='utf8')]
        for line in eval_lines:
            print(line)
        # Remove temp files
        # os.remove(output_path)
        # os.remove(scores_path)
        return eval_lines

    def evaluate(self, model, data):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()

        loss = self.loss
        loss.reset()
        match = 0
        total = 0

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort_key=lambda batch: -len(getattr(batch, seq2seq.src_field_name)),
            device=device, train=False)
        tgt_vocab = data.fields[seq2seq.tgt_field_name].vocab
        pad = tgt_vocab.stoi[data.fields[seq2seq.tgt_field_name].pad_token]

        all_indices = []
        all_predictions = []
        all_groundtruths = []

        # print(data)
        # List of str
        # print(tgt_vocab.itos)
        # Counter: str to int
        # print(tgt_vocab.freqs)
        # defaultdict: str to int
        # print(tgt_vocab.stoi)
        src_vocab = data.fields[seq2seq.src_field_name].vocab
        # print(src_vocab.itos)
        # print(src_vocab.freqs)
        # print(src_vocab.stoi)
        # print(data.fields[seq2seq.src_field_name].vocab.data)

        n_tags = len(tgt_vocab)
        print('Number of tags = ', n_tags)
        count = np.zeros((n_tags, n_tags), dtype=np.int32)
        for batch in batch_iterator:
            input_variables, input_lengths  = getattr(batch, seq2seq.src_field_name)
            target_variables = getattr(batch, seq2seq.tgt_field_name)

            decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist(), target_variables)

            # Evaluation
            seqlist = other['sequence']
            groudtruths = []
            predictions = []
            for step, step_output in enumerate(decoder_outputs):
                target = target_variables[:, step + 1]
                loss.eval_batch(step_output.view(target_variables.size(0), -1), target)

                non_padding = target.ne(pad)
                correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().data[0]

                # print(seqlist[step].view(-1).data.tolist())
                # print(target.data.tolist())
                # print(tgt_vocab.itos[target.data.tolist()])
                # print(seqlist[step].view(-1), target)
                # print(step_output)
                # print(src)

                y_true = target.data.tolist()
                y_pred = seqlist[step].view(-1).data.tolist()
                groudtruths.append(y_true)
                predictions.append(y_pred)
                match += correct
                total += non_padding.sum().data[0]

            for b, l in enumerate(input_lengths.tolist()):
                # print(b, l)
                indices = input_variables[b, 0:l].data.tolist()
                all_indices.append(indices)
                new_predictions = []
                new_groundtruths = []
                # for j in xrange(l+1):
                for j in xrange(l):
                    pred = predictions[j][b]
                    gt = groudtruths[j][b]
                    new_predictions.append(pred)
                    new_groundtruths.append(gt)
                    count[gt, pred] += 1
                all_predictions.append(new_predictions)
                all_groundtruths.append(new_groundtruths)
        # print(all_indices[0:50], all_predictions[0:50], all_groundtruths[0:50])
        predictions_str = []
        for sent, tags, gt_tags in zip(all_indices, all_predictions, all_groundtruths):
            str_sent = [src_vocab.itos[i] for i in sent]
            str_tags = [tgt_vocab.itos[i] for i in tags]
            str_tags = self.iobes_iob(str_tags)
            str_gt_tags = [tgt_vocab.itos[i] for i in gt_tags]
            str_gt_tags = self.iobes_iob(str_gt_tags)
            # print('sentence: ', ' '.join(str_sent))
            # print('pred    : ', ' '.join(str_tags))
            # print('real    : ', ' '.join(str_gt_tags))
            # print('')
            for k in xrange(len(str_sent)):
                predictions_str.append(' '.join([str_sent[k], str_gt_tags[k], str_tags[k]]))
            predictions_str.append('')
        eval_lines = self.evaluate_conll(predictions_str)
        self.print_confusion_matrix(tgt_vocab.itos, count, "NE")
        f1s = float(eval_lines[1].strip().split()[-1])
        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total
        print('F1 score:', f1s)
        print('Accuracy:', accuracy)
        # return loss.get_loss(), accuracy
        return loss.get_loss(), f1s
