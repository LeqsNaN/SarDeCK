import os

import torch


class InputExample(object):
    def __init__(self, text, data_id, knowledge=None, label=None):
        """Constructs an InputExample."""
        self.text = text
        self.knowledge = knowledge
        self.data_id = data_id
        self.label = label


class InputFeatures(object):
    def __init__(self, input_id,
                 input_mask,
                 label_id, ):
        self.input_id = input_id
        self.input_mask = input_mask
        self.label_id = label_id


class KnowInputFeatures(object):
    def __init__(self, input_id,
                 input_mask,
                 know_id,
                 know_mask,
                 label_id):
        self.input_id = input_id
        self.input_mask = input_mask
        self.know_id = know_id
        self.know_mask = know_mask
        self.label_id = label_id


class Processer():

    def __init__(self, data_dir, model_select, know_strategy, max_seq_length, max_know_length, know_num):
        self.model_select = model_select
        self.know_strategy = know_strategy
        self.max_seq_length = max_seq_length
        self.max_know_length = max_know_length
        path = os.path.join(data_dir, self.know_strategy)
        self.common_knowledge = self.get_knowledge_examples(path, know_num)
        self.data_dir = data_dir

    def get_train_examples(self):
        return self._create_examples(os.path.join(self.data_dir, "train.txt"))

    def get_eval_examples(self):
        return self._create_examples(os.path.join(self.data_dir, "dev.txt"))

    def get_test_examples(self):
        return self._create_examples(os.path.join(self.data_dir, "test.txt"))

    def get_labels(self):
        return [0, 1]

    def _create_examples(self, data_file):
        """Creates examples for the training and dev sets."""
        examples = []
        with open(data_file) as f:
            for line in f.readlines():
                tmpLS = line.split(" ==sep== ")
                data_id = tmpLS[0]
                text = tmpLS[1]
                label = int(tmpLS[2])
                knowledge = self.common_knowledge[data_id]
                examples.append(InputExample(text=text, data_id=data_id, knowledge=knowledge, label=label))
        return examples

    def get_knowledge_examples(self, path, know_num):
        common_data = {}
        with open(path) as f:
            for line in f.readlines():
                tmpLS = line.split(" ==sep== ")
                temp = []
                start = 2
                end = start+know_num if know_num < len(tmpLS[2:-1]) else -1
                for know in tmpLS[start:end]:
                    temp.append(know)
                common_data[tmpLS[0]] = temp
        return common_data

    def convert_examples_to_features(self, examples, label_list, tokenizer):
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []

        for (ex_index, example) in enumerate(examples):
            tokens = tokenizer.tokenize(example.text)
            if len(tokens) > self.max_seq_length - 2:
                tokens = tokens[:(self.max_seq_length - 2)]
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            input_id = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_id)
            padding = [0] * (self.max_seq_length - len(input_id))
            input_id += padding
            input_mask += padding
            assert len(input_id) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length

            knowledges = " ".join(example.knowledge)
            knowledges = tokenizer.tokenize(knowledges)
            if len(knowledges) > self.max_know_length - 2:
                knowledges = knowledges[:self.max_know_length - 2]
            knowledges = ["[CLS]"] + knowledges + ["[SEP]"]
            know_id = tokenizer.convert_tokens_to_ids(knowledges)
            know_mask = [1] * len(know_id)
            padding = [0] * (self.max_know_length - len(know_id))
            know_id += padding
            know_mask += padding
            assert len(know_id) == self.max_know_length
            assert len(know_mask) == self.max_know_length
            label_id = label_map[example.label]

            features.append(KnowInputFeatures(input_id=input_id, input_mask=input_mask, know_id=know_id,
                                              know_mask=know_mask, label_id=label_id))
        print('the number of examples: ' + str(len(features)))
        all_input_ids = torch.tensor([f.input_id for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        all_know_ids = torch.tensor([f.know_id for f in features], dtype=torch.long)
        all_know_mask = torch.tensor([f.know_mask for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_know_ids, all_know_mask, all_label_ids
