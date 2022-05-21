from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import argparse
import transformers
import torch
from transformers import Trainer, TrainingArguments
from transformers import HfArgumentParser
from datasets import load_metric
import numpy as np

def arg_parse():
    parser = argparse.ArgumentParser(
            description = "Training for NLP")
    parser.add_argument('--restore-file', type = str, default=None)
    parser.add_argument('--valid-subset', type = str, default=None)
    parser.add_argument('--max-positions', type = int, default = 512)
    parser.add_argument('--max-sentences', type = int, default = 16)
    parser.add_argument('--max-tokens', type = int, default = 4400)
    parser.add_argument('--criterion', type = str, default = 'sentence_prediction')
    parser.add_argument('--reset-optimizer', action = 'store_true')
    parser.add_argument('--reset-dataloader', action = 'store_true')
    parser.add_argument('--reset-meters', action = 'store_true')
    parser.add_argument('--required-batch-size-multiple', type = int, default = 1)
    parser.add_argument('--init-token', type = int, default = 0)
    parser.add_argument('--num-classes', type = int, default = 2)
    parser.add_argument('--optimizer', type = str, default = 'adam')
    parser.add_argument('--adam-eps', type = float, default = 1e-06)
    parser.add_argument('--clip-norm', type = float, default = 0.0)
    parser.add_argument('--lr-scheduler', type= str, default = "polynomial_decay")
    parser.add_argument('--total-num-update', type= int, default = 5336)
    parser.add_argument('--warmup-update', type= int, default = 320)
    parser.add_argument('--max-epoch', type = int, default = 12)
    parser.add_argument('--find-unused-parameters', action = 'store_true')
    parser.add_argument('--best-checkpoint-metric', type = str, default = "accuracy")
    parser.add_argument('--save-dir', type = str, default = "./outputs")
    parser.add_argument('--log-file', type = str, default = "./outputs")
    
    parser.add_argument('--quant-mode', type = str, default = "none")
    parser.add_argument('--attention-dropout', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default = 0.1)
    parser.add_argument('--weight-decay', type=float, default = 0.1)
    parser.add_argument('--lr', type = float, default = None)
    parser.add_argument('--arch', type = str, default = 'roberta_base',
            choices=['roberta_base', 'roberta_large'],
            help = 'model_architecture')
    parser.add_argument('--task', type = str, default = 'CoLA')
    parser.add_argument('--dataset', type = str, default = 'CoLA')
    parser.add_argument('--maximize-best-checkpoint-metric', action = 'store_true')
    parser.add_argument('--separator-token', type = int, default = 2)
    parser.add_argument('--warmup-updates', type= int, default = 320)
    parser.add_argument('--force-dequant', type = str, default = "none")
    parser.add_argument('--learning-rate', '-lr', type = float, default = 1e-5)
    parser.add_argument('--adam-beta1', type = float, default = 0.9, required = False)
    parser.add_argument('--adam-beta2', type = float, default = 0.98, required = False)
    parser.add_argument('--adam-epsilon', type = float, default = 1e-06, required = False)
    parser.add_argument('--lr-scheduler-type', type = str, default = "polynomial_decay")
    parser.add_argument('--max-train-epochs', type = int, default = 12)
    parser.add_argument('--metric-for-best-model', default = "accuracy")
    args = parser.parse_args()
    args.output_dir = args.save_dir
    return args

def tokenizer(args):
    if args.arch == 'roberta_base':
        tokenizer = AutoTokenizer.from_pretrained("kssteven/ibert-roberta-base")
    return tokenizer

def prepare_dataset(args):
    if args.dataset == "CoLA":
        dataset = load_dataset("glue", "cola")
    return dataset

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding = "max_length", truncation = True)

def model(args):
    if args.arch == 'roberta_base':
        model = AutoModelForSequenceClassification.from_pretrained("kssteven/ibert-roberta-base", num_labels = args.num_classes)
    print(model.config.quant_mode)
    if args.quant_mode != 'none':
        model.quant_mode = True
    return model

def set_TrainingArguments(args):
    TrainingArgument = TrainingArguments(output_dir = args.save_dir, do_train= True,learning_rate = args.learning_rate,\
                                            weight_decay = args.weight_decay, adam_beta1 = 0.9, adam_beta2 = 0.99, 
                                            num_train_epochs = 12, lr_scheduler_type = "linear", warmup_steps =
                                            args.warmup_updates, logging_dir = args.log_file, evaluation_strategy = "epoch", logging_steps = 10,
                                            metric_for_best_model = "eval_loss", save_strategy = "epoch", load_best_model_at_end = True)
    return TrainingArgument
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric.compute(predictions = predictions, references = labels)
    metthew = cola_metric.compute(predictions = predictions, references = labels)
    return { "accuracy" : accuracy, "metthew_correlation" : metthew }

if __name__ == "__main__":
    args = arg_parse()
    tokenizer = tokenizer(args)
    dataset = prepare_dataset(args)  
    tokenized_data = dataset.map(tokenize_function, batched = True)
    #valid_tokenized_data = dataset["validation"].map(tokenize_function, batched = True)
    #print(valid_tokenized_data)
    model = model(args)
    TrainingArguments = set_TrainingArguments(args)
    metric = load_metric("accuracy")
    cola_metric = load_metric("glue", "cola")
    trainer = Trainer(
            model = model,
            args = TrainingArguments,
            train_dataset = tokenized_data['train'],
            eval_dataset = tokenized_data['validation'],
            compute_metrics = compute_metrics,
            )
    '''
    trainer = Trainer(
            model = model,
            train_dataset = tokenized_data['train'],
            eval_dataset = tokenized_data['validation'],
            compute_metrics = compute_metrics,
            )
    '''
    trainer.train()
    trainer.save_model(args.save_dir + '/best_model.ckpt')
    #Trainer = Trainer(model, TrainingArguments, train_dataset = 
