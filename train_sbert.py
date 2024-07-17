'''
Running this script:
python train_sbert.py
'''

from sentence_transformers import losses, models, SentenceTransformer
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib, os
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

dataset = "ToolLens"
data_path = "./datasets/ToolLens"
corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")
dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="test")
model_name="./PLMs/contriever-base-msmarco"

word_embedding_model = models.Transformer(model_name, max_seq_length=350)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


retriever = TrainRetriever(model=model, batch_size=16)

train_samples = retriever.load_train(corpus, queries, qrels)
train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)

ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)
model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "{}-{}".format(model_name, dataset))
os.makedirs(model_save_path, exist_ok=True)

num_epochs = 5
evaluation_steps = 5000
warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)

retriever.fit(train_objectives=[(train_dataloader, train_loss)], 
                evaluator=ir_evaluator, 
                epochs=num_epochs,
                output_path=model_save_path,
                warmup_steps=warmup_steps,
                evaluation_steps=evaluation_steps,
                use_amp=True)
