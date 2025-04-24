from trainer import Trainer

if __name__ == '__main__':
    trainer = Trainer("vinai/phobert-base", "E:/Agentic AI/dataset/data", save_path="E:/Agentic AI/dataset/training_crossencoder/model")
    trainer.train()