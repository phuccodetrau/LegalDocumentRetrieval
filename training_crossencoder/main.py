from trainer import Trainer

if __name__ == '__main__':
    trainer = Trainer("vinai/phobert-base", "../data", save_path="./model")
    trainer.train()