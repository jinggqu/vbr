from config.config import CFG
from model.model import LSTM

if __name__ == '__main__':
    # Initialize model
    model = LSTM(CFG)

    # Load data
    model.load_data()

    # Build model
    model.build()

    # Train model
    model.train()
