from TrainingUtils import run_pipeline
import pandas as pd       

LR_PATIENCE = 100
LR = 2e-3
SCHEDULER = 'ReduceLROnPlateau'
BATCH_SIZE = 64
DEVICE = 'cuda'
MAX_EPOCHS = 100
SAVE_PATH = 'TrainedModels/Github/MixHop'
DATA_FOLDER = "data/Github"
MODEL_TYPE = "MixHop"

hidden_features = [32, 64, 128, 33, 63, 129]
n_blocks = [4, 16, 32]
powers = [[0, 1, 2], [0, 1]]

RESULT_LIST = []
RESULT_COLUMNS = ["hidden_feature", "num_blocks", "power", "F1_score", "Accuracy"]

for h_feats in hidden_features: 
    for n_block in n_blocks: 
        for power in powers: 
            if h_feats % len(power) != 0: 
                continue
            MODEL_CONFIG = {"h_feats":h_feats, "n_blocks":n_block, "powers": power}
            test_value, test_acc = run_pipeline(LR_PATIENCE, LR, SCHEDULER, 
                                                BATCH_SIZE, DEVICE, MAX_EPOCHS, DATA_FOLDER, 
                                                MODEL_TYPE, SAVE_PATH, MODEL_CONFIG)
            
            RESULT_LIST.append([h_feats, n_block, power[-1], test_value, test_acc])

result_df = pd.DataFrame(data = RESULT_LIST, columns = RESULT_COLUMNS)
result_df.to_csv("result/MixHop.csv")
