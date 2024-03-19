from TrainingUtils import run_pipeline
import pandas as pd
       
LR_PATIENCE = 100
LR = 2e-3
SCHEDULER = 'ReduceLROnPlateau'
BATCH_SIZE = 64
DEVICE = 'cuda'
MAX_EPOCHS = 100
SAVE_PATH = 'TrainedModels/Github/DeepGCN'
DATA_FOLDER = "data/Github"
MODEL_TYPE = "DeepGCN"

n_filters = [32, 64, 128]
n_blocks = [4, 16, 32]
convs = ["mr", "gat"]
drop_outs = [0.2, 0.0]

RESULT_LIST = []
RESULT_COLUMNS = ["num_filters", "num_blocks", "conv", "dropout", "F1_score", "Accuracy"]

for n_filter in n_filters: 
    for n_block in n_blocks: 
        for conv in convs: 
            for dropout in drop_outs: 
                MODEL_CONFIG = {"n_filters":n_filter, "n_blocks":n_block, "conv": conv, "dropout":dropout}
                test_value, test_acc = run_pipeline(LR_PATIENCE, LR, SCHEDULER, 
                                                    BATCH_SIZE, DEVICE, MAX_EPOCHS, DATA_FOLDER, 
                                                    MODEL_TYPE, SAVE_PATH, MODEL_CONFIG)
                
                RESULT_LIST.append([n_filters, n_blocks, conv, dropout, test_value, test_acc])

result_df = pd.DataFrame(data = RESULT_LIST, columns = RESULT_COLUMNS)
result_df.to_csv("result/DeepGCN.csv")
