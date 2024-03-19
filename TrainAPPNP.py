from TrainingUtils import run_pipeline
import pandas as pd

LR_PATIENCE = 100
LR = 2e-3
SCHEDULER = 'ReduceLROnPlateau'
BATCH_SIZE = 64
DEVICE = 'cuda'
MAX_EPOCHS = 100
SAVE_PATH = 'TrainedModels/Github/APPNP'
DATA_FOLDER = "data/Github"
MODEL_TYPE = "APPNP"

hidden_features = [32, 64, 128]
num_iterations = [4, 16, 32]
alphas = [0.2, 0.05]
drop_outs = [0.2, 0.0] 

RESULT_LIST = []
RESULT_COLUMNS = ["hidden_feature", "num_iteration", "alpha", "dropout", "F1_score", "Accuracy"]

for h_feats in hidden_features: 
    for n_iter in num_iterations: 
        for alpha in alphas: 
            for dropout in drop_outs: 
                MODEL_CONFIG = {"h_feats":h_feats, "num_iterations":n_iter, 
                                "alpha":alpha, "dropout":dropout}
                test_value, test_acc = run_pipeline(LR_PATIENCE, LR, SCHEDULER, 
                                                    BATCH_SIZE, DEVICE, MAX_EPOCHS, DATA_FOLDER, 
                                                    MODEL_TYPE, SAVE_PATH, MODEL_CONFIG)
                
                RESULT_LIST.append([h_feats, n_iter, alpha, dropout, test_value, test_acc])

result_df = pd.DataFrame(data = RESULT_LIST, columns = RESULT_COLUMNS)
result_df.to_csv("result/APPNP.csv")
