import pandas as pd
import torch
from torch.utils.data import Dataset



class DatasetHandler(Dataset): 
    def __init__(self, csv_file):

        self.df = pd.read_csv(csv_file); 

        self.feature_cols = ["u1_distance_m", "u1_angle_deg", "u1_required_rate_bpshz", "u1_pilot_snr_db", "u2_distance_m", "u2_angle_deg", "u2_required_rate_bpshz", "u2_pilot_snr_db"]; 

        self.features = self.df[self.feature_cols].values.astype("float32"); 

        self.tx0_sector = self.df["tx0_sector_label"].values.astype("int64"); 
        self.tx1_sector = self.df["tx1_sector_label"].values.astype("int64"); 

        self.tx0_codebook = self.df["tx0_codebook_label"].values.astype("int64"); 
        self.tx1_codebook = self.df["tx1_codebook_label"].values.astype("int64"); 

    def __len__(self):
        return len(self.df); 

    def __getitem__(self, idx):

        x = torch.tensor(self.features[idx]); 

        return (x, torch.tensor(self.tx0_sector[idx]), torch.tensor(self.tx1_sector[idx]), torch.tensor(self.tx0_codebook[idx]), torch.tensor(self.tx1_codebook[idx])); 
