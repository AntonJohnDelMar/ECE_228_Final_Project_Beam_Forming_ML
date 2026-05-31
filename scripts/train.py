import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, random_split 

from dataset_handler import DatasetHandler 
from mlp import MLP 



def train(trainset): 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu"); 

    BATCH_SIZE = 10; 
    EPOCHS = 50; 
    LR = 1e-3; 

    NUM_SECTORS = 8; 
    NUM_CODEBOOKS = 3; 

    dataset = DatasetHandler(trainset); 

    train_size = int(0.8 * len(dataset)); 
    val_size = len(dataset) - train_size; 

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size]
    ); 

    train_loader = DataLoader(
        train_set,
        batch_size = BATCH_SIZE,
        shuffle = True
    ); 

    val_loader = DataLoader(
        val_set,
        batch_size = BATCH_SIZE,
        shuffle = False
    ); 


    model = MLP(
        input = 8,
        h1_dim = 128,
        h2_dim = 128,
        h3_dim = 64,
        out_1 = NUM_SECTORS,
        out_2 = NUM_CODEBOOKS
    ).to(DEVICE); 

    criterion = nn.CrossEntropyLoss(); 

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = LR
    ); 


    for epoch in range(EPOCHS):

        model.train(); 

        running_loss = 0.0; 

        for (
            x,
            tx0_sector,
            tx1_sector,
            tx0_codebook,
            tx1_codebook
        ) in train_loader:

            x = x.to(DEVICE); 

            tx0_sector = tx0_sector.to(DEVICE); 
            tx1_sector = tx1_sector.to(DEVICE); 

            tx0_codebook = tx0_codebook.to(DEVICE); 
            tx1_codebook = tx1_codebook.to(DEVICE); 

            optimizer.zero_grad(); 

            s0, s1, c0, c1 = model(x); 

            loss_s0 = criterion(s0, tx0_sector); 
            loss_s1 = criterion(s1, tx1_sector); 

            loss_c0 = criterion(c0, tx0_codebook); 
            loss_c1 = criterion(c1, tx1_codebook); 

            loss = (
                loss_s0
                + loss_s1
                + loss_c0
                + loss_c1
            ); 

            loss.backward(); 
            optimizer.step(); 

            running_loss += loss.item(); 

        avg_train_loss = running_loss / len(train_loader); 

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {avg_train_loss:.4f}"
        ); 

    torch.save(
        model.state_dict(),
        "beam_classifier.pth"
    ); 

    print("Model saved."); 