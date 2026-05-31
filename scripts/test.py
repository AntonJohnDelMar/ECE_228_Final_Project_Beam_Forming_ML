import torch 
from torch.utils.data import DataLoader 

from dataset_handler import DatasetHandler 
from mlp import MLP 



def test(testset): 
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ); 

    NUM_SECTORS = 16; 
    NUM_CODEBOOKS = 32; 

    dataset = DatasetHandler(testset); 

    loader = DataLoader(
        dataset,
        batch_size = 512,
        shuffle = False
    ); 

    model = MLP(
        input = 8,
        h1_dim = 128,
        h2_dim = 128,
        h3_dim = 64,
        out_1 = NUM_SECTORS,
        out_2 = NUM_CODEBOOKS
    ); 

    model.load_state_dict(
        torch.load(
            "beam_classifier.pth",
            map_location = DEVICE
        )
    ); 

    model.to(DEVICE); 
    model.eval(); 

    correct_s0 = 0; 
    correct_s1 = 0; 

    correct_c0 = 0; 
    correct_c1 = 0; 

    total = 0; 

    with torch.no_grad():

        for (
            x,
            tx0_sector,
            tx1_sector,
            tx0_codebook,
            tx1_codebook
        ) in loader:

            x = x.to(DEVICE); 

            s0, s1, c0, c1 = model(x); 

            pred_s0 = torch.argmax(s0, dim = 1); 
            pred_s1 = torch.argmax(s1, dim = 1); 

            pred_c0 = torch.argmax(c0, dim = 1); 
            pred_c1 = torch.argmax(c1, dim = 1); 

            correct_s0 += (
                pred_s0.cpu() == tx0_sector
            ).sum().item(); 

            correct_s1 += (
                pred_s1.cpu() == tx1_sector
            ).sum().item(); 

            correct_c0 += (
                pred_c0.cpu() == tx0_codebook
            ).sum().item(); 

            correct_c1 += (
                pred_c1.cpu() == tx1_codebook
            ).sum().item(); 

            total += x.size(0); 

    print(
        f"TX0 Sector Accuracy: "
        f"{100 * correct_s0 / total:.2f}%"
    ); 

    print(
        f"TX1 Sector Accuracy: "
        f"{100 * correct_s1 / total:.2f}%"
    ); 

    print(
        f"TX0 Codebook Accuracy: "
        f"{100 * correct_c0 / total:.2f}%"
    ); 

    print(
        f"TX1 Codebook Accuracy: "
        f"{100 * correct_c1 / total:.2f}%"
    ); 