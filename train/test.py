import torch
from pickle import load
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.encoder import CNNEncoder
from models.decoder import RNNDecoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 64
START_TOKEN = 62
END_TOKEN = 63
MAX_LEN = 150

# Load models
encoder = CNNEncoder().to(DEVICE)
decoder = RNNDecoder(vocab_size=VOCAB_SIZE).to(DEVICE)

checkpoint = torch.load("handtex_model.pt", map_location=DEVICE)  # load trained weights
encoder.load_state_dict(checkpoint["encoder"])  # all learned weights and biases
decoder.load_state_dict(checkpoint["decoder"])

encoder.eval()  # set to evaluation mode
decoder.eval()

images = torch.load("data/mathwriting/images_train.pt")
tokens = torch.load("data/mathwriting/tokens_train.pt")

img = images[:1].to(DEVICE)    # (1, 1, 128, 128)
gt_tokens = tokens[0]  # ground truth sequence

with open("notebooks/latex_tokenizer.pkl", "rb") as f:
    tokenizer = load(f)

inv_vocab = {v: k for k, v in tokenizer.word_index.items()}  # reverse mapping

def decode(seq):
    return "".join(inv_vocab.get(t, "") for t in seq)

# inference
with torch.no_grad():  
    features = encoder(img)
    input_token = torch.tensor([[START_TOKEN]], device=DEVICE) 
    hidden = None
    pred_tokens = []

    for _ in range(MAX_LEN):
        logits, hidden = decoder(input_token, features, hidden=hidden)
        next_token = logits.argmax(-1)  # index of highest logit
        token_id = next_token.item() # get scalar token ID
        if token_id == END_TOKEN:
            break
        pred_tokens.append(token_id)
        input_token = next_token  # feed predicted token back

print("GT:", decode(gt_tokens.tolist()))
print("PRED:", decode(pred_tokens))