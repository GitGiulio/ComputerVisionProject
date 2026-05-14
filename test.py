import torch
from shared_code import I_HAVE_A_THEORY, get_tensor_transform

MODEL_PATH = "models/model_kernel=[5,9,15]_64_3_64_1_wd0.0_do0.0.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Match filename: model_kernel=[5,9,19]_32_3_512_1_wd0.001_do0.0.pth
model = I_HAVE_A_THEORY(
    kernel_size=15,
    conv_filters=64,
    conv_layers=3,
    dense_neurons=64,
    dense_layers=1,
    dropout_rate=0.0,
).to(DEVICE)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

# Black image: 3 x 224 x 224
black = torch.zeros(1, 3, 224, 224).to(DEVICE)

with torch.no_grad():
    logit = model(black).reshape(-1)[0]
    p_dog = torch.sigmoid(logit).item()
    p_cat = 1.0 - p_dog

print(f"logit: {logit.item():.6f}")
print(f"p(Cat): {p_cat:.6f}")
print(f"p(Dog): {p_dog:.6f}")

if p_cat > p_dog:
    print("Prediction: Cat")
else:
    print("Prediction: Dog")