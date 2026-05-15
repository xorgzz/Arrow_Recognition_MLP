from gen_lib import gen_data_set
from mlp_utils import *

outs = ["Góra", "lewo", "prawo", "X"]



x1 = [[0,0,0,0,0,0,0,0],
[0,0,0,1,0,0,0,0],
[0,0,1,1,1,0,0,0],
[0,1,1,1,1,1,0,0],
[0,0,0,1,0,0,0,0],
[1,0,0,0,0,0,0,0],
[0,0,0,1,0,0,0,0],
[0,1,1,0,0,0,0,1]]



net = load_model("model_strzalek.pth")
result = predict(net, x1)

print(f"wynik: {outs[result]}")