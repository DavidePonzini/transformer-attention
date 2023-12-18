import model
import torch
from encoder_decoder import _subsequent_mask

m = model.make_model(11, 11, 1)
src = torch.arange(1, 11).unsqueeze(0)
src_mask = torch.ones(1, 1, src.size(1))

memory = m.encode(src, src_mask)
ys = torch.zeros(1, 1).type_as(src)


def generate_word():
    global ys

    out = m.decode(
        memory, src_mask, ys, _subsequent_mask(ys.size(1)).type_as(src.data)
    )

    prob = m.generator(out[:, -1])
    _, next_word = torch.max(prob, dim=1)
    next_word = next_word.data[0]
    ys = torch.cat([ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    print(f'Example Untrained Model Prediction: IN{src.squeeze().tolist()} - OUT{ys.squeeze().tolist()}')

if __name__ == '__main__':
    for _ in range(10):
        generate_word()