import time
import torch
from torch.autograd import Variable
from dataloader import load_data, Batch
from Config import config
from metrics import *
from tqdm import tqdm
from utils import *

# Load configuration settings
config = config()

MAX_SEQ_LENGTH = config.max_seq_length  # Maximum sequence length
DEVICE = config.device  # Computation device (CPU/GPU)
BOS_TOKEN = config.bos_token  # Begin of sequence
EOS_TOKEN = config.eos_token  # End of sequence

def subsequent_mask(size, batch_size):
    """ Generate a mask to prevent attending to future tokens. """
    attn_shape = (batch_size, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


def greedy_decode(model, src, src_mask, target, max_len, start_symbol=6):
    """ Greedy decoding function for sequence generation. """
    batch_size = src.shape[0]
    memory = model.encode(src, src_mask)  # Encode the source sequence

    # Initialize output sequence with start symbol
    decoded_seq = torch.ones(batch_size, 1).fill_(start_symbol).long().to(DEVICE)

    for i in range(max_len):
        out = model.decode(
            memory,
            src_mask,
            Variable(decoded_seq),
            Variable(subsequent_mask(decoded_seq.size(1), batch_size).long().to(DEVICE))
        )
        prob = model.generator(out[:, -1])  # Compute probabilities
        _, next_word = torch.max(prob, dim=1)  # Select most probable word
        next_word = next_word.data
        decoded_seq = torch.cat([decoded_seq, next_word[:, None]], dim=1)

    return decoded_seq


def evaluate(data, model):
    """ Evaluate the model on the test dataset. """
    evaluation = Score(Evaluator(7))

    with torch.no_grad():
        for data_batch, label_batch in data:
            batch = Batch(data_batch, label_batch)

            # Generate predictions using greedy decoding
            predictions = greedy_decode(
                model,
                batch.src.to(DEVICE),
                batch.src_mask.to(DEVICE),
                batch.trg_y.to(DEVICE),
                max_len=MAX_SEQ_LENGTH,
                start_symbol=BOS_TOKEN
            )

            for j in range(batch.src.shape[0]):
                predicted_seq = predictions.cpu().numpy()[j][1:]
                label_seq = batch.trg_y.cpu().numpy()[j][:-1]

                # Find end-of-sequence index and truncate sequences
                label_seq = label_seq[:find_stop_idx(label_seq)] - 1
                predicted_seq = predicted_seq[:find_stop_idx(predicted_seq)] - 1

                # Add sequences to evaluation metrics
                evaluation.add(label_seq, predicted_seq)

        # Print final evaluation score
        score_dict = evaluation.getScore()
        print(score_dict)
        score = 0.5 * ((score_dict['OA'] + score_dict['AA'] + score_dict['F1'] + score_dict['Kappa'])/4) + 0.5 * score_dict[
            'CT']
        return score


def find_stop_idx(seq):
    """ Find the index where the sequence ends (EOS token). """
    try:
        stop_label_idx = np.where(seq == EOS_TOKEN)[0][0]
    except:
        stop_label_idx = len(seq)
    return stop_label_idx


if __name__ == '__main__':

    config.area = 'Xiongan_New_Area'
    # Load dataset
    dataset = load_data(config.area)

    # Initialize and load model
    model = built(config)
    model.load_state_dict(torch.load(rf'SaveModel/Xiongan_New_Area_len_12_88.25.pt'))
    model.eval()

    print(">>>>>>> Start evaluation")
    eval_start_time = time.time()

    # Evaluate model on the test set
    evaluate(dataset[2], model)

    print(f"<<<<<< Evaluation completed in {time.time() - eval_start_time:.4f} seconds")
