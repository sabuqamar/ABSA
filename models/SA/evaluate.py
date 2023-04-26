import tokenizer
import _pickle as cPickle 


def get_sent_mask(sent, terms):
	with open("../Checkpoints/sa_sent_word2id", "rb") as f:
		word2id = cPickle.load(f)
	sent_tokens = tokenize(sent)
	sent_inds = [word2id[x] if x in word2id else word2id["UNK"] for x in sent_tokens]
	masks = []
	for term in terms:
		target_tokens = tokenize(term)
		try:
			target_start = sent_tokens.index(target_tokens[0])
			target_end = sent_tokens[max(0, target_start - 1):].index(target_tokens[-1])  + max(0, target_start - 1)
		except:
			continue
		mask = [0] * len(sent_tokens)
		for m_i in range(target_start, target_end + 1):
			mask[m_i] = 1
		masks.append(mask)
	return sent_inds, masks

def evaluate(sent, terms):
	sent_inds, masks = get_sent_mask(sent, terms)
	model = torch.load("../Checkpoints/sa_sent_model")
	model.eval()

	print("transitions matrix ", model.inter_crf.transitions.data)
	# Initialize empty lists to store true labels and predicted labels
	labels = []
	for mask in masks:
		pred_label, best_seq = model.predict(sent_inds, mask)
		labels.append(pred_label)
	return labels

def tokenize(sent_str):
	sent_str = " ".join(sent_str.split("-"))
	sent_str = " ".join(sent_str.split("/"))
	sent_str = " ".join(sent_str.split("!"))
	return tokenizer.tokenize(sent_str)


