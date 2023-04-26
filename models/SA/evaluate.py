import os, sys, torch
import _pickle as cPickle
import tokenizer
import nltk

def get_sent_mask(sent, terms):
# 	Get the absolute path of the directory containing the script
	script_dir = os.path.dirname(os.path.abspath(__file__))

	# Append the directory containing the script to sys.path
	sys.path.append(script_dir)

	# Update the relative path of the file being opened
	file_path = os.path.join(script_dir,"../Checkpoints/sa_sent_word2id")
	with open(file_path, "rb") as f:
		word2id = cPickle.load(f)
	sent_tokens = tokenize(sent)
	sent_inds = [word2id[x] if x in word2id else word2id["UNK"] for x in sent_tokens]
	masks = []

	for term in terms:
		target_tokens = tokenize(term)

		target_start = sent_tokens.index(target_tokens[0])
		target_end = sent_tokens[max(0, target_start - 1):].index(target_tokens[-1])  + max(0, target_start - 1)

		mask = [0] * len(sent_tokens)
		for m_i in range(target_start, target_end + 1):
			mask[m_i] = 1
		masks.append(mask)
	return sent_inds, masks

def evaluate(sent, terms):
	sent_inds, masks = get_sent_mask(sent, terms)
	script_dir = os.path.dirname(os.path.abspath(__file__))

	# Append the directory containing the script to sys.path
	sys.path.append(script_dir)

	# Update the relative path of the file being opened
	file_path = os.path.join(script_dir,"../Checkpoints/sa_sent_model")
	model = torch.load(file_path)
	model.eval()

	# print("transitions matrix ", model.inter_crf.transitions.data)
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
	tokens = tokenizer.tokenize(sent_str)
	return [str(token) if not isinstance(token, str) else token for token in tokens]


