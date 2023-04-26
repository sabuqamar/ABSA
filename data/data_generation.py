import pandas as pd
import xml.etree.ElementTree as ET

semeval_sentiments = ["positive", "neutral", "negative", "conflict", "none"]
semeval_label2id = {"positive": 0, "neutral": 1, "negative": 2, "conflict": 3, "none": 4}


def create_dataset(root, output_path):
	output = [["raw_text", "aspectTerms"]]

	for sentence in root:
		original_sentence = sentence.find("text").text
		term_polarities = []
		terms = sentence.find("aspectTerms")
		if terms is None:
			continue
		for term in sentence.find("aspectTerms"):
			termDict = {}
			termDict["term"] = term.attrib["term"]
			termDict["polarity"] = term.attrib["polarity"]
			term_polarities.append(termDict)
		output.append([original_sentence, term_polarities])
	output[1:] = sorted(output[1:], key=lambda el: el[0])
	df = pd.DataFrame(output)
	df.to_csv(output_path, sep=',', index=False, header=False, quotechar='"', quoting=1)


file_path = "Restaurants_Train_v2.xml"
with open(file_path, "r") as f:
	tree = ET.parse(f)
	create_dataset(tree.getroot(), "./Restaurants_Train_v2.csv")

file_path = "Restaurants_Test_Gold.xml"
with open(file_path, "r") as f:
	tree = ET.parse(f)
	create_dataset(tree.getroot(), "./Restaurants_Test_Gold.csv")