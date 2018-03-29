from app.model.alpha_cnn_predict import LiteOCR
from app.model.preprocessor import Preprocessor as img_prep
from PIL import Image
import sys



def main(): 
	# prediction = ocr.predict(); 
	# ocr = LiteOCR(fn="app/model/alpha_weights.pkl")
	ocr = LiteOCR()
	pp = img_prep(fn="dataset.txt")

	# n is an array of pixel values between -2 and 2 I think
	if(len(sys.argv) < 2): 
		print "Must enter an image name.. temp0.jpg is T"
		exit(1)

	image = sys.argv[1]
	n = pp.preprocess(image)
	char_prediction= ocr.predict(n)


	print char_prediction


if __name__ == "__main__":
	main()