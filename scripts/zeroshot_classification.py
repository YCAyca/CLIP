import torch
import clip
from PIL import Image, UnidentifiedImageError
import os
from matplotlib import pyplot as plt 
import numpy as np
import argparse


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def classify(args):
    class_names = []
    file = open(args.classnames_file)
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        class_names.append(line)

    print("classnames: ", class_names)

    text = clip.tokenize(class_names).to(device)
    image_paths = os.listdir(args.input_dir)

    for i, image_path in enumerate(image_paths):
        print("Classifying image: ", image_path)
        fig = plt.figure()
        img = Image.open(os.path.join(args.input_dir, image_path))
        plt.imshow(img)
        img = preprocess(img).unsqueeze(0).to(device)
    
        with torch.no_grad():
            image_features = model.encode_image(img) # [1,512]
            text_features = model.encode_text(text)  # [class names x 512]

            logits_per_image, logits_per_text = model(img, text) # model(..) is equal to  (logit_scale * image_features @ text_features.T)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # Find the index of the maximum probability
        index = np.argmax(probs[0])

        # Get the maximum probability value
        max_val = probs[0][index]

        # Get the corresponding class name
        predicted_class = class_names[index]

    
        title = "Predicted Class: " + predicted_class + "\n Score: " + str(max_val)
    
        plt.title(title, fontsize = 10)
        # plt.show()
        plt.savefig(os.path.join(args.outDir, "plot_"+str(i)+".jpg"))
        plt.clf()
        plt.close("all")
        

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="CLIP Zero-shot classifier")

    # Add arguments
    parser.add_argument(
        '--input_dir', 
        type=str, 
        required="dataset/classification/", 
        help="Directory containing the input images"
    )
    parser.add_argument(
        '--classnames_file', 
        type=str, 
        default="dataset/classnames.txt", 
        help="Path to a file containing class names, one per line"
    )
    parser.add_argument(
        '--outDir', 
        type=str, 
        default="outputs/classification/",
        help="Directory containing the output plots"
    )


    # Parse the arguments
    args = parser.parse_args()
    classify(args)

