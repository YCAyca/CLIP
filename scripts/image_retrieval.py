import torch
import clip
from PIL import Image
import os
from matplotlib import pyplot as plt 
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def create_gallery(gallery_paths):
    gallery = []

    for path in gallery_paths:
        img = Image.open(os.path.join(args.gallery_path,path))
        img_processed = preprocess(img).unsqueeze(0).to(device)
        image_embedding = model.encode_image(img_processed)
        gallery.append([image_embedding, os.path.join(args.gallery_path, path)])

    return gallery

def retrieval(args):
    gallery_paths = os.listdir(args.gallery_path)
    query_paths = os.listdir(args.query_path)

    print("--- Initalizing gallery ---")
    gallery = create_gallery(gallery_paths)

    for k, query_path in enumerate(query_paths):
        query_image = Image.open(os.path.join(args.query_path, query_path))
        img_processed = preprocess(query_image).unsqueeze(0).to(device)
        query_embedding = model.encode_image(img_processed)


        fig = plt.figure()
        plot_length = 10
        rank_list = []

        gallery_ax = fig.add_subplot(1,plot_length,1) #add query image in the left top place in plot
        gallery_ax.imshow(query_image)

        print(f"--- Starting image retrieval for query image: {query_path}")
        logit_scale = model.logit_scale.exp()
        query_normalized = query_embedding / query_embedding.norm(dim=1, keepdim=True)

        for item in gallery:
             # normalized features
            gallery_normalized = item[0] / item[0].norm(dim=1, keepdim=True)
            # cosine similarity as logits
            similarity_score = (logit_scale * query_normalized @ gallery_normalized.t()).item()
            similarity_score = round(similarity_score,3)
            rank_list.append([similarity_score, item[1]])  # add gallery image with its similarity score to this query image in ranking list 
    
        rank_list = sorted(rank_list, key=lambda x: x[0], reverse = True)

    
        for i in range(2,plot_length): 
            gallery_ax = fig.add_subplot(1,plot_length,i)
            img = Image.open(rank_list[i][1])
            gallery_ax.imshow(img)
            gallery_ax.set_title('%.1f'% rank_list[i][0], fontsize=8) #add similarity score as title
            gallery_ax.axis('off')
        plt.savefig(os.path.join(args.outDir, "plot_"+ str(k)+".jpg"))
        plt.close()



if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="CLIP Image Retriever")

    # Add arguments
    parser.add_argument(
        '--gallery-path', 
        type=str, 
        default="dataset/gallery/",
        help="Directory containing the gallery images"
    )
    parser.add_argument(
        '--query-path', 
        type=str, 
        default="dataset/query/",
        help="Directory containing the query images"
    )
    parser.add_argument(
        '--outDir', 
        type=str, 
        default="outputs/retrieval",
        help="Directory containing the output plots"
    )
  

    # Parse the arguments
    args = parser.parse_args()
    retrieval(args)