import pdb 
import os

from datasets import Dataset
import matplotlib.pyplot as plt
import mediapy as media
import torch
from human_shadow.utils.file_utils import get_parent_folder_of_package
# from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
# from transformers import AutoProcessor, GroundingDinoForObjectDetection
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
print("done with imports")

root_folder = get_parent_folder_of_package("human_shadow")
videos_folder = os.path.join(root_folder, "human_shadow/data/videos/demo_marion_calib_2/")
video_idx = 0
video_folder = os.path.join(videos_folder, str(video_idx))
video_idx = 0
video_path = os.path.join(video_folder, f"video_{video_idx}_L.mp4")
imgs_rgb = media.read_video(video_path)[:2]

dataset = Dataset.from_dict({"image": [img for img in imgs_rgb]})

print("done with dataset")

detector_id = "IDEA-Research/grounding-dino-tiny"
detector = pipeline(
            model=detector_id,
            task="zero-shot-object-detection",
            device="cuda",
            batch_size=2,
        )

print("done with pipeline")

for out in detector(KeyDataset(dataset, "image"), candidate_labels=["hand"], threshold=0.8):
    print(out)

# results = detector(dataset, candidate_labels=["hand"], threshold=0.8)

pdb.set_trace()


# checkpoint = "IDEA-Research/grounding-dino-tiny"
# # model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint)
# # processor = AutoProcessor.from_pretrained(checkpoint)

# processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
# model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")



# images = [img for img in imgs_rgb[:2]]
# text_queries = ["hand.", "hand."]
# inputs = processor(text=text_queries, images=images, return_tensors="pt")



# with torch.no_grad():
#     outputs = model(**inputs)
#     # target_sizes = torch.tensor([images[0].shape[::-1] for _ in range(len(images))])
#     target_sizes = [images[0].shape[::-1] for _ in range(len(images))]
#     # pdb.set_trace()
#     results = processor.image_processor.post_process_object_detection(outputs, threshold=0.8, target_sizes=target_sizes)[0]

#     pdb.set_trace()
    
#     # results = processor.post_process_grounded_object_detection(outputs, input_ids=box_threshold=0.8, target_sizes=target_sizes)[0]

# pdb.set_trace()
