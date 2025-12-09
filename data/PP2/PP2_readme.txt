1. image.rar  is the urban photo folder with  110,688 pictures.
2. metadata.rar contains two files: votes.csv and  final_data.csv.

The votes.csv file contains 111,390 image URLs and 1,223,649 pairwise comparisons  from the Place Pulse 2.0 (PP2) dataset(http://pulse.media.mit.edu/data/). When we download the images, some URLs are unavailable. We finally
obtain 110,688 images and 1,208,808 pairwise comparisons, including 1,046,926 unequal ones and 161,882 equal ones in the final_data.csv file.

The final_data.csv has the following fields:

left_id: Serialized ID for the left image.
---
right_id: Serialized ID for the right image.
---
winner : One of {left,right,equal}, indicating which image was voted for in this comparison. 'equal' denotes that both were rated equally.
---
left_lat : Latitude for the left image.
left_long : Longitude for the left image.
---
right_lat : Latitude for the right image.
right_long : Longitude for the left image.
---
category : Category the vote belongs to, one of {safety, beautiful, lively, wealthy, boring, depressing}.

================== 
3. The dataset should be only used for non-commercial research and/or educational purposes. 


4. Besides some related works from http://pulse.media.mit.edu/papers/, the following work is also relevant to this dataset

author={W. {Min} and S. {Mei} and L. {Liu} and Y. {Wang} and S. {Jiang}}, 
journal={IEEE Transactions on Image Processing}, 
title={Multi-Task Deep Relative Attribute Learning for Visual Urban Perception}, 
year={2020}, 
volume={29}, 
number={1}, 
pages={657-669}, 
}







